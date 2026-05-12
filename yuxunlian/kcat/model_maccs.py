import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy


class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    def forward(self, embed_reaction, embed_enzyme, enzyme_mask=None):
        Q1 = embed_reaction  # [batch_size, 1, 256]
        K2 = embed_enzyme  # [batch_size, n, 256]
        V2 = embed_enzyme  # [batch_size, n, 256]
        embed_dim = embed_reaction.shape[-1]

        # 掩码处理
        attn_mask = (1 - enzyme_mask) * -10000.0 if enzyme_mask is not None else None
        if attn_mask is not None:
            attn_mask = attn_mask.to(Q1.device)

        # 第一阶段注意力
        K2_t = K2.permute(0, 2, 1)
        attn_scores = th.matmul(Q1, K2_t) / (embed_dim ** 0.5)
        if attn_mask is not None:
            attn_scores += attn_mask
        attn_weights_1 = F.softmax(attn_scores, dim=-1)
        updated_V2 = th.matmul(attn_weights_1, V2)
        updated_V2 = updated_V2 + embed_reaction  # Add
        updated_V2 = F.layer_norm(updated_V2, normalized_shape=[embed_dim])  # Norm

        # 第二阶段注意力
        Q2 = embed_enzyme
        K1 = embed_reaction
        V1 = embed_reaction
        K1_t = K1.permute(0, 2, 1)
        attn_scores2 = th.matmul(Q2, K1_t) / (embed_dim ** 0.5)
        if enzyme_mask is not None:
            attn_mask2 = (1 - enzyme_mask.permute(0, 2, 1)) * -10000.0
            attn_scores2 += attn_mask2
        attn_weights2 = F.softmax(attn_scores2, dim=-1)
        updated_V1 = th.matmul(attn_weights2, V1)
        updated_V1 = updated_V1 + embed_enzyme  # Add
        updated_V1 = F.layer_norm(updated_V1, normalized_shape=[embed_dim])  # Norm

        # 最大池化+拼接
        updated_V1 = updated_V1.permute(0, 2, 1)
        updated_V1 = F.max_pool1d(updated_V1, kernel_size=updated_V1.shape[-1])
        updated_V1 = updated_V1.permute(0, 2, 1)
        out = th.cat((updated_V2, updated_V1), dim=2).squeeze(1)  # [batch_size, 512]

        return out


class ModelFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = 1024
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, self.hidden1)) for k in self.filter_sizes
        ])
        self.cross_attention = CrossAttention()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)

        # 🔴 核心修改：Kcat模型内置底物压缩层（768→256，适配molt5_feat）
        self.sbt_compress = nn.Linear(167, 256)  # 原935→768，匹配拆分后的molt5_feat

        # 初始化权重
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)
        self.sbt_compress.weight.data.normal_(0, 0.01)
        self.sbt_compress.bias.data.fill_(0.0)

    def conv_and_pool(self, x, conv):
        x = conv(x)  # [batch_size, 256, seq_len-k+1, 1]
        x = F.relu(x).squeeze(3)  # [batch_size, 256, seq_len-k+1]
        return x

    def forward(self, reactions, protein, enzyme_mask=None):
        # 🔴 内置底物压缩：768→256
        reactions = self.sbt_compress(reactions)  # [batch_size, 256]
        reactions = reactions.unsqueeze(1)  # [batch_size, 1, 256]

        # 酶特征处理
        out = protein.unsqueeze(1)  # [batch_size, 1, seq_len, 1024]
        conv_outputs = [self.conv_and_pool(out, conv) for conv in self.convs]
        out = th.cat(conv_outputs, 2)  # [batch_size, 256, n]

        # 掩码适配
        if enzyme_mask is not None:
            batch_size, _, seq_len = enzyme_mask.shape
            n = out.shape[2]
            conv_mask = th.zeros((batch_size, 1, n), device=enzyme_mask.device)
            start = 0
            for k in self.filter_sizes:
                Lk = seq_len - k + 1
                conv_mask[:, :, start:start + Lk] = enzyme_mask[:, :, k - 1:seq_len]
                start += Lk
            enzyme_mask = conv_mask

        # 交叉注意力
        protein1 = out.permute(0, 2, 1)  # [batch_size, n, 256]
        aggreX1 = self.cross_attention(reactions, protein1, enzyme_mask=enzyme_mask)

        # 全连接层
        output = self.drop1(F.leaky_relu(self.fc1(aggreX1)))
        output = self.drop2(F.leaky_relu(self.fc2(output)))
        return output


class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression, self).__init__()
        self.model_fc = ModelFC()
        self.classifier_layer = nn.Linear(64, 1)

        # 初始化权重
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, reactions, protein, enzyme_mask=None):
        # 底物压缩已在model_fc内部完成
        feature = self.model_fc(reactions, protein, enzyme_mask=enzyme_mask)
        outC = self.classifier_layer(feature)  # [batch_size, 1]
        return (outC, feature)


class ActivityModel_Freeze(nn.Module):
    """纯冻结版：复用预训练Kcat/Km模型 + 内置压缩层 + 左路可训练"""

    def __init__(self, kcat_km_model, Km_model, alpha=0.5, device="cuda:0"):
        super(ActivityModel_Freeze, self).__init__()
        self.alpha = alpha
        self.device = device

        # 固定右路预训练模型
        self.kcat_km_model = kcat_km_model.to(device)
        self.Km_model = Km_model.to(device)

        # 左路Kcat模型（内置压缩层）
        self.left_logkcat_model = Model_Regression().to(device)

        # 冻结右路所有参数
        for param in self.kcat_km_model.parameters():
            param.requires_grad = False
        for param in self.Km_model.parameters():
            param.requires_grad = False

    def forward(self, ezy_feats, sbt_feats, enzyme_mask=None):
        # 左路预测（内置768→256压缩）
        pred_left_logkcat, _ = self.left_logkcat_model(sbt_feats, ezy_feats, enzyme_mask)

        # 右路计算（内置压缩，detach禁止梯度）
        pred_log_kcat_over_km, _ = self.kcat_km_model(sbt_feats, ezy_feats, enzyme_mask)
        pred_log_km, _ = self.Km_model(sbt_feats, ezy_feats, enzyme_mask)
        pred_right_logkcat = pred_log_kcat_over_km + pred_log_km

        # 加权融合
        final_logkcat = self.alpha * pred_left_logkcat + (1 - self.alpha) * pred_right_logkcat.detach()
        return final_logkcat, pred_left_logkcat, pred_right_logkcat