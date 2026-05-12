import copy

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    def forward(self, embed_reaction, embed_enzyme, enzyme_mask=None):
        Q1 = embed_reaction  # [batch_size, 1, 256]
        K2 = embed_enzyme    # [batch_size, n, 256]
        V2 = embed_enzyme    # [batch_size, n, 256]
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

        # 初始化权重
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)

    def conv_and_pool(self, x, conv):
        x = conv(x)  # [batch_size, 256, seq_len-k+1, 1]
        x = F.relu(x).squeeze(3)  # [batch_size, 256, seq_len-k+1]
        return x

    def forward(self, reactions, protein, enzyme_mask=None):
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
        feature = self.model_fc(reactions, protein, enzyme_mask=enzyme_mask)
        outC = self.classifier_layer(feature)  # [batch_size, 1]
        return (outC, feature)


class ActivityModel_Freeze(nn.Module):
    """纯冻结版：复用预训练压缩器 + 固定右路所有参数 + 保留detach()"""
    def __init__(self, kcat_km_model, Km_model, kcat_km_compress_state, km_compress_state, alpha=0.5, device="cuda:0"):
        super(ActivityModel_Freeze, self).__init__()
        self.alpha = alpha
        self.device = device

        # 纯固定
        self.kcat_km_model = kcat_km_model.to(device)
        self.Km_model = Km_model.to(device)

        # 左路模型（可训练）
        self.left_logkcat_model = Model_Regression().to(device)

        # 核心：分离压缩器（各自用自己的预训练权重）
        self.compressor_kcat_km = nn.Linear(935, 256).to(device)  # kcat/km专属
        self.compressor_km = nn.Linear(935, 256).to(device)  # Km专属
        self.compressor_left = nn.Linear(935, 256).to(device)  # 左路专属（新初始化或复用其一）
        # 加载各自的预训练压缩器权重
        self._load_separate_compressors(kcat_km_compress_state, km_compress_state)

    import copy  # 顶部需导入copy模块（若未导入）

    def _load_separate_compressors(self, kcat_km_compress, km_compress):
        """加载各自的预训练压缩器权重（右路压缩器固定，左路压缩器可训练），添加深复制+一致性校验"""
        # 1. kcat/km模型：深复制+校验
        # 深复制原始权重（避免共享内存，防止原模型权重被污染）
        kcat_km_compress_copy = copy.deepcopy(kcat_km_compress)
        # 加载复制后的权重
        self.compressor_kcat_km.weight.data.copy_(kcat_km_compress_copy["weight"])
        self.compressor_kcat_km.bias.data.copy_(kcat_km_compress_copy["bias"])
        # # 一致性校验：计算复制后与原始权重的绝对误差均值（应接近0）
        # weight_error = th.mean(th.abs(self.compressor_kcat_km.weight.data - kcat_km_compress["weight"]))
        # bias_error = th.mean(th.abs(self.compressor_kcat_km.bias.data - kcat_km_compress["bias"]))
        # if weight_error > 1e-6 or bias_error > 1e-6:
        #     raise ValueError(f"kcat/km压缩器权重复制失败！权重误差：{weight_error:.8f}，偏置误差：{bias_error:.8f}")
        # print(f"kcat/km压缩器权重复制校验通过（误差：{weight_error:.8f}）")

        # 2. Km模型：深复制+校验（逻辑同上）
        km_compress_copy = copy.deepcopy(km_compress)
        self.compressor_km.weight.data.copy_(km_compress_copy["weight"])
        self.compressor_km.bias.data.copy_(km_compress_copy["bias"])
        # 校验
        # weight_error_km = th.mean(th.abs(self.compressor_km.weight.data - km_compress["weight"]))
        # bias_error_km = th.mean(th.abs(self.compressor_km.bias.data - km_compress["bias"]))
        # if weight_error_km > 1e-6 or bias_error_km > 1e-6:
        #     raise ValueError(f"Km压缩器权重复制失败！权重误差：{weight_error_km:.8f}，偏置误差：{bias_error_km:.8f}")
        # print(f"Km压缩器权重复制校验通过（误差：{weight_error_km:.8f}）")

        # 3. 左路压缩器：新初始化（可训练，适配左路模型）
        self.compressor_left.weight.data.normal_(0, 0.01)
        self.compressor_left.bias.data.fill_(0.0)
        # 方案B：复用kcat/km的压缩器权重（需同样深复制+校验）
        # kcat_km_left_copy = copy.deepcopy(kcat_km_compress)
        # self.compressor_left.weight.data.copy_(kcat_km_left_copy["weight"])
        # self.compressor_left.bias.data.copy_(kcat_km_left_copy["bias"])
        # # 校验
        # left_weight_error = th.mean(th.abs(self.compressor_left.weight.data - kcat_km_compress["weight"]))
        # left_bias_error = th.mean(th.abs(self.compressor_left.bias.data - kcat_km_compress["bias"]))
        # if left_weight_error > 1e-6 or left_bias_error > 1e-6:
        #     raise ValueError(f"左路压缩器权重复制失败！权重误差：{left_weight_error:.8f}")
        # print(f"左路压缩器权重复制校验通过（误差：{left_weight_error:.8f}）")

    def forward(self, ezy_feats, sbt_feats, enzyme_mask=None):
        # 底物特征压缩（复用预训练权重）
        sbt_compressed = self.compressor_left(sbt_feats)  # [batch_size, 256]
        sbt_feats_seq = sbt_compressed.unsqueeze(1)  # [batch_size, 1, 256]

        # 左路预测
        pred_left_logkcat, _ = self.left_logkcat_model(sbt_feats_seq, ezy_feats, enzyme_mask)

        # 右路计算（带detach()，禁止梯度传播）
        sbt_kcat_km = self.compressor_kcat_km(sbt_feats).unsqueeze(1)
        pred_log_kcat_over_km, _ = self.kcat_km_model(sbt_kcat_km, ezy_feats, enzyme_mask)

        # 3. 右路Km：用自己的压缩器
        sbt_km = self.compressor_km(sbt_feats).unsqueeze(1)
        pred_log_km, _ = self.Km_model(sbt_km, ezy_feats, enzyme_mask)

        pred_right_logkcat = pred_log_kcat_over_km + pred_log_km

        # 加权融合（保留detach()）
        final_logkcat = self.alpha * pred_left_logkcat + (1 - self.alpha) * pred_right_logkcat.detach()
        return final_logkcat, pred_left_logkcat, pred_right_logkcat