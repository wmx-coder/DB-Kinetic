import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    def forward(self, embed_reaction, embed_enzyme, enzyme_mask=None):
        Q1 = embed_reaction  # [batch_size, 1, 256]
        K2 = embed_enzyme  # [batch_size, n, 256]
        V2 = embed_enzyme  # [batch_size, n, 256]
        embed_dim = embed_reaction.shape[-1]

        # 掩码处理
        if enzyme_mask is not None:
            attn_mask = (1 - enzyme_mask) * -10000.0
            attn_mask = attn_mask.to(Q1.device)
        else:
            attn_mask = None

        # 第一阶段注意力计算
        K2_t = K2.permute(0, 2, 1)  # [batch_size, 256, n]
        attn_scores = torch.matmul(Q1, K2_t) / (embed_dim ** 0.5)  # [batch_size, 1, n]

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights_1 = F.softmax(attn_scores, dim=-1)  # [batch_size, 1, n]
        updated_V2 = torch.matmul(attn_weights_1, V2)  # [batch_size, 1, 256]

        # Add & Norm
        updated_V2 = updated_V2 + embed_reaction
        updated_V2 = F.layer_norm(updated_V2, normalized_shape=[embed_dim])

        # 第二阶段注意力计算
        Q2 = embed_enzyme  # [batch_size, n, 256]
        K1 = embed_reaction  # [batch_size, 1, 256]
        V1 = embed_reaction  # [batch_size, 1, 256]

        K1_t = K1.permute(0, 2, 1)  # [batch_size, 256, 1]
        attn_scores2 = torch.matmul(Q2, K1_t) / (embed_dim ** 0.5)  # [batch_size, n, 1]

        if enzyme_mask is not None:
            attn_mask2 = (1 - enzyme_mask.permute(0, 2, 1)) * -10000.0
            attn_scores2 = attn_scores2 + attn_mask2

        attn_weights2 = F.softmax(attn_scores2, dim=-1)  # [batch_size, n, 1]
        updated_V1 = torch.matmul(attn_weights2, V1)  # [batch_size, n, 256]

        # Add & Norm
        updated_V1 = updated_V1 + embed_enzyme
        updated_V1 = F.layer_norm(updated_V1, normalized_shape=[embed_dim])

        # 最大池化
        updated_V1 = updated_V1.permute(0, 2, 1)  # [batch_size, 256, n]
        updated_V1 = F.max_pool1d(updated_V1, kernel_size=updated_V1.shape[-1])
        updated_V1 = updated_V1.permute(0, 2, 1)  # [batch_size, 1, 256]

        # 拼接输出
        out = torch.cat((updated_V2, updated_V1), dim=2).squeeze(1)  # [batch_size, 512]
        return out


class ModelFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = 1024
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, self.hidden1))
            for k in self.filter_sizes
        ])
        self.cross_attention = CrossAttention()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 256)  # 匹配CrossAttention输出的512维
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(256, 64)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)

    def conv_and_pool(self, x, conv):
        x = conv(x)  # [batch_size, 256, seq_len-filter_size+1, 1]
        x = F.relu(x).squeeze(3)  # [batch_size, 256, seq_len-filter_size+1]
        return x

    def forward(self, reactions, protein, enzyme_mask=None):
        # 1. 酶特征添加通道维度
        out = protein.unsqueeze(1)  # [batch_size, 1, seq_len, 1024]

        # 2. 多卷积核并行计算
        conv_outputs = [self.conv_and_pool(out, conv) for conv in self.convs]

        # 3. 拼接卷积结果
        out = torch.cat(conv_outputs, 2)  # [batch_size, 256, n]，n=各卷积输出长度之和

        # 4. 掩码转换（适配卷积后长度）
        if enzyme_mask is not None:
            batch_size, _, seq_len = enzyme_mask.shape
            n = out.shape[2]
            conv_mask = torch.zeros((batch_size, 1, n), device=enzyme_mask.device)
            start = 0
            for k in self.filter_sizes:
                Lk = seq_len - k + 1
                conv_mask[:, :, start:start + Lk] = enzyme_mask[:, :, k - 1:seq_len]
                start += Lk
            enzyme_mask = conv_mask

        # 5. 转置适配注意力层
        protein1 = out.permute(0, 2, 1)  # [batch_size, n, 256]

        # 6. 交叉注意力计算
        aggreX1 = self.cross_attention(reactions, protein1, enzyme_mask=enzyme_mask)

        # 7. 全连接层
        output = self.drop1(F.leaky_relu(self.fc1(aggreX1)))  # [batch_size, 256]
        output = self.drop2(F.leaky_relu(self.fc2(output)))  # [batch_size, 64]
        return output


class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression, self).__init__()
        self.model_fc = ModelFC()
        self.classifier_layer = nn.Linear(64, 1)  # 匹配ModelFC输出的64维
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, reactions, protein, enzyme_mask=None):
        # 特征提取
        feature = self.model_fc(reactions, protein, enzyme_mask=enzyme_mask)
        # 回归预测
        outC = self.classifier_layer(feature)  # [batch_size, 1]
        return (outC, feature)