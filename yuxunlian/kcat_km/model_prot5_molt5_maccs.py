import os
import torch as th
import torch.nn as nn


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

class ActivityModel(nn.Module):
    def __init__(self, kcat_model, km_model, rate=0.0, alpha=0.5, device="cuda:0"):
        super(ActivityModel, self).__init__()
        self.alpha = alpha
        self.device = device

        # 新的kcat/Km子模型（需要256维底物特征）
        self.kcat_model = kcat_model.to(device)
        self.km_model = km_model.to(device)

        # 新增：底物特征压缩器（935→256），用于适配kcat/Km模型的输入
        self.substrate_compressor = nn.Sequential(
            nn.Linear(935, 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # 压缩到子模型需要的256维
        ).to(device)

        # 活性2的计算模块（直接使用原始935维底物特征，无需转换）
        self.prot_norm = nn.BatchNorm1d(1024).to(device)
        self.molt5_norm = nn.BatchNorm1d(768).to(device)  # 935=768+167（原始拆分）
        self.decoder = nn.Sequential(
            nn.Linear(1959, 256),  # 1024+768+167=1959（原始维度不变）
            nn.BatchNorm1d(256),
            nn.Dropout(p=rate),
            nn.ReLU()
        ).to(device)
        self.attn = nn.Sequential(nn.Linear(256, 256), nn.Softmax(dim=1)).to(device)
        self.out = nn.Linear(256, 1).to(device)

    def forward(self, enzyme_feats, substrate_feats, enzyme_mask):
        # ------------------------- 输入格式 -------------------------
        # enzyme_feats: (batch, max_seq_len, 1024) （带序列的酶特征）
        # substrate_feats: (batch, 935) （原始底物特征，无需序列维度）
        # enzyme_mask: (batch,1,max_seq_len) （酶掩码）

        # ------------------------- 步骤1：底物特征压缩（适配kcat/Km模型） -------------------------
        # 压缩935维→256维，并增加子模型需要的序列维度（1维）
        substrate_compressed = self.substrate_compressor(substrate_feats)  # (batch,256)
        substrate_feats_seq = substrate_compressed.unsqueeze(1)  # (batch,1,256)

        # ------------------------- 步骤2：调用kcat/Km模型 -------------------------
        pred_kcat, _ = self.kcat_model(
            reactions=substrate_feats_seq,  # (batch,1,256) 适配子模型
            protein=enzyme_feats,           # (batch,max_seq_len,1024) 直接传入
            enzyme_mask=enzyme_mask         # (batch,1,max_seq_len) 直接传入
        )
        pred_km, _ = self.km_model(
            reactions=substrate_feats_seq,
            protein=enzyme_feats,
            enzyme_mask=enzyme_mask
        )
        pred_activity_1 = pred_kcat - pred_km  # 基础活性

        # ------------------------- 步骤3：计算活性2（直接用原始特征，无需转换） -------------------------
        # 1. 酶特征转换：(batch, max_seq_len, 1024) → (batch,1024)（池化降维）
        ezy_feats_flat = enzyme_feats.mean(dim=1)  # 按序列维度平均
        ezy_feats_norm = self.prot_norm(ezy_feats_flat)

        # 2. 底物特征直接拆分（原始935维，无需扩展）
        molt5_feats_norm = self.molt5_norm(substrate_feats[:, :768])  # 前768维
        macc_feats = substrate_feats[:, 768:]  # 后167维

        # 3. 复用原活性2逻辑
        cplx_feats = th.cat([ezy_feats_norm, molt5_feats_norm, macc_feats], axis=1)
        feats = self.decoder(cplx_feats)
        attn_score = self.attn(feats)
        attn_feats = attn_score * feats
        pred_activity_2 = self.out(attn_feats)

        # ------------------------- 步骤4：融合活性 -------------------------
        pred_activity = pred_activity_1.detach() * (1 - self.alpha) + pred_activity_2 * self.alpha

        return pred_kcat, pred_km, pred_activity