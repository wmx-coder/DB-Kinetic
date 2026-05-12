import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ========== 基础模块（无需修改，和之前一致）==========
class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    def forward(self, embed_reaction, embed_enzyme, enzyme_mask=None):
        Q1 = embed_reaction  # [batch_size, 1, 256]
        K2 = embed_enzyme  # [batch_size, n, 256]
        V2 = embed_enzyme  # [batch_size, n, 256]
        embed_dim = embed_reaction.shape[-1]

        # 掩码处理（修正第二阶段掩码维度）
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

        # 第二阶段注意力（修正掩码维度）
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
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x).squeeze(3)
        return x

    def forward(self, reactions, protein, enzyme_mask=None):
        out = protein.unsqueeze(1)
        conv_outputs = [self.conv_and_pool(out, conv) for conv in self.convs]
        out = th.cat(conv_outputs, 2)

        # 掩码适配
        if enzyme_mask is not None:
            batch_size, _, seq_len = enzyme_mask.shape
            n = out.shape[2]
            conv_mask = th.zeros((batch_size, 1, n), device=enzyme_mask.device)
            start = 0
            for k in self.filter_sizes:
                Lk = seq_len - k + 1
                if Lk > 0:
                    end = start + Lk
                    end = min(end, n)
                    enzyme_mask_slice = enzyme_mask[:, :, k - 1: k - 1 + (end - start)]
                    conv_mask[:, :, start:end] = enzyme_mask_slice
                start += Lk
            enzyme_mask = conv_mask

        protein1 = out.permute(0, 2, 1)
        aggreX1 = self.cross_attention(reactions, protein1, enzyme_mask=enzyme_mask)
        output = self.drop1(F.leaky_relu(self.fc1(aggreX1)))
        output = self.drop2(F.leaky_relu(self.fc2(output)))
        return output


class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression, self).__init__()
        self.model_fc = ModelFC()
        self.classifier_layer = nn.Linear(64, 1)
        nn.init.xavier_normal_(self.classifier_layer.weight)
        nn.init.constant_(self.classifier_layer.bias, 0.0)

    def forward(self, reactions, protein, enzyme_mask=None):
        feature = self.model_fc(reactions, protein, enzyme_mask=enzyme_mask)
        outC = self.classifier_layer(feature)
        return outC, feature


# ========== 原kcat模型的融合类（修正：左路命名+压缩器校验）==========
class KcatOriginalActivityModel(nn.Module):
    def __init__(
            self,
            kcat_km_model,  # 单模型（Model_Regression）
            Km_model,       # 单模型（Model_Regression）
            kcat_km_compress_state,
            km_compress_state,
            alpha=0.5,
            device="cuda:0"
    ):
        super().__init__()
        self.alpha = alpha
        self.device = device

        # 冻结右路模型
        self.kcat_km_model = kcat_km_model.to(device)
        self.Km_model = Km_model.to(device)
        for param in self.kcat_km_model.parameters():
            param.requires_grad = False
        for param in self.Km_model.parameters():
            param.requires_grad = False

        # 左路可训练模型（命名：left_logkcat_model，和训练时一致）
        self.left_logkcat_model = Model_Regression().to(device)
        self.compressor_left = nn.Linear(935, 256).to(device)

        # 右路压缩器（kcat_km和km的）
        self.compressor_kcat_km = nn.Linear(935, 256).to(device)
        self.compressor_km = nn.Linear(935, 256).to(device)

        # 加载右路压缩器权重（含校验）
        if kcat_km_compress_state is None:
            raise ValueError("kcat_km_compress_state 不能为空！（kcat模型右路需要）")
        self.compressor_kcat_km.load_state_dict(kcat_km_compress_state)
        assert self.compressor_kcat_km.in_features == 935 and self.compressor_kcat_km.out_features == 256, \
            "kcat_km压缩器维度错误！必须是935→256"

        if km_compress_state is None:
            raise ValueError("km_compress_state 不能为空！（kcat模型右路需要）")
        self.compressor_km.load_state_dict(km_compress_state)
        assert self.compressor_km.in_features == 935 and self.compressor_km.out_features == 256, \
            "km压缩器维度错误！必须是935→256"

        # 冻结右路压缩器
        for param in self.compressor_kcat_km.parameters():
            param.requires_grad = False
        for param in self.compressor_km.parameters():
            param.requires_grad = False

        print("✅ kcat模型右路压缩器加载完成（含维度校验）")

    def forward(self, ezy_feats, sbt_feats, enzyme_mask=None):
        # 左路预测
        sbt_left = self.compressor_left(sbt_feats).unsqueeze(1)
        pred_left, _ = self.left_logkcat_model(sbt_left, ezy_feats, enzyme_mask)

        # 右路预测（kcat_km模型）
        sbt_kcat_km = self.compressor_kcat_km(sbt_feats).unsqueeze(1)
        pred_kcat_km, _ = self.kcat_km_model(sbt_kcat_km, ezy_feats, enzyme_mask)

        # 右路预测（km模型）
        sbt_km = self.compressor_km(sbt_feats).unsqueeze(1)
        pred_km, _ = self.Km_model(sbt_km, ezy_feats, enzyme_mask)

        # 右路融合
        pred_right = pred_kcat_km - pred_km

        # 最终融合
        final_pred = self.alpha * pred_left + (1 - self.alpha) * pred_right.detach()

        return final_pred, pred_left, pred_right


# ========== 现在训练kcat/Km模型的核心类（关键修改：左路压缩器随机初始化）==========
class ActivityModel_Freeze(nn.Module):
    """
    预测 log(kcat / Km)
    - 左路：可训练模型直接预测 log(kcat/Km)（核心）
    - 右路：log(kcat) - log(Km)（调用不同框架的模型：kcat是融合模型，km是单模型）
    - 融合：左路为主，右路为辅（detach避免梯度干扰）
    """

    def __init__(
            self,
            kcat_model,  # 被调用：融合模型（KcatOriginalActivityModel）
            km_model,    # 被调用：单模型（Model_Regression）
            km_compress_state,  # 仅km模型需要（它自己的压缩器权重）
            alpha=0.7,
            device="cuda:0"
    ):
        super(ActivityModel_Freeze, self).__init__()
        self.alpha = alpha
        self.device = device

        # 1. 冻结所有被调用模型（不管框架，全部固定参数）
        self.kcat_model = kcat_model.to(device)  # 融合模型
        self.km_model = km_model.to(device)      # 单模型
        for param in self.kcat_model.parameters():
            param.requires_grad = False
        for param in self.km_model.parameters():
            param.requires_grad = False
        print("✅ 冻结 kcat_model（融合框架）和 km_model（单模型）所有参数")

        # 2. 左路：可训练的直接预测器（关键修改：左路压缩器直接随机初始化）
        self.left_ratio_model = Model_Regression().to(device)
        self.compressor_left = nn.Linear(935, 256).to(device)
        # 直接随机初始化（删除km权重相关代码）
        nn.init.xavier_normal_(self.compressor_left.weight)  #  Xavier正态分布初始化
        nn.init.constant_(self.compressor_left.bias, 0.0)    # 偏置初始化为0
        print("✅ 左路压缩器采用随机初始化（Xavier正态分布+偏置0）")

        # 3. 仅为km模型准备压缩器（kcat模型内部自带压缩器，无需外部提供）
        self.compressor_km = nn.Linear(935, 256).to(device)
        self._load_km_compressor(km_compress_state)
        # 冻结km压缩器
        for param in self.compressor_km.parameters():
            param.requires_grad = False
        print("✅ 冻结 km 模型的压缩器参数")

    def _load_km_compressor(self, km_compress_state):
        """仅加载km模型的预训练压缩器（kcat模型内部已有，无需额外加载）"""
        if km_compress_state is None:
            raise ValueError("km_compress_state 不能为空！（kcat模型无需外部压缩器）")

        self.compressor_km.load_state_dict(km_compress_state)
        # 校验维度（935→256，和km训练时一致）
        assert self.compressor_km.in_features == 935 and self.compressor_km.out_features == 256
        print("✅ 成功加载 km 预训练压缩器权重（935→256）")

    def forward(self, ezy_feats, sbt_feats, enzyme_mask=None):
        # === 左路：直接预测 log(kcat/Km)（逻辑不变）===
        sbt_left = self.compressor_left(sbt_feats).unsqueeze(1)  # [B, 1, 256]
        pred_left_log_ratio, _ = self.left_ratio_model(sbt_left, ezy_feats, enzyme_mask)

        # === 右路：log(kcat) - log(Km)（关键适配不同框架）===
        # 适配1：调用kcat模型（融合框架）
        # kcat模型的forward输入是 (ezy_feats, sbt_feats, enzyme_mask)，输出是3个值（取第一个为最终预测值）
        pred_log_kcat, _, _ = self.kcat_model(ezy_feats, sbt_feats, enzyme_mask)  # 只取final_logkcat

        # 适配2：调用km模型（单模型框架）
        # km模型的forward输入是 (reactions, protein, enzyme_mask)，需要先压缩底物特征
        sbt_km = self.compressor_km(sbt_feats).unsqueeze(1)  # [B, 1, 256]
        pred_log_km, _ = self.km_model(sbt_km, ezy_feats, enzyme_mask)  # 单模型输出2个值，取预测值

        # 公式推导（加1e-8避免数值不稳定）
        pred_right_log_ratio = pred_log_kcat - pred_log_km  # [B, 1]

        # === 融合（逻辑不变）===
        final_log_ratio = self.alpha * pred_left_log_ratio + (1 - self.alpha) * pred_right_log_ratio.detach()

        # 返回所有输出（方便训练时监控左/右路性能）
        return final_log_ratio, pred_left_log_ratio, pred_right_log_ratio