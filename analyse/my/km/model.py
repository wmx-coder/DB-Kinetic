import torch
from torch import nn
import torch.nn.functional as F


DEBUG = True   # ⭐总开关


def dbg(name, x):
    if DEBUG:
        print(f"\n[{name}]")
        print("shape:", tuple(x.shape))
        print("mean:", x.mean().item(), "std:", x.std().item())


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embed_reaction, embed_enzyme, enzyme_mask=None):

        Q1 = embed_reaction
        K2 = embed_enzyme
        V2 = embed_enzyme
        dim = Q1.shape[-1]

        dbg("CA input reaction", Q1)
        dbg("CA input enzyme", K2)

        # mask
        if enzyme_mask is not None:
            attn_mask = (1 - enzyme_mask) * -10000.0
            attn_mask = attn_mask.to(Q1.device)
        else:
            attn_mask = None

        # attention 1
        attn_scores = torch.matmul(Q1, K2.permute(0,2,1)) / (dim ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        dbg("attn_scores1", attn_scores)

        attn_weights_1 = F.softmax(attn_scores, dim=-1)
        dbg("attn_weights1", attn_weights_1)

        updated_V2 = torch.matmul(attn_weights_1, V2)
        updated_V2 = updated_V2 + embed_reaction
        updated_V2 = F.layer_norm(updated_V2, [dim])

        dbg("updated_V2", updated_V2)

        # attention 2
        Q2 = embed_enzyme
        K1 = embed_reaction
        V1 = embed_reaction

        attn_scores2 = torch.matmul(Q2, K1.permute(0,2,1)) / (dim ** 0.5)

        if enzyme_mask is not None:
            attn_scores2 = attn_scores2 + (1 - enzyme_mask.permute(0,2,1)) * -10000.0

        dbg("attn_scores2", attn_scores2)

        attn_weights2 = F.softmax(attn_scores2, dim=-1)
        updated_V1 = torch.matmul(attn_weights2, V1)

        updated_V1 = updated_V1 + embed_enzyme
        updated_V1 = F.layer_norm(updated_V1, [dim])

        updated_V1 = updated_V1.permute(0,2,1)
        updated_V1 = F.max_pool1d(updated_V1, updated_V1.shape[-1])
        updated_V1 = updated_V1.permute(0,2,1)

        dbg("updated_V1", updated_V1)

        out = torch.cat([updated_V2, updated_V1], dim=-1).squeeze(1)

        dbg("CA output", out)

        return out


class ModelFC(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden1 = 1024
        self.filter_sizes = (2,3,4)
        self.num_filters = 256

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 256, (k, 1024))
            for k in self.filter_sizes
        ])

        self.cross_attention = CrossAttention()

        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,64)

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.2)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x).squeeze(3)
        return x

    def forward(self, reactions, protein, enzyme_mask=None):

        dbg("RAW protein", protein)

        out = protein.unsqueeze(1)

        conv_outputs = [self.conv_and_pool(out, c) for c in self.convs]
        out = torch.cat(conv_outputs, 2)

        dbg("CNN out", out)

        # mask fix
        if enzyme_mask is not None:
            batch, _, seq = enzyme_mask.shape
            n = out.shape[2]

            conv_mask = torch.zeros((batch,1,n), device=out.device)

            start = 0
            for k in self.filter_sizes:
                Lk = seq - k + 1
                conv_mask[:,:,start:start+Lk] = enzyme_mask[:,:,k-1:seq]
                start += Lk

            enzyme_mask = conv_mask

        dbg("mask after conv", enzyme_mask)

        protein1 = out.permute(0,2,1)

        out = self.cross_attention(reactions, protein1, enzyme_mask)

        x = self.drop1(F.leaky_relu(self.fc1(out)))
        x = self.drop2(F.leaky_relu(self.fc2(x)))

        dbg("FC output", x)

        return x


class Model_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_fc = ModelFC()
        self.classifier_layer = nn.Linear(64,1)

    def forward(self, reactions, protein, enzyme_mask=None):
        feat = self.model_fc(reactions, protein, enzyme_mask)
        out = self.classifier_layer(feat)

        dbg("FINAL OUTPUT", out)

        return out, feat