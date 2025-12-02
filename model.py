import torch
import torch.nn as nn
import torch.nn.functional as F


# UNet encoder
class UNetEncoder1D(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, stride=2),  # L -> L/2
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1, stride=2), # L/2 -> L/4
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim*2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1, stride=2), # L/4 -> L/8
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim*4),
        )

    def forward(self, x):
        f1 = self.layer1(x)  # (B, 64, L/2)
        f2 = self.layer2(f1) # (B, 128, L/4)
        f3 = self.layer3(f2) # (B, 256, L/8)
        return [f1, f2, f3]


# Cross-Attention
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)

    def forward(self, A, B, return_attn=False):
        out, attn_weights = self.attn(A, B, B, need_weights=True)
        if return_attn:
            return out, attn_weights
        return out


# Self-attention Transformer block
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# Level embedding for Transformer
class LevelEmbedding(nn.Module):
    def __init__(self, num_levels=3, dim=256):
        super().__init__()
        self.level_emb = nn.Embedding(num_levels, dim)

    def forward(self, x, level_idx):
        emb = self.level_emb(torch.tensor(level_idx, device=x.device))
        return x + emb


# Cross-level fusion (learnable weighted average)
class CrossLevelFusion(nn.Module):
    def __init__(self, num_levels=3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_levels) / num_levels)  # start with equal weight

    def forward(self, t_list):
        stacked = torch.stack(t_list, dim=0)  # (num_levels, B, C)
        w = torch.softmax(self.weights, dim=0).view(-1,1,1)
        fused = (stacked * w).sum(dim=0)  # (B, C)
        return fused


# Final Model
class HypertensionPredictor(nn.Module):
    def __init__(self, dna_dim, c1_dim, c2_dim, c3_dim, hidden_dim=256):
        super().__init__()

        # UNet Encoders
        self.encoder_dna = UNetEncoder1D(dna_dim)
        self.encoder_c1 = UNetEncoder1D(c1_dim)
        self.encoder_c2 = UNetEncoder1D(c2_dim)
        self.encoder_c3 = UNetEncoder1D(c3_dim)

        self.level_emb1 = LevelEmbedding(num_levels=3, dim=64)   # f1
        self.level_emb2 = LevelEmbedding(num_levels=3, dim=128)  # f2
        self.level_emb3 = LevelEmbedding(num_levels=3, dim=256)  # f3

        # Cross-Attention per layer
        self.cross1 = CrossAttention(64)
        self.cross2 = CrossAttention(128)
        self.cross3 = CrossAttention(256)

        # Projection to hidden_dim
        self.proj = nn.Linear(64+128+256, hidden_dim)

        # Self-attention Transformers per sequence
        self.trans1 = SelfAttentionBlock(hidden_dim)
        self.trans2 = SelfAttentionBlock(hidden_dim)
        self.trans3 = SelfAttentionBlock(hidden_dim)

        # Level embedding
        self.level_emb = LevelEmbedding(num_levels=3, dim=hidden_dim)

        # Cross-level fusion
        self.fusion = CrossLevelFusion(num_levels=3)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, dna, c1, c2, c3, return_attn=False):
        # Encoder
        dna_feats = self.encoder_dna(dna)
        c1_feats = self.encoder_c1(c1)
        c2_feats = self.encoder_c2(c2)
        c3_feats = self.encoder_c3(c3)

        f_c1, f_c2, f_c3 = [], [], []

        # save attention weight
        attn_weights_all = {"c1": {}, "c2": {}, "c3": {}} if return_attn else None

        for i, (d, v1, v2, v3) in enumerate(zip(dna_feats, c1_feats, c2_feats, c3_feats)):
            # (B, C, L) -> (B, L, C)
            d, v1, v2, v3 = d.permute(0,2,1), v1.permute(0,2,1), v2.permute(0,2,1), v3.permute(0,2,1)
            cross = [self.cross1, self.cross2, self.cross3][i]

            # layer1
            if i == 0:
                if return_attn:
                    # cross attention (dna & clinic)
                    out1, w1 = self.cross1(d, v1, return_attn=True)
                    out2, w2 = self.cross1(d, v2, return_attn=True)
                    out3, w3 = self.cross1(d, v3, return_attn=True)
                    attn_weights_all["c1"]['f1'] = w1
                    attn_weights_all["c2"]['f1'] = w2
                    attn_weights_all["c3"]['f1'] = w3
                else:
                    out1 = self.cross1(d, v1)
                    out2 = self.cross1(d, v2)
                    out3 = self.cross1(d, v3)
                # level embedding
                out1 = self.level_emb1(out1, i)
                out2 = self.level_emb1(out2, i)
                out3 = self.level_emb1(out3, i)
            # layer2
            elif i == 1:
                if return_attn:
                    # cross attentions
                    out1, w1 = self.cross2(d, v1, return_attn=True)
                    out2, w2 = self.cross2(d, v2, return_attn=True)
                    out3, w3 = self.cross2(d, v3, return_attn=True)
                    attn_weights_all["c1"]['f2'] = w1
                    attn_weights_all["c2"]['f2'] = w2
                    attn_weights_all["c3"]['f2'] = w3
                else:
                    out1 = self.cross2(d, v1)
                    out2 = self.cross2(d, v2)
                    out3 = self.cross2(d, v3)
                # level embedding
                out1 = self.level_emb2(out1, i)
                out2 = self.level_emb2(out2, i)
                out3 = self.level_emb2(out3, i)
            # layer3
            else:
                if return_attn:
                    # corss attention
                    out1, w1 = self.cross3(d, v1, return_attn=True)
                    out2, w2 = self.cross3(d, v2, return_attn=True)
                    out3, w3 = self.cross3(d, v3, return_attn=True)
                    attn_weights_all["c1"]['f3'] = w1
                    attn_weights_all["c2"]['f3'] = w2
                    attn_weights_all["c3"]['f3'] = w3
                else:
                    out1 = self.cross3(d, v1)
                    out2 = self.cross3(d, v2)
                    out3 = self.cross3(d, v3)
                # level embedding
                out1 = self.level_emb3(out1, i)
                out2 = self.level_emb3(out2, i)
                out3 = self.level_emb3(out3, i)

            target_len = dna_feats[0].size(2)  # f1 長度
            out1 = F.interpolate(out1.permute(0,2,1), size=target_len, mode='linear', align_corners=False).permute(0,2,1)
            out2 = F.interpolate(out2.permute(0,2,1), size=target_len, mode='linear', align_corners=False).permute(0,2,1)
            out3 = F.interpolate(out3.permute(0,2,1), size=target_len, mode='linear', align_corners=False).permute(0,2,1)

            f_c1.append(out1)
            f_c2.append(out2)
            f_c3.append(out3)

        # Concat channel
        t1 = self.proj(torch.cat(f_c1, dim=2))
        t2 = self.proj(torch.cat(f_c2, dim=2))
        t3 = self.proj(torch.cat(f_c3, dim=2))


        # Transformer + mean pooling (self-attention for each clinical modal)
        t1 = self.trans1(t1).mean(dim=1)
        t2 = self.trans2(t2).mean(dim=1)
        t3 = self.trans3(t3).mean(dim=1)

        # Cross-level fusion (fuse c1 c2 c3)
        fused = self.fusion([t1, t2, t3])

        # final prediction
        out = self.classifier(fused)

        if return_attn:
            return out, attn_weights_all
        return out
    
