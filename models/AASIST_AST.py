"""
AASIST + AST (Audio Spectrogram Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Import AASIST components
from models.AASIST import (
    GraphAttentionLayer,
    HtrgGraphAttentionLayer,
    GraphPool,
    CONV,
    Residual_block
)


class ASTEncoder(nn.Module):
    def __init__(self, input_tdim=646, input_fdim=128, embed_dim=384, depth=6, num_heads=6):
        super().__init__()

        # Mel spectrogram extraction
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=80,
            n_mels=128
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            1, embed_dim, kernel_size=(16, 16), stride=(16, 16)
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, embed_dim))

    def forward(self, x):
        # المتوقع أن x شكله (B, T) أو (B, 1, T)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() != 2:
            raise ValueError(f"Expected input shape (B, T) or (B, 1, T), but got {x.shape}")

        # Extract Mel Spectrogram
        with torch.no_grad():
            x = self.mel_spec(x)          # (B, F, T)
            x = self.amplitude_to_db(x)

        # (B, F, T) -> (B, 1, F, T)
        x = x.unsqueeze(1)

        # Patch embedding
        x = self.patch_embed(x)           # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        B, N, _ = x.shape

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)

        # Add positional embedding
        if N + 1 > self.pos_embed.size(1):
            raise ValueError(
                f"Number of AST tokens ({N+1}) exceeds positional embedding size ({self.pos_embed.size(1)})."
            )

        x = x + self.pos_embed[:, :N + 1, :]

        # Transformer
        x = self.transformer(x)

        # Return CLS token output
        return x[:, 0]


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()

        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        self.conv_time = CONV(
            out_channels=filts[0],
            kernel_size=d_args["first_conv"],
            in_channels=1
        )
        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4]))
        )

        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1],
            gat_dims[0],
            temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1],
            gat_dims[0],
            temperature=temperatures[1]
        )

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        # AST encoder
        self.ast_embed_dim = 384
        self.ast_encoder = ASTEncoder(embed_dim=self.ast_embed_dim)

        # AASIST feature dim = 5 * gat_dims[1]
        self.out_layer = nn.Linear(5 * gat_dims[1] + self.ast_embed_dim, 2)

    def load_aasist_weights(self, pretrained_path):
        print(f"Loading pretrained AASIST weights from: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location="cpu")

        # handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        current_state = self.state_dict()

        compatible_state = {
            k: v for k, v in state_dict.items()
            if k in current_state and current_state[k].shape == v.shape
        }

        skipped_from_pretrained = [
            k for k in state_dict.keys() if k not in compatible_state
        ]
        missing_in_pretrained = [
            k for k in current_state.keys() if k not in compatible_state
        ]

        current_state.update(compatible_state)
        self.load_state_dict(current_state, strict=False)

        print(f"Loaded {len(compatible_state)} matching layers from pretrained AASIST.")
        print(f"Skipped {len(skipped_from_pretrained)} unmatched pretrained layers.")
        print(f"Uninitialized/new layers in current model: {len(missing_in_pretrained)}")

    def forward(self, x, Freq_aug=False):
        # x expected: (B, T)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        # AST branch
        ast_features = self.ast_encoder(x)

        # AASIST branch
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # get embeddings using encoder
        e = self.encoder(x)

        # spectral GAT (GAT-S)
        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # temporal GAT (GAT-T)
        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=master1
        )

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)],
            dim=1
        )

        last_hidden = self.drop(last_hidden)

        # combine AASIST + AST
        combined_features = torch.cat([last_hidden, ast_features], dim=1)
        output = self.out_layer(combined_features)

        return combined_features, output
