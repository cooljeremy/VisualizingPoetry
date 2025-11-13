from typing import List, Dict, Optional
import torch
from torch import nn
from .pmsum.encoder import TextEncoder
from .pmsum.multitask_heads import MultiTaskHeads
from .fusion.fuse import FeatureFusion
from .fusion.attention import MultiHeadAttn
from .fusion.interaction import tensor_interaction
from .fusion.decoder import PromptDecoder


class PoeticModel(nn.Module):
    def __init__(self, text_model: str = "bert-base-chinese", hidden_dim: int = 768, fusion_dim: int = 512, decoder_layers: int = 4, num_heads: int = 8, vocab_size: int = 30522) -> None:
        super().__init__()
        self.encoder = TextEncoder(text_model)
        self.heads = MultiTaskHeads(hidden_dim, 8, 16, 20)
        self.proj_emo = nn.Linear(8, fusion_dim)
        self.proj_img = nn.Linear(16, fusion_dim)
        self.proj_rhe = nn.Linear(20, fusion_dim)
        self.f_fuse1 = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.GELU())
        self.g1 = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.GELU())
        self.f_fuse2 = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.GELU())
        self.g2 = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.GELU())
        self.att_in = nn.Linear(fusion_dim, fusion_dim)
        self.attn = MultiHeadAttn(fusion_dim, num_heads)
        self.att_pool = nn.AdaptiveAvgPool1d(1)
        self.inter_proj = nn.Linear(fusion_dim * fusion_dim, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)
        self.decoder = PromptDecoder(fusion_dim, decoder_layers, vocab_size)

    def forward(self, texts: List[str], tgt_ids: Optional[torch.Tensor] = None, max_length: int = 128) -> Dict[str, torch.Tensor]:
        seq, pooled = self.encoder(texts, max_length)
        mtl = self.heads(pooled)
        E = self.proj_emo(mtl["emo"])
        I = self.proj_img(mtl["img"])
        R = self.proj_rhe(mtl["rhe"])
        h_low = self.f_fuse1(E) + self.g1(torch.cat([E, I], dim=-1))
        h_mid = self.f_fuse2(h_low) + self.g2(torch.cat([h_low, R], dim=-1))
        feats = torch.stack([E, I, R], dim=1)
        feats = self.att_in(feats)
        att_out = self.attn(feats, feats, feats)
        att_vec = self.att_pool(att_out.transpose(1, 2)).squeeze(-1)
        inter_ei = tensor_interaction(E, I)
        inter_ir = tensor_interaction(I, R)
        h_interact = self.inter_proj(inter_ei + inter_ir)
        h_poem = self.norm(h_interact + att_vec + h_mid)
        out = {"h_poem": h_poem, "heads": mtl}
        if tgt_ids is not None:
            out["logits"] = self.decoder(h_poem, tgt_ids)
        return out
