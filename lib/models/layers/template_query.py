"""
LearnableTemplateQuery — 可学习模板查询。

用一组可学习的 query token 通过交叉注意力聚合模板特征，
替代简单的均值池化，更智能地提取模板中的目标信息。
比 CDA 更轻量，无需变形注意力。
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class LearnableTemplateQuery(nn.Module):
    """
    可学习模板查询：1 个 query token × 交叉注意力 → 模板特征聚合。

    Args:
        dim: 特征维度
        num_heads: 注意力头数
        mlp_ratio: FFN 隐藏层倍数
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 可学习查询 token
        self.query_token = nn.Parameter(torch.zeros(1, 1, dim))

        # 交叉注意力投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # FFN
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, template_tokens):
        """
        Args:
            template_tokens: (B, N_template, C)  模板 patch tokens
        Returns:
            query: (B, 1, C)  聚合后的模板查询
        """
        B, N, C = template_tokens.shape

        # query: 可学习 token
        q = self.query_token.expand(B, -1, -1)       # (B, 1, C)
        q = self.q_proj(q).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, h, 1, d)
        k = self.k_proj(template_tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(template_tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, 1, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (B, 1, C)
        x = self.out_proj(x)

        # 残差 + FFN
        query = q.permute(0, 2, 1, 3).reshape(B, 1, C)  # projected query
        query = self.norm1(query + x)
        query = self.norm2(query + self.mlp(query))

        return query  # (B, 1, C)
