"""
SpatiallyAdaptiveAggregator — 空间自适应层聚合器（来自 TriGrATrack）。

实现"频域角色解耦 + 空间路由"的聚合机制：
  - 按深度将层分为三组：细节(detail)、结构(structure)、语义(semantic)
  - 每个 token 位置独立决定从各组获取多少信息
  - 路由器零初始化 → 训练初期等价于等权平均聚合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class SpatiallyAdaptiveAggregator(nn.Module):
    """
    空间自适应层聚合：每个 token 位置独立决定从细节/结构/语义组获取多少信息。

    Args:
        dim: 特征维度
        interact_layer: 交互层列表，用于确定总层数
        template_size: 模板图像尺寸（预留，forward 中未使用）
        patch_size: patch 大小
    """

    def __init__(self, dim, interact_layer=None, template_size=128, patch_size=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_layers = len(interact_layer) if interact_layer else 12
        self.interact_layer = interact_layer if interact_layer else list(range(12))
        self.template_tokens = (template_size // patch_size) ** 2

        # 1. 角色分组初始化
        self._build_role_groups()

        # 2. 空间路由器：输入各层统计量 (mean+std)，输出 3 组权重
        router_dim = self.num_layers * 2
        self.router = nn.Sequential(
            nn.Linear(router_dim, router_dim // 2),
            nn.GELU(),
            nn.Linear(router_dim // 2, 3),  # detail, structure, semantic
        )

        # 3. 组内投影层
        self.detail_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.struct_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.semantic_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

        # 4. 跨组融合与归一化
        self.cross_group_fuse = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(dim)
        self._init_weights()

    def _build_role_groups(self):
        """按深度将层分为三组：前1/3(细节), 中1/3(结构), 后1/3(语义)"""
        n = self.num_layers
        third = max(n // 3, 1)
        self.detail_indices = list(range(0, third))
        self.struct_indices = list(range(third, 2 * third))
        self.semantic_indices = list(range(2 * third, n))

    def _init_weights(self):
        """零初始化路由器和融合层，初始时等价于等权求和"""
        nn.init.constant_(self.router[-1].weight, 0)
        nn.init.constant_(self.router[-1].bias, 0)
        nn.init.constant_(self.cross_group_fuse[0].weight, 0)
        nn.init.constant_(self.cross_group_fuse[0].bias, 0)

    def _compute_token_statistics(self, feats):
        """计算每个 token 在各层的 [mean, std] 作为路由特征"""
        stats = [torch.cat([f.mean(dim=-1, keepdim=True),
                            f.std(dim=-1, keepdim=True)], dim=-1)
                 for f in feats]
        return torch.cat(stats, dim=-1)

    def forward(self, feats):
        """
        Args:
            feats: List[Tensor], 各层特征，每个 shape (B, N, dim)
        Returns:
            aggregated: (B, N, dim)
        """
        # 1) 路由权重计算 (B, N, 3)
        token_stats = self._compute_token_statistics(feats)
        role_weights = F.softmax(self.router(token_stats), dim=-1)

        # 2) 组内聚合 (Mean Pooling)
        d_feat = self.detail_proj(
            torch.stack([feats[i] for i in self.detail_indices]).mean(0))
        s_feat = self.struct_proj(
            torch.stack([feats[i] for i in self.struct_indices]).mean(0))
        m_feat = self.semantic_proj(
            torch.stack([feats[i] for i in self.semantic_indices]).mean(0))

        # 3) 空间自适应加权混合
        spatial_mixed = (role_weights[:, :, 0:1] * d_feat +
                         role_weights[:, :, 1:2] * s_feat +
                         role_weights[:, :, 2:3] * m_feat)

        # 4) 跨组残差投影
        cross_fused = self.cross_group_fuse(
            torch.cat([d_feat, s_feat, m_feat], dim=-1))
        return self.norm(spatial_mixed + cross_fused)
