"""
SCA — Semantic Correspondence Alignment (来自 TriGrATrack)。
将 TIR 模板对齐到 RGB 模板空间，补偿传感器物理位置偏差。

设计原则:
  - 软分配矩阵: 通过全局 softmax 实现平滑的特征重采样。
  - 两阶段策略: 语义聚类粗对齐 → Token 级精细对齐（残差细化）。
  - 置信度引导: 逐 token 自适应融合 refined/original TIR。
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class SemanticCorrespondenceAlign(nn.Module):
    """
    语义对应对齐: 将 TIR 模板对齐到 RGB 模板空间。

    Args:
        dim: 特征维度
        num_clusters: 聚类中心数
        mlp_ratio: 置信度 MLP 的隐藏层比例
    """

    def __init__(self, dim=768, num_clusters=16, mlp_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters

        # ── Stage 1: 粗粒度聚类对齐 ─────────────────────────────────
        self.cluster_proj = nn.Linear(dim, num_clusters)
        self.q_coarse = nn.Linear(dim, dim)
        self.k_coarse = nn.Linear(dim, dim)

        # ── Stage 2: 残差精细对齐 ───────────────────────────────────
        self.q_fine = nn.Linear(dim, dim)
        self.k_fine = nn.Linear(dim, dim)
        self.v_fine = nn.Linear(dim, dim)

        # ── 置信度估计 ─────────────────────────────────────────────
        hidden_dim = max(int(dim * mlp_ratio), 32)
        self.conf_mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ── 对齐后融合 (RGB + aligned TIR) ──────────────────────────
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.apply(self._init_weights)
        self._residual_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _residual_init(self):
        """零/小值初始化策略: 初始近似恒等，保证梯度链流畅。"""
        # v_fine 小值初始化 → tir_coarse_mapped ≈ 0，精细对齐初始贡献小
        nn.init.normal_(self.v_fine.weight, mean=0, std=1e-4)
        nn.init.constant_(self.v_fine.bias, 0)

        # conf_mlp 最后一层 bias = -1 → sigmoid ≈ 0.27
        # aligned_tir = 0.27 * refined + 0.73 * original (初始偏向保留原始 TIR)
        final_conf = self.conf_mlp[-1]
        nn.init.constant_(final_conf.weight, 0)
        nn.init.constant_(final_conf.bias, -1.0)

    def forward(self, rgb_z, tir_z):
        """
        Args:
            rgb_z: RGB 模板 tokens, shape (B, N, C)
            tir_z: TIR 模板 tokens, shape (B, N, C)
        Returns:
            tir_aligned: 对齐后的 TIR, (B, N, C)
            confidence:  逐 token 置信度, (B, N, 1)
            fused_template: RGB + aligned TIR 融合特征, (B, N, C)
        """
        B, N, C = rgb_z.shape
        scale = C ** -0.5

        # ── Stage 1: 粗粒度聚类对齐 ─────────────────────────────────
        cluster_w = torch.softmax(self.cluster_proj(rgb_z), dim=-1)  # (B, N, K)
        rgb_cluster = torch.bmm(cluster_w.transpose(1, 2), rgb_z)    # (B, K, C)
        tir_cluster = torch.bmm(cluster_w.transpose(1, 2), tir_z)    # (B, K, C)

        q_c, k_c = self.q_coarse(rgb_cluster), self.k_coarse(tir_cluster)
        S_coarse = torch.softmax(
            torch.bmm(q_c, k_c.transpose(1, 2)) * scale, dim=-1)     # (B, K, K)

        # 广播到 token 级对应矩阵
        S_fine = torch.bmm(torch.bmm(cluster_w, S_coarse),
                           cluster_w.transpose(1, 2))                 # (B, N, N)

        # ── Stage 2: 残差精细对齐 ───────────────────────────────────
        tir_coarse_mapped = torch.bmm(S_fine, self.v_fine(tir_z))
        q_f, k_f = self.q_fine(rgb_z), self.k_fine(tir_coarse_mapped)
        S_residual = torch.softmax(
            torch.bmm(q_f, k_f.transpose(1, 2)) * scale, dim=-1)
        tir_refined = torch.bmm(S_residual, tir_coarse_mapped)

        # ── 置信度引导融合 ─────────────────────────────────────────
        conf_input = torch.cat(
            [rgb_z, tir_refined, torch.abs(rgb_z - tir_refined)], dim=-1)
        confidence = torch.sigmoid(self.conf_mlp(conf_input))          # (B, N, 1)
        tir_aligned = confidence * tir_refined + (1 - confidence) * tir_z

        # ── 融合模板 ───────────────────────────────────────────────
        fused_template = self.fuse(torch.cat([rgb_z, tir_aligned], dim=-1))

        return tir_aligned, confidence, fused_template
