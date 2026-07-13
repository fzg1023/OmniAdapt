"""
ReliabilityGuidedFuse — 跨模态线性融合模块（来自 TriGrATrack）。
参考 LRes 的 BimodalLinearFuse，包含：
  1. 模态可靠性门控 (reliability_gate): 估计 RGB/TIR 的样本级可信度，初始等价双模态等权
  2. 差异感知残差 (diff_proj): |vi-ir| 产生差异残差，初始为零输出
  3. 主路径 fuse(cat([vi, ir])): Linear(2C→C) + LayerNorm + GELU

兼容性设计：
  - reliability_gate 最后一层零初始化，初始 softmax=[0.5,0.5]，
    再乘以 2 后得到 scale=[1,1]，因此主路径初始等价于原始 fuse。
  - diff_proj 最后一层零初始化，初始差异残差为 0。
  - vi/ir 主干 token 不被写回修改，只生成 fused_feat。
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class ReliabilityGuidedFuse(nn.Module):
    """
    可靠性引导 + 差异感知的跨模态线性融合。
    """

    def __init__(self, dim, drop_path=0., fusion_layers=12,
                 use_reliability=True, use_difference=True):
        super().__init__()
        self.fusion_layers = fusion_layers
        self.use_reliability = use_reliability
        self.use_difference = use_difference

        # ── 主路径 fuse（Linear 2C→C + LayerNorm + GELU）────────────────
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # ── 模态可靠性门控 ──────────────────────────────────────────────
        # 输入: cat([mean(vi), mean(ir), mean(|vi-ir|)]) → dim*3
        # 输出: softmax 权重 × 2 → 初始 [1,1]
        hidden = max(dim // 4, 32)
        self.reliability_gate = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

        # ── 差异感知残差投影 ────────────────────────────────────────────
        # 输入: |vi - ir|，输出: 残差 (初始为零)
        self.diff_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)
        self._init_identity_enhancements()

        self.last_reliability = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_identity_enhancements(self):
        """将增强模块输出层零初始化 → 初始等价于纯 fuse。"""
        # reliability_gate: 最后一层零初始化 → softmax=[0.5,0.5] → ×2=[1,1]
        final_gate = self.reliability_gate[-1]
        nn.init.constant_(final_gate.weight, 0)
        nn.init.constant_(final_gate.bias, 0)

        # diff_proj: 最后一层零初始化 → 残差=0
        final_diff = self.diff_proj[-1]
        nn.init.constant_(final_diff.weight, 0)
        nn.init.constant_(final_diff.bias, 0)

    def _modal_reliability(self, vi, ir):
        """估计 RGB/TIR 的样本级可信度权重。"""
        diff = torch.abs(vi - ir)
        stats = torch.cat(
            [vi.mean(dim=1), ir.mean(dim=1), diff.mean(dim=1)],
            dim=-1,  # (B, 3D)
        )
        weights = torch.softmax(self.reliability_gate(stats), dim=-1)  # (B, 2)
        return weights * 2.0  # 初始 [0.5,0.5] → scale [1,1]

    def forward(self, oral_vi, oral_ir):
        # ── 1. 可靠性调制：初始 scale=[1,1]，等价于原始融合 ────────────
        if self.use_reliability:
            reliability = self._modal_reliability(oral_vi, oral_ir)
            vi_scale = reliability[:, 0].view(-1, 1, 1)
            ir_scale = reliability[:, 1].view(-1, 1, 1)
            vi_in = oral_vi * vi_scale
            ir_in = oral_ir * ir_scale
            self.last_reliability = reliability.detach()
        else:
            vi_in, ir_in = oral_vi, oral_ir
            self.last_reliability = None

        # ── 2. 主路径融合 ──────────────────────────────────────────────
        fused = self.fuse(torch.cat([vi_in, ir_in], dim=-1))

        # ── 3. 差异感知残差：初始为零，不破坏现有输出 ──────────────────
        if self.use_difference:
            diff_residual = self.diff_proj(torch.abs(oral_vi - oral_ir))
            fused = fused + self.drop_path(diff_residual)

        return fused, oral_vi, oral_ir
