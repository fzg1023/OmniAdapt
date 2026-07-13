"""
AdaptiveFusionGate (AFG) — 后置自适应融合门控。
灵感来自 DropFuseRGBT 的 AdaptiveBimodalFuse (T2 模块)。

在 ReliabilityGuidedFuse 的融合结果之上，增加 SE-style 自适应门控：
  - 从 RGB/TIR 均值特征中学习 per-channel 调制权重
  - 三层保护：差异置信度 + 自适应幅度 + 硬关闭
  - 零初始化 → 训练初期等价于原 ReliabilityGuidedFuse 输出
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class AdaptiveFusionGate(nn.Module):
    """
    后置自适应融合门控 — SE-style channel-wise modulation on fused features.

    Args:
        dim: 特征维度
        reduction: SE 压缩比
        gate_scale: 调制最大幅度

    设计：
      gate = SE( cat(mean(vi), mean(ir)) )  → per-channel weights [B, C]
      gated_fused = fused * (1 + gate)
      初始 gate=0 → 等价于原融合输出
    """

    GATE_SCALE: float = 0.05
    GATE_KILL_RATIO: float = 0.15

    def __init__(self, dim, reduction=4, gate_scale=0.05):
        super().__init__()
        self.dim = dim
        self.GATE_SCALE = gate_scale
        mid_dim = max(dim // reduction, 32)

        # SE-style 门控
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, dim),
        )

        self.apply(self._init_weights)
        self._zero_init_gate()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _zero_init_gate(self):
        """门控最后一层零初始化 → 初始 gate=0 → 等价原融合。"""
        last = self.gate_net[-1]
        nn.init.constant_(last.weight, 0)
        nn.init.constant_(last.bias, 0)

    def forward(self, fused, vi, ir):
        """
        Args:
            fused: 已融合特征 (B, N, C)
            vi:    RGB 特征 (B, N, C)
            ir:    TIR 特征 (B, N, C)
        Returns:
            gated_fused: (B, N, C)
        """
        B, N, C = fused.shape

        # 模态全局特征
        v_g = vi.mean(dim=1)  # (B, C)
        i_g = ir.mean(dim=1)  # (B, C)
        gate_raw = self.gate_net(torch.cat([v_g, i_g], dim=-1))  # (B, C)

        # 差异置信度：模态差异越大，门控越可信
        diff_norm = (v_g - i_g).norm(dim=-1, keepdim=True)
        avg_norm = ((v_g.norm(dim=-1, keepdim=True) +
                     i_g.norm(dim=-1, keepdim=True)) / 2).clamp(min=1e-6)
        gate_confidence = (diff_norm / avg_norm).clamp(0.0, 1.0)  # (B, 1)

        # 自适应幅度：融合特征越强，门控越小（已融合得很好就不需要调）
        fuse_norm = fused.detach().norm(dim=-1).mean(dim=1, keepdim=True)  # (B, 1)
        batch_mean = fuse_norm.mean().clamp(min=1e-6)
        gate_scale = self.GATE_SCALE * torch.sigmoid(2.0 - fuse_norm / batch_mean)  # (B, 1)

        # 最终 gate
        gate = gate_scale * torch.tanh(gate_raw) * gate_confidence  # (B, C)

        # 硬关闭：gate 太小时直接跳过
        kill = gate_scale * gate_confidence < self.GATE_SCALE * self.GATE_KILL_RATIO
        gate = gate.masked_fill(kill, 0.0)

        return fused * (1.0 + gate.unsqueeze(1))  # (B, N, C)
