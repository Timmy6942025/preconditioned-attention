import torch
import torch.nn as nn
from torch import Tensor


class ConditionNumberMonitor:
    def __init__(self):
        self.history: list[dict] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def register_hook(self, module: nn.Module, layer_idx: int, head_idx: int) -> None:
        def hook_fn(mod: nn.Module, inp: tuple, out: tuple[Tensor, Tensor]) -> None:
            attn_output = out[0]
            with torch.no_grad():
                B, H, N, D = attn_output.shape
                matrix = attn_output.float().view(B * H * N, D)
                if matrix.shape[0] > 0 and matrix.shape[1] > 0:
                    s = torch.linalg.svdvals(matrix)
                    if len(s) >= 2 and s[-1] > 1e-10:
                        cond = (s[0] / s[-1]).item()
                    else:
                        cond = float("inf")
                else:
                    cond = 0.0
            self.history.append({"layer": layer_idx, "head": head_idx, "condition_number": cond})

        self._hooks.append(module.register_forward_hook(hook_fn))

    def get_average_condition_number(self, step: int | None = None) -> float:
        if step is not None:
            recent = [h for h in self.history if h.get("step") == step]
        else:
            recent = self.history[-100:] if self.history else []
        if not recent:
            return 0.0
        return sum(h["condition_number"] for h in recent) / len(recent)

    def clear(self) -> None:
        self.history.clear()

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class StableRank:
    @staticmethod
    def compute(matrix: Tensor) -> float:
        fro_norm_sq = torch.norm(matrix, "fro") ** 2
        spectral_norm_sq = torch.linalg.norm(matrix, 2) ** 2
        if spectral_norm_sq < 1e-10:
            return 0.0
        return (fro_norm_sq / spectral_norm_sq).item()
