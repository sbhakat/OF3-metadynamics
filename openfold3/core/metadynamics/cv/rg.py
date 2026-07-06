# openfold3/core/metadynamics/cv/rg.py
import torch
from openfold3.core.metrics.quality import _get_atom_name_mask

def rg_cv(
    coords: torch.Tensor,         # [B, S, N_atom, 3]
    batch: dict,                  # OF3 batch dict
    atom_names: tuple[str, ...] = ("CA",),
) -> torch.Tensor:
    """Returns [B, S] Rg over selected atoms (default: all Cα)."""
    sel_mask = _get_atom_name_mask(batch["ref_atom_name_chars"], list(atom_names))
    # sel_mask: [B, N_atom] bool. Multiply by atom_mask to exclude padding.
    sel = (sel_mask & batch["atom_mask"].bool()).float()           # [B, N_atom]
    n_eff = sel.sum(dim=-1, keepdim=True).clamp(min=1.0)            # [B, 1]
    w = sel.unsqueeze(1).unsqueeze(-1)                              # [B, 1, N_atom, 1]
    com = (coords * w).sum(dim=-2, keepdim=True) / n_eff.unsqueeze(1).unsqueeze(-1)
    sq = ((coords - com) * w).pow(2).sum(dim=(-1, -2)) / n_eff      # [B, S]
    return torch.sqrt(sq + 1e-8)
