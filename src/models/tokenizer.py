import torch

class SparseCodeTokenizer:
    def __init__(self, num_atoms, value_bins=1024, vmin=-1.0, vmax=1.0, max_nonzero=2):
        self.num_atoms = num_atoms
        self.value_bins = value_bins
        self.vmin, self.vmax = vmin, vmax
        self.max_nonzero = max_nonzero

        # layout in the vocabulary
        self.idx_base = 0
        self.val_base = self.idx_base + num_atoms
        self.vocab_size = self.val_base + value_bins

        # precompute position offsets for the 2*k slots
        self._pos_idxs = 2 * torch.arange(self.max_nonzero)           # [k]
        self._pos_vals = self._pos_idxs + 1                           # [k]

    def encode(self, codes: torch.Tensor) -> torch.Tensor:
        _, B = codes.shape
        k = self.max_nonzero

        # 1) find top‐k indices and values
        vals, idxs = codes.abs().topk(k, dim=0)          # both [B, k]
        signed_vals = torch.gather(codes, 0, idxs)      # [B, k]

        # 2) quantize
        clamped = signed_vals.clamp(self.vmin, self.vmax)
        norm    = (clamped - self.vmin) / (self.vmax - self.vmin)
        bins    = (norm * (self.value_bins - 1)).round().long()  # [B, k]

        # 3) prepare output
        seq_len = 2 * k
        out = torch.full((seq_len, B), 0, device=codes.device, dtype=torch.long)

        # 4) scatter in idx tokens and value tokens
        #    add bases
        idx_tokens = idxs + self.idx_base       # [B, k]
        val_tokens = bins + self.val_base       # [B, k]

        # fancy indexing assignment
        out[self._pos_idxs, :] = idx_tokens.long()
        out[self._pos_vals, :] = val_tokens.long()

        return out

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        # unchanged from your vectorized version...
        device = tokens.device
        L, B = tokens.shape
        k = L // 2

        body       = tokens                      # [B, 2*k]
        idx_tokens = body[:, 0::2]               # [B, k]
        val_tokens = body[:, 1::2]               # [B, k]

        idxs = idx_tokens - self.idx_base        # [B, k]
        bins = val_tokens - self.val_base        # [B, k]

        valid = (idxs >= 0) & (idxs < self.num_atoms) & \
                (bins >= 0) & (bins < self.value_bins)

        bins_clamped = bins.clamp(0, self.value_bins - 1).float()
        norm = (bins_clamped + 0.5) / self.value_bins
        vals = norm * (self.vmax - self.vmin) + self.vmin
        vals = vals * valid.float()

        codes = torch.zeros(self.num_atoms, B, device=device, dtype=torch.float32)
        codes = codes.scatter_add(0, idxs.clamp(0, self.num_atoms - 1), vals)
        return codes
