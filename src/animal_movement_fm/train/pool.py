"""
Pre encode Tier 2 source pool and keep it resident on the training device.

This avoids us repeatedly encoding NumPy batches on the CPU while the GPU waits. Windows are encoded in chunks, concatenated 
once on the target device, and sampled with the same registered species then window rule used by SourcePool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from animal_movement_fm.data.enu import SCALE_FLOOR_M
from animal_movement_fm.data.multispecies import SourcePool

ENCODE_CHUNK = 50_000


@dataclass
class PreparedPool:
    """
    Encoded windows for one training pool, store contiguously on a single device.

    Each species occupies one half-open row range in `species_bounds`. Sampling therefore keeps the registered hierarchical 
    rule without needing a separate row index lookup.
    """

    names: tuple[str, ...]
    features: torch.Tensor # (N, seq, n_features)
    targets: torch.Tensor # (N, 2)
    species_bounds: torch.Tensor  # (S, 2) int64
    probabilities: torch.Tensor  # (S,)

    @property
    def n_windows(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[-1])

    def bytes_on_device(self) -> int:
        """
        Return memory used by the encoded feature tensor
        """
        return self.features.element_size() * self.features.nelement()

    @classmethod
    def from_source_pool(
        cls,
        pool: SourcePool,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        chunk: int = ENCODE_CHUNK,
        sigma: float = SCALE_FLOOR_M,
    ) -> "PreparedPool":
        """
        Encode every window in pool and move the resulting tensors to device`.

        Encoding is chunked to limit temporary host memory use. Species remain contiguous in the final tensors so sampling 
        can use simple row ranges.
        """
        device = torch.device(device)
        names = tuple(pool.names)

        feature_blocks = []
        target_blocks = []
        bounds = []
        cursor = 0

        for name in names:
            windows = pool.species[name]
            total = windows.n_windows

            if total == 0:
                raise ValueError(
                    f"species {name!r} contributes no windows to this pool. An empty species cannot be "
                    "drawn from and its registered weight would silently redistribute."
                )

            for start in range(0, total, chunk):
                ids = np.arange(start, min(start + chunk, total), dtype=np.int64)
                features, targets, _ = windows.gather(ids, sigma=sigma)

                feature_blocks.append(torch.as_tensor(features, dtype=dtype).to(device))
                target_blocks.append(torch.as_tensor(targets, dtype=dtype).to(device))

            bounds.append((cursor, cursor + total))
            cursor += total

        return cls(
            names=names,
            features=torch.cat(feature_blocks),
            targets=torch.cat(target_blocks),
            species_bounds=torch.tensor(bounds, dtype=torch.int64, device=device),
            probabilities=torch.as_tensor(
                pool.probabilities,
                dtype=torch.float64,
            ).to(device),
        )

    def sample(
        self,
        generator: torch.Generator,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw one batch under the registered species-then-window sampling rule.

        Returns `(features, targets, species_index)`. The species index is retained so realised loss/sample shares can be audited later.
        """
        picks = torch.multinomial(
            self.probabilities,
            batch_size,
            replacement=True,
            generator=generator,
        )

        starts = self.species_bounds[picks, 0]
        widths = self.species_bounds[picks, 1] - starts

        offsets = (
            torch.rand(
                batch_size,
                generator=generator,
                device=widths.device,
                dtype=torch.float64,
            )
            * widths.to(torch.float64)
        ).to(torch.int64)

        rows = starts + torch.clamp(offsets, max=widths - 1)
        return self.features[rows], self.targets[rows], picks

    def realised_species_shares(
        self,
        generator: torch.Generator,
        draws: int = 200_000,
    ) -> dict[str, float]:
        """
        Estimate the realised species shares produced by the sampler.

        Reporting diagnostic only; doesnt feed back into the registered probabilities or training configuration.
        """
        picks = torch.multinomial(
            self.probabilities,
            draws,
            replacement=True,
            generator=generator,
        )

        return {
            name: float((picks == i).to(torch.float64).mean())
            for i, name in enumerate(self.names)
        }
