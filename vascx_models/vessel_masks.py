from __future__ import annotations

import numpy as np


def typed_vessel_masks(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a vessel mask into artery and vein masks using the AV segmentation."""
    artery_mask = vessel_mask & np.isin(av_mask, (1, 3))
    vein_mask = vessel_mask & np.isin(av_mask, (2, 3))
    return artery_mask, vein_mask
