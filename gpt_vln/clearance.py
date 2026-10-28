"""Conservative forward clearance check from the current front depth image."""
import numpy as np


def forward_clearance(depth, threshold_m=0.5):
    values = np.asarray(depth, dtype=float).squeeze()
    if values.ndim != 2:
        raise ValueError('Front depth must be a two-dimensional metric image')
    height, width = values.shape
    if height < 20 or width < 20:
        raise ValueError('Front depth resolution is too small')
    # Central body-height rays avoid floor and stair tread returns near the image bottom.
    patch = values[int(height * .44):int(height * .60),
                   int(width * .40):int(width * .60)]
    valid = patch[np.isfinite(patch) & (patch > .05) & (patch < 10.0)]
    valid_fraction = valid.size / patch.size
    if valid_fraction < .75:
        return {'interrupt': False, 'verdict': 'unknown', 'clearance_m': None,
                'reason': 'insufficient valid forward depth',
                'valid_fraction': round(valid_fraction, 3)}
    clearance = float(np.median(valid))
    near_fraction = float(np.mean(valid < threshold_m))
    interrupt = clearance < threshold_m and near_fraction >= .7
    return {'interrupt': interrupt,
            'verdict': 'near_surface' if interrupt else 'clear_or_uncertain',
            'clearance_m': round(clearance, 3),
            'threshold_m': threshold_m,
            'near_fraction': round(near_fraction, 3),
            'valid_fraction': round(valid_fraction, 3),
            'reason': ('front depth shows a near surface before the next 25 cm forward step; '
                       'choose a new view or direction' if interrupt
                       else 'no dense near surface in the central forward depth region')}
