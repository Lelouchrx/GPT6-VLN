"""Conservative near-field depth warning for the next two forward primitives."""

import numpy as np


def forward_clearance(observation, distance_m=0.5, radius_m=0.18):
    depth = observation.get('rgb_map_depth')
    if depth is None:
        return {'status': 'unknown', 'reason': 'front depth unavailable'}
    pixels = np.asarray(depth, dtype=float).squeeze()
    if pixels.ndim != 2 or min(pixels.shape) < 16:
        return {'status': 'unknown', 'reason': 'invalid front depth shape'}
    height, width = pixels.shape
    # The 120-degree camera sees a narrow forward body corridor near its center.
    crop = pixels[int(.38*height):int(.62*height), int(.43*width):int(.57*width)]
    valid = np.isfinite(crop) & (crop > .02) & (crop < 10.0)
    valid_fraction = float(valid.mean())
    if valid_fraction < .6:
        return {'status': 'unknown', 'reason': 'insufficient valid depth',
                'valid_fraction': round(valid_fraction, 3)}
    clearance = float(np.quantile(crop[valid], .25))
    # This is an advisory view check. The horizontal camera cannot certify
    # low steps, glass, or traversability of an unseen floor patch.
    warning = clearance < distance_m + radius_m
    return {'status': 'possible_obstacle' if warning else 'no_near_obstacle_seen',
            'clearance_m': round(clearance, 3),
            'lookahead_m': distance_m, 'valid_fraction': round(valid_fraction, 3),
            'reason': 'front depth corridor within two forward steps' if warning
                      else 'no near surface in sampled corridor; floor remains unverified'}
