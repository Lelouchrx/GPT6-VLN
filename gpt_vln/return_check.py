
import numpy as np


def return_arrival(target_position, current_position, radius_m=0.5):
    target = np.asarray(target_position, dtype=float)[[0, 2]]
    current = np.asarray(current_position, dtype=float)[[0, 2]]
    distance = float(np.linalg.norm(target - current))
    return {'arrived': distance <= radius_m, 'distance_xz_m': round(distance, 3),
            'radius_m': radius_m, 'source': 'measured own simulator pose',
            'semantic_route': 'unknown'}
