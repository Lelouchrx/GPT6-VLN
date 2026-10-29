
import math


class PoseStallCritic:
    def __init__(self, min_forward_m=0.075, consecutive=2):
        self.min_forward_m = min_forward_m
        self.consecutive = consecutive
        self.low_count = 0

    def check(self, action, delta_xz, heading_deg, step):
        if action != 1:
            self.low_count = 0
            return {'step': step, 'blocked': False, 'interrupt': False,
                    'reason': 'not a forward primitive', 'completion': 'unknown'}
        angle = math.radians(heading_deg)
        forward = delta_xz[0] * math.cos(angle) + delta_xz[1] * math.sin(angle)
        self.low_count = self.low_count + 1 if forward < self.min_forward_m else 0
        blocked = self.low_count >= self.consecutive
        return {'step': step, 'forward_m': round(forward, 4),
                'low_motion_count': self.low_count, 'blocked': blocked,
                'interrupt': blocked,
                'reason': 'measured forward motion below threshold; choose a new local route' if blocked else 'motion check',
                'completion': 'unknown',
                'limitation': 'Blocked motion says nothing about target identity or arrival.'}
