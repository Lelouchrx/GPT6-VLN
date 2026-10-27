
import math

import numpy as np
import quaternion


VIEWS = {'front': (0.0, ''), 'right': (-90.0, 'right_'),
         'rear': (180.0, 'rear_'), 'left': (90.0, 'left_')}


class TargetAnchorCritic:
    def __init__(self):
        self.anchor = None
        self.last = {'status': 'unknown', 'reason': 'no target bound'}

    def bind(self, target, obs, sim, frame):
        if target['status'] not in ('seen', 'approaching') or target['point_norm'] is None:
            return self.last
        view = target['view']
        u, v = target['point_norm']
        depth = np.asarray(obs[VIEWS[view][1] + 'rgb_map_depth']).squeeze()
        h, w = depth.shape

        x_ray = (2 * u - 1) * math.tan(math.radians(90) / 2)
        y_ray = (1 - 2 * v) * math.tan(math.radians(90) / 2)
        du = (x_ray / math.tan(math.radians(120) / 2) + 1) * w / 2
        dv = (1 - y_ray / math.tan(math.radians(120) / 2)) * h / 2
        cx, cy = int(du), int(dv)
        patch = depth[max(0, cy-3):min(h, cy+4), max(0, cx-3):min(w, cx+4)]
        valid = patch[np.isfinite(patch) & (patch > .1) & (patch < 10)]
        if valid.size < 9 or np.percentile(valid, 75) - np.percentile(valid, 25) > .6:
            self.last = {'status': 'unknown', 'reason': 'unstable target depth'}
            return self.last
        distance = float(np.median(valid))
        ray = np.array([x_ray, y_ray, -1.0]) * distance
        angle = math.radians(VIEWS[view][0])
        c, s = math.cos(angle), math.sin(angle)
        ray = np.array([c*ray[0]+s*ray[2], ray[1], -s*ray[0]+c*ray[2]])
        pose = sim.get_agent_state()
        world = np.asarray(pose.position, dtype=float) + np.array([0, 1.25, 0])
        world += quaternion.as_rotation_matrix(pose.rotation) @ ray
        self.anchor = {'xyz': world, 'frame': frame, 'view': view,
                       'description': target['description'], 'source_depth_m': round(distance, 2)}
        return self.observe(sim)

    def observe(self, sim):
        if self.anchor is None:
            return self.last
        pose = sim.get_agent_state()
        delta = self.anchor['xyz'] - np.asarray(pose.position, dtype=float) - np.array([0, 1.25, 0])
        local = quaternion.as_rotation_matrix(pose.rotation).T @ delta
        distance = float(np.linalg.norm(delta[[0, 2]]))
        bearing = math.degrees(math.atan2(local[0], -local[2]))
        self.last = {'status': 'tracked', 'source_frame': self.anchor['frame'],
                     'description_hypothesis': self.anchor['description'],
                     'distance_xz_m': round(distance, 2), 'bearing_deg_right': round(bearing, 1),
                     'source_depth_m': self.anchor['source_depth_m'],
                     'limitation': 'Depth ray may hit a surface behind an opening; geometry cannot verify doorway identity or crossing.'}
        return self.last

    def after_primitive(self, sim, action, previous, expected):
        now = self.observe(sim)
        if now['status'] != 'tracked' or previous.get('status') != 'tracked':
            return {'interrupt': False, 'target': now, 'verdict': 'unknown'}
        delta = now['distance_xz_m'] - previous['distance_xz_m']
        turned_behind = abs(previous['bearing_deg_right']) < 90 and abs(now['bearing_deg_right']) > 110
        moving_away = action == 1 and expected == 'decrease' and delta > .35
        interrupt = turned_behind or moving_away
        return {'interrupt': interrupt, 'target': now,
                'verdict': 'possibly_passed' if turned_behind else 'inconsistent' if moving_away else 'unknown',
                'measured_distance_change_m': round(delta, 2),
                'reason': 'anchor moved behind' if turned_behind else 'measured distance increased' if moving_away else 'no decisive geometric conflict'}
