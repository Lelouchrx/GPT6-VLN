
import copy
import cv2
import numpy as np


def create_rgb_env(split, gpu, map_depth=False, horizontal_only=False):
    from eval import ROOT, WORKSPACE, Env, get_config, habitat, require_habitat
    from habitat.core.registry import registry
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimRGBSensor, HabitatSimDepthSensor
    require_habitat()
    config = get_config(str(ROOT / 'config/vln_r2r.yaml'))
    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = str(WORKSPACE / 'StreamVLN/data/datasets/r2r/{split}/{split}.json.gz')
        config.habitat.dataset.scenes_dir = str(WORKSPACE / 'StreamVLN/data/scene_datasets')
        config.habitat.environment.max_episode_steps = 401
        sim = config.habitat.simulator
        sim.habitat_sim_v0.gpu_device_id = gpu
        sim.forward_step_size = .25
        sim.turn_angle = 15
        sensors = sim.agents.main_agent.sim_sensors
        template = copy.deepcopy(sensors.rgb_sensor)
        depth_template = copy.deepcopy(sensors.depth_sensor)
        for key in list(sensors.keys()):
            del sensors[key]
        template.width, template.height, template.hfov = 640, 480, 90

        views = [('rgb', 0., 0.), ('right_rgb', 0., -np.pi/2),
                 ('rear_rgb', 0., np.pi), ('left_rgb', 0., np.pi/2),
                 ('floor_rgb', -np.pi/4, 0.), ('floor_right_rgb', -np.pi/4, -np.pi/2),
                 ('floor_rear_rgb', -np.pi/4, np.pi), ('floor_left_rgb', -np.pi/4, np.pi/2)]
        if horizontal_only:
            views = views[:4]
        for uuid, pitch, yaw in views:
            name = 'NavigationRGB' + uuid.title().replace('_', '')
            if registry.get_sensor(name) is None:
                cls = type(name, (HabitatSimRGBSensor,), {'_get_uuid': lambda self, *args, value=uuid, **kw: value})
                registry.register_sensor(name=name)(cls)
            sensor = copy.deepcopy(template)
            sensor.type = name
            sensor.orientation = [pitch, yaw, 0.]
            sensors[uuid + '_sensor'] = sensor
        if map_depth:
            depth_template.width, depth_template.height, depth_template.hfov = 400, 400, 120
            depth_template.normalize_depth = False
            for uuid, _, yaw in views[:4]:
                color_name = 'NavigationMapColor' + uuid.title().replace('_', '')
                if registry.get_sensor(color_name) is None:
                    cls = type(color_name, (HabitatSimRGBSensor,), {'_get_uuid': lambda self, *args, value=uuid+'_map_color', **kw: value})
                    registry.register_sensor(name=color_name)(cls)
                color_sensor = copy.deepcopy(template)
                color_sensor.width, color_sensor.height, color_sensor.hfov = 400, 400, 120
                color_sensor.type = color_name
                color_sensor.orientation = [0, yaw, 0]
                sensors[uuid + '_map_color_sensor'] = color_sensor
                name = 'NavigationMapDepth' + uuid.title().replace('_', '')
                if registry.get_sensor(name) is None:
                    cls = type(name, (HabitatSimDepthSensor,), {'_get_uuid': lambda self, *args, value=uuid+'_map_depth', **kw: value})
                    registry.register_sensor(name=name)(cls)
                sensor = copy.deepcopy(depth_template)
                sensor.type = name
                sensor.orientation = [0, yaw, 0]
                sensors[uuid + '_map_depth_sensor'] = sensor
    return Env(config)


def views(obs, horizontal_only=False):
    result = []
    for floor in ((False,) if horizontal_only else (False, True)):
        tiles = []
        for label, key in [('FRONT', 'rgb'), ('RIGHT', 'right_rgb'), ('REAR', 'rear_rgb'), ('LEFT', 'left_rgb')]:
            key = 'floor_' + key if floor else key
            tile = cv2.resize(np.asarray(obs[key])[..., :3], (384, 288)).copy()
            cv2.putText(tile, label + (' DOWN45' if floor else ''), (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,0,0), 4)
            cv2.putText(tile, label + (' DOWN45' if floor else ''), (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            tiles.append(tile)
        result.append(('FLOOR4' if floor else 'PANORAMA_FRONT_RIGHT_REAR_LEFT',
                       np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)))
    return result


def visual_change(before, after):

    scores = []
    for key in ('rgb', 'right_rgb', 'rear_rgb', 'left_rgb'):
        a = cv2.resize(np.asarray(before[key])[..., :3], (160,120)).astype(float)
        b = cv2.resize(np.asarray(after[key])[..., :3], (160,120)).astype(float)
        scores.append(float(np.mean(np.abs(a-b))))
    return round(float(np.median(scores)), 3)


def integrate(before, after, action, step):
    score = visual_change(before, after) if action == 1 else None
    return {'action':action, 'step':step, 'estimated_move_m':None,
            'executed_turn_deg':15 if action == 2 else -15 if action == 3 else 0,
            'rgb_change':score,
            'translation_status':'unknown; RGB appearance change is not metric motion'}


def execution_summary(execution):
    actions = execution.get('actual', [])
    forward = [a for a in actions if a['action'] == 1]
    return {'forward_attempts':len(forward),
            'low_change_attempts':sum(a.get('rgb_change') is not None and a['rgb_change'] < 1.5 for a in forward),
            'rgb_change_scores':[a.get('rgb_change') for a in forward],
            'turn_deg':sum(a.get('executed_turn_deg',0) for a in actions),
            'metric_displacement':'unknown',
            'limitation':'Low image change proposes a check; texture, lighting and view ambiguity can mislead.'}


class EvidenceLedger:
    def __init__(self):
        self.clauses = {}

    def update(self, events, allowed):
        for event in events:
            row = copy.deepcopy(event)
            row['frames'] = sorted(set(row.get('frames', [])) & set(allowed))[-3:]
            row['status'] = 'actor_hypothesis; frame citation is not independent confirmation'
            self.clauses[' '.join(row['clause'].lower().split())] = row

    def snapshot(self):
        return copy.deepcopy(list(self.clauses.values())[-8:])


class RecoveryEvents:

    def __init__(self):
        self.events = []
        self.active = None

    def observe(self, decision, frame, verdict, execution, evidence):
        problem = verdict['motion'] == 'not_supported' or verdict['route'] == 'contradicted'
        if self.active is None and problem:
            self.active = {'id':len(self.events), 'start_decision':decision, 'start_frame':frame,
                           'state':'active', 'cause':copy.deepcopy(verdict), 'transitions':[],
                           'verification':[], 'candidate_workflow_observed':False}
            self.events.append(self.active)
        if self.active is not None:
            self.active['verification'].append({'decision':decision,'frame':frame, **copy.deepcopy(verdict)})
            if decision > self.active['start_decision'] and verdict['motion'] == 'supported':
                self.active['motion_restored_frame']=frame
            if decision > self.active['start_decision'] and verdict['route'] == 'supported':
                self.active['route_reentry_frame']=frame
            if self.active.get('motion_restored_frame') is not None and self.active.get('route_reentry_frame') is not None:
                self.active.update(state='verified_recovered', end_decision=decision, end_frame=frame)
                self.active = None
            elif decision - self.active['start_decision'] >= 6:
                self.active.update(state='budget_exit_unverified', end_decision=decision, end_frame=frame)
                self.active = None

    def record_action(self, decision, execution, workflow_active):
        if self.active is not None:
            self.active['transitions'].append({'decision':decision,'execution':copy.deepcopy(execution)})
            self.active['candidate_workflow_observed'] |= bool(workflow_active)

    def finish(self, decision, frame):
        if self.active is not None:
            self.active.update(state='episode_end_unverified', end_decision=decision, end_frame=frame)
            self.active = None

    def summary(self):
        recovered = [e for e in self.events if e['state'] == 'verified_recovered']
        motion = [e for e in self.events if e.get('motion_restored_frame') is not None]
        route = [e for e in self.events if e.get('route_reentry_frame') is not None]
        with_workflow = [e for e in self.events if e['candidate_workflow_observed'] and any(t['execution'].get('actual') for t in e['transitions'])]
        return {'triggers':len(self.events), 'executed_interventions':len(with_workflow),
                'motion_restored':len(motion), 'route_verified':len(route),
                'route_reentry_claims':0, 'verified_events':len(recovered),
                'unverified_events':len(self.events)-len(recovered),
                'interpretation':'Counts complete events. Workflow participation is not causal proof; visual verdicts are fallible.'}
