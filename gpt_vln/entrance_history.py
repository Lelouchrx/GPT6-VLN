

import numpy as np

from gpt_vln.target_tracker import TargetAnchorCritic


class EntranceGraph:
    def __init__(self):
        self.candidates = []
        self.next_id = 0

    def observe_candidates(self, proposals, obs, sim, frame, place_id):
        for proposal in proposals:

            if proposal['instruction_relevance'] == 'unlikely':
                continue
            box = proposal['bbox_norm']
            if box is None:
                continue
            point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            tracker = TargetAnchorCritic()
            measured = tracker.bind({'description': proposal['description'],
                                     'view': proposal['view'], 'point_norm': point,
                                     'status': 'seen'}, obs, sim, frame)
            match = None
            if tracker.anchor is not None:
                for item in reversed(self.candidates):
                    old = item['tracker'].anchor
                    if old is not None and np.linalg.norm((old['xyz']-tracker.anchor['xyz'])[[0, 2]]) < 1.8:
                        match = item
                        break
            if match is None:
                match = {'id': f'entrance_{self.next_id:02d}', 'first_seen_frame': frame,
                         'from_visited_node': place_id, 'tracker': tracker,
                         'status': 'unverified' if measured['status'] == 'tracked' else 'unknown',
                         'previous_bearing': None, 'last_seen_frame': frame}
                self.next_id += 1
                self.candidates.append(match)
            elif tracker.anchor is not None:
                match['tracker'] = tracker
                if match['status'] in ('archived', 'unknown'):
                    match['status'] = 'unverified'
            match.update(description_hypothesis=proposal['description'],
                         destination_hypothesis=proposal['possible_destination'],
                         instruction_relevance=proposal['instruction_relevance'],
                         last_seen_frame=frame)
            if measured['status'] == 'tracked':
                match['previous_bearing'] = measured['bearing_deg_right']
            self._limit_active(sim)

    def _limit_active(self, sim):
        active = [item for item in self.candidates if item['status'] != 'archived']
        while len(active) > 3:

            victim = min(active, key=lambda item: (
                item['instruction_relevance'] == 'potentially_relevant',
                item['status'] == 'possibly_passed', item['last_seen_frame']))
            victim['status'] = 'archived'
            active.remove(victim)

    def after_primitive(self, sim, action, translation_m):
        alerts = []
        for item in self.candidates:
            if item['status'] == 'archived':
                continue
            measured = item['tracker'].observe(sim)
            if measured['status'] != 'tracked':
                continue
            before = item['previous_bearing']
            now = measured['bearing_deg_right']
            item['previous_bearing'] = now
            if (item['status'] in ('unverified', 'active') and action == 1 and translation_m > .08
                    and before is not None and abs(before) < 85 and abs(now) > 105):
                item['status'] = 'possibly_passed'
                if item['instruction_relevance'] == 'potentially_relevant':
                    alerts.append({'candidate_id': item['id'], 'first_seen_frame': item['first_seen_frame'],
                                   'reason': 'unconfirmed relevant entrance moved behind after measured travel'})
        return alerts

    def context(self, sim):
        rows = []
        selected = [item for item in self.candidates if item['status'] != 'archived']
        for item in selected:
            measured = item['tracker'].observe(sim)
            rows.append({'candidate_id': item['id'], 'from_visited_node': item['from_visited_node'],
                         'first_seen_frame': item['first_seen_frame'],
                         'description_hypothesis': item['description_hypothesis'],
                         'destination_hypothesis': item['destination_hypothesis'],
                         'instruction_relevance': item['instruction_relevance'],
                         'status': item['status'], 'geometry': measured})
        return {'predicted_entrances': rows, 'archived_count':len(self.candidates)-len(selected),
                'confirmed_edges': 'Only observed_semantic_map.edges records actually walked connections',
                'meaning': 'Entrances and destinations are Claude hypotheses. Geometry does not verify room identity or traversability.'}
