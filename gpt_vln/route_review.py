
import copy
from gpt_vln.context import execution


class RouteReviewMemory:
    def __init__(self, interval=8):
        if type(interval) is not int or interval < 1:
            raise ValueError('Review interval must be a positive integer')
        self.interval = interval
        self.keyframes = {}
        self.latest = None
        self.reviews = []

    def due(self, decision):

        return decision > 0 and decision % self.interval == 0

    def remember(self, record, previous=None):
        reasons = []
        if previous is None:
            reasons.append('start')
        if previous is not None and record['stage'] != previous['stage']:
            reasons.append('actor_claimed_stage_change')
        out = record.get('output', {})
        if out.get('return_anchor_frame') is not None:
            reasons.append('branch_return_reference')
        if out.get('maneuver') in ('climb_up', 'climb_down', 'return_to_anchor'):
            reasons.append(out['maneuver'])
        if record.get('events'):
            reasons.append('actor_cited_route_event')
        if reasons:
            self.keyframes[record['id']] = {
                'frame_id': record['id'], 'reasons': reasons,
                'description_hypothesis': record.get('place', ''),
                'verification': 'unverified actor hypothesis; image is recorded evidence',
                'image': copy.deepcopy(record['image']),
            }

    def select(self, history, anchor=None, limit=6):
        by_id = {r['id']: r for r in history}

        ids = []
        if anchor in by_id:
            ids.append(anchor)
        proposed = (self.latest or {}).get('suggested_return_anchor_frame')
        if proposed in by_id:
            ids.append(proposed)
        keys = sorted(self.keyframes)
        if keys:
            ids.append(keys[0])
            ids.extend(reversed(keys))
        selected = []
        for fid in ids:
            if fid in by_id and fid not in selected:
                selected.append(fid)
            if len(selected) >= limit:
                break
        return [dict(by_id[fid]['image'], label=f'KEY frame {fid} PANORAMA saved route evidence')
                for fid in sorted(selected)]

    def accept(self, verdict, decision, supplied_frames, history):
        value = copy.deepcopy(verdict)
        if value.get('route') not in ('supported', 'contradicted', 'unknown'):
            raise ValueError('Invalid route review verdict')
        if not set(value.get('frames', [])) <= set(supplied_frames):
            raise ValueError('Reviewer cited an image it did not receive')
        target = value.get('suggested_return_anchor_frame')
        prior = {r['id'] for r in history}
        if target is not None and (type(target) is not int or target not in prior
                                   or target not in supplied_frames):
            raise ValueError('Reviewer return target must be a supplied historical image')
        if value['route'] != 'unknown' and not value.get('frames'):
            raise ValueError('Non-unknown judgment needs visual citations')
        value.update(checked_before_decision=decision, authority='fallible visual review; no action or STOP veto')
        self.latest = value
        self.reviews.append(copy.deepcopy(value))

    def context(self, decision, history, anchor=None):
        verdict = copy.deepcopy(self.latest)
        if verdict:
            verdict['age_in_decisions'] = decision - verdict['checked_before_decision']
        ids = [r['id'] for r in history]
        start = ids.index(anchor) if anchor in ids else max(0, len(history)-8)

        trail = [{'frame': r['id'], 'executed_after_frame': execution(r['last_execution'])}
                 for r in history[start:]]
        return {'interval_decisions': self.interval, 'latest_review': verdict,
                'keyframe_index': [{k: v for k, v in row.items() if k != 'image'}
                                   for row in self.keyframes.values()],
                'return_trail': trail,
                'return_policy': 'Unknown is not deviation. Compare supplied anchor RGB, choose short visible segments; never invert commands or claim route recovery from motion alone.',
                'return_verified': False}


def merge_images(*groups):
    result = []
    seen = set()
    for group in groups:
        for im in group:
            key = (im['frame'], im['source'])
            if key not in seen:
                seen.add(key)
                result.append(im)
    return result
