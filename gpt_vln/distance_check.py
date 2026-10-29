


class GoalDistanceCheckpoint:
    def __init__(self, interval=10, regression_tolerance_m=0.5):
        if type(interval) is not int or interval < 1:
            raise ValueError('interval must be a positive integer')
        if regression_tolerance_m < 0:
            raise ValueError('regression tolerance cannot be negative')
        self.interval = interval
        self.regression_tolerance_m = float(regression_tolerance_m)
        self.initial_distance_m = None
        self.initial_decision = None
        self.samples = []

    def set_initial(self, distance_m, completed_decisions=0):
        if self.initial_distance_m is None and distance_m is not None:
            self.initial_distance_m = float(distance_m)
            self.initial_decision = int(completed_decisions)
        return self.initial_distance_m

    def check(self, completed_decisions, anchor_observation):
        if completed_decisions < 1 or completed_decisions % self.interval:
            return None
        if anchor_observation.get('status') != 'tracked':
            row = {'completed_decisions': completed_decisions, 'status': 'unknown',
                   'distance_m': None, 'regression_m': None,
                   'initial_distance_m': self.initial_distance_m,
                   'stop': False, 'reason': 'task endpoint has no stable RGB-D anchor'}
            self.samples.append(row)
            return row
        distance = float(anchor_observation['distance_xz_m'])
        self.set_initial(distance, completed_decisions)
        regression = distance - self.initial_distance_m
        stop = regression is not None and regression > self.regression_tolerance_m
        row = {'completed_decisions': completed_decisions, 'status': 'tracked',
               'distance_m': round(distance, 2),
               'initial_distance_m': round(self.initial_distance_m, 2),
               'regression_m': round(regression, 2),
               'baseline_decision': self.initial_decision, 'stop': stop,
               'reason': ('measured task-endpoint distance increased beyond tolerance'
                          if stop else 'no checkpoint regression beyond tolerance')}
        self.samples.append(row)
        return row
