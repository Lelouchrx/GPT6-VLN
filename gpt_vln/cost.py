"""Account for provider cost counters that are cumulative within a session."""


class SessionCostLedger:
    def __init__(self):
        self.totals = {}

    def add(self, response, fallback_key):
        cumulative = response.get('cost_usd')
        if not isinstance(cumulative, (int, float)) or cumulative < 0:
            return 0.0
        key = response.get('session_id') or fallback_key
        previous = self.totals.get(key, 0.0)
        self.totals[key] = max(previous, float(cumulative))
        return max(0.0, float(cumulative) - previous)
