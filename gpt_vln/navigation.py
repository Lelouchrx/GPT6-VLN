
import copy

import gpt_vln.runtime as runtime
from gpt_vln.model_runtime import RGBBackend as BaseBackend


class RGBBackend(BaseBackend):
    def _codex_request(self, payload, directory, timeout, adapter):
        if adapter == 'gpt_vln/model.py':
            payload = dict(payload, horizon_mode=self.config['horizon_mode'])
            adapter = 'gpt_vln/action_policy.py'
        return super()._codex_request(payload, directory, timeout, adapter)

    def evaluate(self, version, task, directory):
        if self.config['horizon_mode'] == 'free':
            return super().evaluate(version, task, directory)
        original = runtime.validate_action

        def horizon_validate(output, instruction, available_frames=None, history_frames=None):
            if output.get('maneuver') not in ('climb_up', 'climb_down'):
                return original(output, instruction, available_frames, history_frames)


            ordinary = copy.deepcopy(output)
            ordinary['maneuver'] = 'travel'
            original(ordinary, instruction, available_frames, history_frames)
            return output

        runtime.validate_action = horizon_validate
        try:
            return super().evaluate(version, task, directory)
        finally:
            runtime.validate_action = original
