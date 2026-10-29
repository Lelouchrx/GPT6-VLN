from typing import Any

from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import DistanceToGoal


@registry.register_measure
class OracleNavigationError(Measure):
    cls_uuid = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = min(self._metric, float(distance))


@registry.register_measure
class OracleSuccess(Measure):
    cls_uuid = "oracle_success"

    def __init__(self, *args: Any, config: Any, **kwargs: Any) -> None:
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        threshold = float(getattr(self._config, "success_distance", 3.0))
        self._metric = float(bool(self._metric) or distance < threshold)
