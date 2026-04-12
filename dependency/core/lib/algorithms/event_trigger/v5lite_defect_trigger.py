import abc
import numpy as np
from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task
from .base_trigger import BaseTrigger

__all__ = ('V5LiteDefectTrigger',)


@ClassFactory.register(ClassType.EVENT_TRIGGER, alias='v5lite_defect')
class V5LiteDefectTrigger(BaseTrigger, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):
        obj_num = task.get_scenario_data().get('obj_num', [])
        if not obj_num:
            return False, {}
        total = int(np.sum(obj_num))
        if total > 0:
            return True, {'检测到缺陷数量': total}
        return False, {}
