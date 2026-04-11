import abc
import random
import time

from core.lib.common import ClassFactory, ClassType, Context, ConfigLoader, LOGGER
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('LoadBalanceAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='load_balance')
class LoadBalanceAgent(BaseAgent, abc.ABC):
    """
    缺陷感知负载均衡调度 Agent。

    空闲态（无缺陷）：使用默认 fps 与固定卸载节点。
    告警态（检测到 bbox）：切换至告警 fps，并按权重随机分发至多个节点，实现负载均衡。
    告警态持续 alert_duration 秒；若期间无新缺陷触发，超时后自动回归空闲态。
    """

    def __init__(self, system, agent_id: int,
                 default_fps: int = 3,
                 alert_fps: int = 6,
                 buffer_size: int = 1,
                 default_target: str = 'edgexn13',
                 service_name: str = 'v5lite-detection',
                 lb_targets: list = None,
                 lb_weights: list = None,
                 alert_duration: float = 3.0):
        super().__init__()

        self.agent_id = agent_id
        self.cloud_device = system.cloud_device

        self.default_fps = default_fps
        self.alert_fps = alert_fps
        self.buffer_size = buffer_size
        self.default_target = default_target
        self.service_name = service_name
        self.lb_targets = lb_targets if lb_targets is not None else ['cloud.kubeedge', 'edgex3', 'edgexn13']
        self.lb_weights = lb_weights if lb_weights is not None else [1, 1, 1]
        self.alert_duration = alert_duration

        # 最近一次检测到缺陷的时间戳；None 表示从未触发
        self._last_defect_time: float = None

        self.overhead_estimator = OverheadEstimator('LoadBalance', 'scheduler/load_balance')

    def get_schedule_plan(self, info):
        with self.overhead_estimator:
            cloud_device = self.cloud_device
            source_edge_device = info['source_device']
            all_edge_devices = info['all_edge_devices']
            all_devices = [*all_edge_devices, cloud_device]

            dag = info['dag']

            in_alert = (self._last_defect_time is not None
                        and time.time() - self._last_defect_time < self.alert_duration)

            if in_alert:
                fps = self.alert_fps
                candidate = random.choices(self.lb_targets, weights=self.lb_weights, k=1)[0]
                target = candidate if candidate in all_devices else cloud_device
                remaining = self.alert_duration - (time.time() - self._last_defect_time)
                LOGGER.info(f'[LoadBalance] Alert state → fps={fps}, target={target}, '
                            f'expires in {remaining:.1f}s')
            else:
                fps = self.default_fps
                target = self.default_target if self.default_target in all_devices else cloud_device
                LOGGER.info(f'[LoadBalance] Idle state → fps={fps}, target={target}')

            for service_name in dag:
                if service_name == 'start':
                    dag[service_name]['service']['execute_device'] = source_edge_device
                elif service_name == self.service_name:
                    dag[service_name]['service']['execute_device'] = target
                else:
                    dag[service_name]['service']['execute_device'] = cloud_device

            policy = {
                'fps': fps,
                'buffer_size': self.buffer_size,
                'dag': dag,
            }

        return policy

    def run(self):
        pass

    def update_scenario(self, scenario):
        obj_num = scenario.get('obj_num', [])
        if obj_num and any(n > 0 for n in obj_num):
            self._last_defect_time = time.time()

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
