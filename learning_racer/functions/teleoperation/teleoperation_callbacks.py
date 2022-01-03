import time

import numpy as np
from functions.base import BaseAgentCallbacks

from logging import getLogger

logger = getLogger(__name__)


def real_world_reward(action, done, min_throttle, max_throttle,
                      crash_reward, crash_reward_weight, throttle_reward_weight):
    """

    :param action: tuple of throttle and steering
    :param done: boolean
    :param min_throttle: float
    :param max_throttle: float
    :param crash_reward: float
    :param crash_reward_weight: float
    :param throttle_reward_weight: float
    :return: float and boolean
    """
    if done:
        norm_throttle = (action[1] - min_throttle) / (
                max_throttle - min_throttle)
        return crash_reward - (crash_reward_weight * norm_throttle), done
    throttle_reward = throttle_reward_weight * (action[1] / max_throttle)
    return 1 + throttle_reward, done


class TeleoperationCallbacks(BaseAgentCallbacks):

    def __init__(self, agent, config, teleoperator):
        super(TeleoperationCallbacks, self).__init__(agent, config)
        self.teleoperator = teleoperator

    # StableBaselines3 Callbacks
    def on_rollout_start(self) -> None:
        if self.teleoperator is not None:
            self.teleoperator.send_status(True)
            message = True
            while self.teleoperator.status:
                if message:
                    logger.info("Press START.")
                message = False
                time.sleep(0.1)

    def on_pre_step_callback(self, action):
        return action

    def on_post_step_callback(self, action, observe, reward, done, info, z):
        # Override Done event.
        done = self._done_override(action, observe, reward, done, info, z)
        # Override Reward value.
        reward, done = real_world_reward(action, done, self.config.min_throttle(), self.config.max_throttle(),
                                         self.config.crash_reward(), self.config.crash_reward_weight(),
                                         self.config.throttle_reward_weight())

        return action, observe, reward, done, info, z

    def _done_override(self, action, observe, reward, done, info, z):
        if self.teleop is not None:
            done = self.teleop.status
            if done and self.train:
                self.env.step(np.array([0., 0.]))
                self.teleop.send_status(False)
        return done

    def on_pre_reset(self):
        return

    def on_post_reset(self, observe):
        return observe
