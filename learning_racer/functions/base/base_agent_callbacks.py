from abc import ABCMeta, abstractmethod
from logging import getLogger

import gym
from stable_baselines3.common.callbacks import BaseCallback

logger = getLogger(__name__)


class BaseAgentCallbacks(BaseCallback):

    def __init__(self, config, env: gym.Env, verbose=0):
        super(BaseAgentCallbacks, self).__init__(verbose)
        self.config = config
        self.env = env

    @abstractmethod
    def on_pre_step_callback(self, action):
        raise NotImplementedError

    @abstractmethod
    def on_post_step_callback(self, action, observe, reward, done, info, z):
        raise NotImplementedError

    @abstractmethod
    def on_pre_reset(self):
        raise NotImplementedError

    @abstractmethod
    def on_post_reset(self, observe):
        raise NotImplementedError

    # StableBaselines3 Callbacks
    def _on_step(self) -> bool:
        return True

    # StableBaselines3 Callbacks
    def on_training_end(self) -> None:
        logger.info("Training is came to ending. Please shutdown software manually(Ctr+C).")
