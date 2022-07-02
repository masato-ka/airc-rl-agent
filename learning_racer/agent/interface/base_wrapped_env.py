from abc import abstractmethod
import numpy as np
import torch
from gym import Env
from gym.vector.utils import spaces
from stable_baselines3.common.callbacks import BaseCallback

from learning_racer.agent.utils import pre_process_image
from learning_racer.config import ConfigReader
from learning_racer.vae import VAE


class BaseWrappedEnv(Env):

    def __init__(self, env: Env, vae: VAE, config: ConfigReader):
        super(BaseWrappedEnv, self).__init__()
        self.vae = vae
        self.env = env
        self.device = next(vae.parameters()).device
        self.config = config
        self.train = True
        self.z_dim = vae.z_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_commands = 2
        self.n_command_history = config.agent_n_command_history()
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(self.z_dim + (self.n_commands * self.n_command_history),),
                                            dtype=np.float32)
        self.action_history = [0.] * (self.n_command_history * self.n_commands)
        self.action_space = spaces.Box(low=np.array([config.agent_min_steering(), -1]),
                                       high=np.array([config.agent_max_steering(), 1]), dtype=np.float32)
        self.callbacks = None

    @abstractmethod
    def on_training_end(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_rollout_start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_pre_step_callback(self, action):
        raise NotImplementedError

    @abstractmethod
    def on_post_step_callback(self, action, observe, reward, done, info, z, train):
        raise NotImplementedError

    @abstractmethod
    def on_pre_reset(self):
        raise NotImplementedError

    @abstractmethod
    def on_post_reset(self, observe):
        raise NotImplementedError

    def train(self):
        self.train = True

    def eval(self):
        self.train = False

    def step(self, action):
        action = self.on_pre_step_callback(action)
        action = self._preprocess_action(action)

        observe, reward, done, e_i = self.env.step(action)

        self._record_action(action)
        z, t_img = self.encode_observe(observe)
        z = torch.squeeze(z.detach()).cpu().numpy()
        observe_action_history = self._concat_action_history(z, self.action_history)

        action, t_img, reward, done, info, z = \
            self.on_post_step_callback(action, t_img, reward, done, e_i, z, self.train)

        return observe_action_history, reward, done, e_i

    def reset(self):
        print('====RESET')
        # Waiting RESET for teleoperation.
        self.on_pre_reset()
        self.action_history = [0.] * (self.n_command_history * self.n_commands)
        observe = self.env.reset()
        z, _ = self.encode_observe(observe)
        z = torch.squeeze(z.detach()).cpu().numpy()
        if self.n_command_history > 0:
            z = np.concatenate([z, np.asarray(self.action_history)], 0)
        observe = self.on_post_reset(observe)
        return z

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def _record_action(self, action):
        if len(self.action_history) >= self.n_command_history * self.n_commands:
            del self.action_history[:2]
        for v in action:
            self.action_history.append(v)

    def _scaled_action(self, action):
        # Convert from [-1, 1] to [0, 1]
        t = (action[1] + 1) / 2
        action[1] = (1 - t) * self.config.agent_min_throttle() + self.config.agent_max_throttle() * t
        return action

    def _smoothing_action(self, action):
        if self.n_command_history > 0:
            prev_steering = self.action_history[-2]
            max_diff = (self.config.agent_max_steering_diff() - 1e-5) * (
                    self.config.agent_max_steering() - self.config.agent_min_steering())
            diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
            action[0] = prev_steering + diff
        return action

    def _preprocess_action(self, action):
        action = self._scaled_action(action)
        action = self._smoothing_action(action)
        return action

    def _concat_action_history(self, z, action_history):
        observe_action_history = None
        if self.n_command_history > 0:
            observe_action_history = np.concatenate([z, np.asarray(action_history)], 0)
        return observe_action_history

    def encode_observe(self, observe):
        t_img = pre_process_image(observe)
        z, _, _ = self.vae.encode(t_img.to(self.device))
        return z, t_img


class StableBaselineCallback(BaseCallback):

    def __init__(self, wrapped_env: BaseWrappedEnv):
        super(StableBaselineCallback, self).__init__()
        self.wrapped_env = wrapped_env

    def on_rollout_start(self) -> None:
        self.wrapped_env.on_rollout_start()

    def on_training_end(self) -> None:
        self.wrapped_env.on_training_end()

    def _on_step(self) -> bool:
        return True
