import time

import torch

import PIL

import numpy as np

from gym import Env, spaces
from torchvision.transforms import transforms


class Agent(Env):

    def __init__(self, _wrapped_env, vae, teleop, device, reward_callback=None, config=None, train=True):

        self.config = config
        self.train = train
        self._wrapped_env = _wrapped_env
        self.vae = vae
        self.z_dim = vae.z_dim
        self.teleop = teleop
        self.device = device
        self.reward_callback = reward_callback
        self.n_commands = 2
        self.n_command_history = config.agent_n_command_history()
        self.reward_callback = reward_callback
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(self.z_dim + (self.n_commands * self.n_command_history), ),
                                            dtype=np.float32)
        self.action_history = [0.] * (self.n_command_history * self.n_commands)
        self.action_space = spaces.Box(low=np.array([config.agent_min_steering(), -1]),
                                       high=np.array([config.agent_max_steering(), 1]), dtype=np.float32)
        if self.teleop is not None:
            self.teleop.start_process()

    def _calc_reward(self, action, done, i_e):
        pass

    def _record_action(self, action):

        if len(self.action_history) >= self.n_command_history * self.n_commands:
            del self.action_history[:2]
        for v in action:
            self.action_history.append(v)

    def _scaled_action(self, action):
        #Convert from [-1, 1] to [0, 1]
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
        self._record_action(action)
        return action

    def _encode_image(self, image):
        observe = PIL.Image.fromarray(image)
        observe = observe.resize((160,120))
        croped = observe.crop((0, 40, 160, 120))
        #self.teleop.set_current_image(croped)
        tensor = transforms.ToTensor()(croped)
        tensor.to(self.device)
        z, _, _ = self.vae.encode(torch.stack((tensor,tensor),dim=0)[:-1].to(self.device))
        return z.detach().cpu().numpy()[0]

    def _postprocess_observe(self,observe, action):
        self._record_action(action)
        observe = self._encode_image(observe)
        if self.n_command_history > 0:
            observe = np.concatenate([observe, np.asarray(self.action_history)], 0)
        return observe


    def step(self, action):

        action = self._preprocess_action(action)
        observe, reward, done, e_i = self._wrapped_env.step(action)
        observe = self._postprocess_observe(observe,action)

        #Override Done event.
        if self.teleop is not None:
            done = self.teleop.status

        if self.reward_callback is not None:
            #Override reward.
            reward = self.reward_callback(action, e_i, done)

        if done and self.train:
            self._wrapped_env.step(np.array([0.,0.]))
            if self.teleop is not None:
                self.teleop.send_status(False)

        return observe, reward, done, e_i

    def reset(self):
        print('====RESET')
        # Waiting RESET for teleoperation.
        if self.teleop is not None:
            self.teleop.send_status(True)
            message = True
            while self.teleop.status:
                if message:
                    print('Press START')
                message = False
                time.sleep(0.1)

        self.action_history = [0.] * (self.n_command_history * self.n_commands)
        observe = self._wrapped_env.reset()
        o = self._encode_image(observe)
        if self.n_command_history > 0:
            o = np.concatenate([o, np.asarray(self.action_history)], 0)
        return o


    def render(self):
        self._wrapped_env.render()

    def close(self):
        self._wrapped_env.close()

    def seed(self, seed=None):
        return self._wrapped_env.seed(seed)

    def jerk_penalty(self):
        """
        Add a continuity penalty to limit jerk.
        :return: (float)
        """
        jerk_penalty = 0
        if self.n_command_history > 1:
            # Take only last command into account
            for i in range(1):
                steering = self.action_history[-2 * (i + 1)]
                prev_steering = self.action_history[-2 * (i + 2)]
                steering_diff = (prev_steering - steering) / (
                            self.config.agent_max_steering - self.config.agent_min_steering)

                if abs(steering_diff) > self.config.agent_max_steering_diff:
                    error = abs(steering_diff) - self.config.agent_max_steering_diff
                    jerk_penalty += 0.00 * (error ** 2)
                else:
                    jerk_penalty += 0
        return jerk_penalty
