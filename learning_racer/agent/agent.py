import torch
import torchvision
from torch.functional import F
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
        return action

    def _preprocess_observation(self, observation):
        """
        Preprocess an observation from the environment.
        :param observation: NumPy array of shape (160, 120, 3)
        :return: Tensor of shape (3, 160, 80)
        """
        observe = PIL.Image.fromarray(observation)
        observe = observe.resize((160, 120))
        croped = observe.crop((0, 40, 160, 120))
        o = transforms.ToTensor()(croped)
        # o = torch.from_numpy(croped.astype(np.float32)).clone()
        return o

    def _encode_image(self, img_t):
        """
        Encode an image into a latent vector.
        :param img_t: Tensor of shape (3, 160, 120) on device
        :return: Tensor of shape (1, z_dim) on device
        """
        # observe = PIL.Image.fromarray(image)
        # observe = observe.resize((160,120))
        # croped = observe.crop((0, 40, 160, 120))
        # #self.teleop.set_current_image(croped)
        # tensor = transforms.ToTensor()(croped)
        z, _, _ = self.vae.encode(torch.unsqueeze(img_t, dim=0))
        return z.detach()

    def _decode_image(self, z):
        """
         Decode a latent vector into an image.

        :param z: Tensor of shape (1, z_dim) on device
        :return (mu_image,sigma_y): Turple tensor of shape (3, 160, 120) on device
        """
        mu_image, sigma_y = self.vae.decode(z)
        return mu_image.detach(), sigma_y.detach()

    def _postprocess_observe(self, z, action):
        """
        Concatenate the latent vector and the action history.
        :param z: Tensor of shape (1, z_dim) on device
        :param action: numpy
        :return: np.array of shape (1, z_dim + n_commands * n_command_history)
        """
        self._record_action(action)
        z = z.cpu().numpy()[0]
        # observe = self._encode_image(observe)
        if self.n_command_history > 0:
            z = np.concatenate([z, np.asarray(self.action_history)], 0)
        return z

    def _is_auto_stop(self, reconst_img, sigma_y, observe_img):
        """
        Calculate the difference between the reconstructed image and the original image.
        :param reconst_img: Tensor of shape (160, 120, 3)
        :param observe_img: Tensor of shape (160, 120, 3)
        :return:
        """
        m_vae_loss = (observe_img - reconst_img) ** 2 / sigma_y
        m_vae_loss = 0.5 * torch.sum(m_vae_loss)
        # bce_loss = torch.mean(torch.sum(observe_img*torch.log(reconst_img)+(1-observe_img)*torch.log(1-reconst_img), dim=1))
        # bce_loss = F.binary_cross_entropy(reconst_img.view(-1,38400), observe_img.view(-1,38400), reduction='sum')
        print(m_vae_loss)
        return m_vae_loss.item() > self.config.vae_auto_stop_threshold()

    def step(self, action):

        action = self._preprocess_action(action)
        observe, reward, done, e_i = self._wrapped_env.step(action)
        img_t = self._preprocess_observation(observe).to(self.device)
        z = self._encode_image(img_t)
        observe = self._postprocess_observe(z, action)
        reconst_img, sigma_y = self._decode_image(torch.unsqueeze(z, dim=0))

        if self._is_auto_stop(reconst_img, sigma_y, torch.unsqueeze(img_t, dim=0)):
            print("Auto stop")
        # done = self._is_auto_stop(reconst_img, sigma_y, torch.unsqueeze(img_t, dim=0))

        # Override Done event.
        if self.teleop is not None:
            # Teleop.status == True is a stop signal.
            done = self.teleop.status

        if self.reward_callback is not None:
            # Override reward.
            reward, done = self.reward_callback(action, e_i, done)

        if done and self.train:
            self._wrapped_env.step(np.array([0.,0.]))
            if self.teleop is not None:
                # slf.teleop.send_status == False is a STOP signal.
                # Change disable checkbox on notebook.
                self.teleop.send_status(False)

        return observe, reward, done, e_i

    def reset(self):
        print('====RESET')
        # Waiting RESET for teleoperation.
        self.action_history = [0.] * (self.n_command_history * self.n_commands)
        observe = self._wrapped_env.reset()
        img_t = self._preprocess_observation(observe).to(self.device)
        o = self._encode_image(img_t)
        o = o.cpu().numpy()[0]
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
