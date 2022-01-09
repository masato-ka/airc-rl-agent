import time
import numpy as np
import torch
from logging import getLogger

logger = getLogger(__name__)

from learning_racer.functions.base import BaseAgentCallbacks


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


class AutoStopCallbacks(BaseAgentCallbacks):

    def __init__(self, env, config, teleoperator, vae):
        super(AutoStopCallbacks, self).__init__(env, config)
        self.vae = vae
        self.teleoperator = teleoperator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def _is_auto_stop(self, reconst, sigma, observe_img):
        """
        Calculate the difference between the reconstructed image and the original image.
        :param reconst: Tensor of shape (160, 120, 3)
        :param observe_img: Tensor of shape (160, 120, 3)
        :return:
        """
        m_vae_loss = (observe_img - reconst) ** 2 / sigma
        m_vae_loss = 0.5 * torch.sum(m_vae_loss)
        print(m_vae_loss.item())
        return m_vae_loss.item() > self.config.vae_auto_stop_threshold()

    def _decode_image(self, z):
        """
        Decode a latent vector into an image.
        :param z: Tensor of shape (1, z_dim) on device
        :return (mu_image,sigma_y): Turple tensor of shape (3, 160, 120) on device
        """
        mu_image, sigma_y = self.vae.decode(z)
        return mu_image.detach(), sigma_y.detach()

    def on_post_step_callback(self, action, t_img, reward, done, info, z, train):

        z = torch.unsqueeze(torch.Tensor(z), dim=0)
        reconst, sigma = self._decode_image(z.to(self.device))
        if self._is_auto_stop(reconst, sigma, t_img.to(self.device)):
            done = True
            self.teleoperator.status = True
            if done and train:
                self.env.step(np.array([0., 0.]))
                self.teleoperator.send_status(False)
        if self.teleoperator.status:
            done = True

        reward, done = real_world_reward(action, done, self.config.agent_min_throttle(),
                                         self.config.agent_max_throttle(),
                                         self.config.reward_reward_crash(), self.config.reward_crash_reward_weight(),
                                         self.config.reward_throttle_reward_weight())

        return action, t_img, reward, done, info, z

    def on_pre_reset(self):
        return

    def on_post_reset(self, observe):
        return observe
