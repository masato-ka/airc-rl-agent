import torch

from agent import SimulatorEnv


class SimulatorAutoStopEnv(SimulatorEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_post_step_callback(self, action, t_img, reward, done, info, z, train):
        z = torch.unsqueeze(torch.Tensor(z), dim=0)
        reconst, sigma = self._decode_image(z.to(self.device))
        done, reward = self.done_and_reward(reconst, sigma, t_img, info['speed'])

        # reward = reward_sim(done, info['speed'], info['cte'], self.config.reward_reward_crash(),
        #                     self.config.reward_crash_reward_weight(), self.config.reward_throttle_reward_weight())
        return action, t_img, reward, done, info, z

    def done_and_reward(self, reconst, sigma, observe_img, speed):
        """
        Calculate the difference between the reconstructed image and the original image.
        :param reconst: Tensor of shape (160, 120, 3)
        :param observe_img: Tensor of shape (160, 120, 3)
        :return:
        """
        m_vae_loss = (observe_img - reconst) ** 2 / sigma
        m_vae_loss = 0.5 * torch.sum(m_vae_loss)
        # Stay trach reward
        done = m_vae_loss.item() > self.config.vae_auto_stop_threshold()

        reward = (self.config.vae_auto_stop_threshold() - m_vae_loss.item()) * \
                 (1 / self.config.vae_auto_stop_threshold())
        norm_speed = speed / 18.0
        reward = reward + self.config.reward_throttle_reward_weight() * norm_speed
        if done:
            reward = self.config.reward_reward_crash() - (self.config.reward_crash_reward_weight() * norm_speed)
        return done, reward

    def _decode_image(self, z):
        """
        Decode a latent vector into an image.
        :param z: Tensor of shape (1, z_dim) on device
        :return (mu_image,sigma_y): Turple tensor of shape (3, 160, 120) on device
        """
        mu_image, sigma_y = self.vae.decode(z)
        return mu_image.detach(), sigma_y.detach()
