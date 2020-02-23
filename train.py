import torch
from stable_baselines import SAC
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
import tensorflow as tf
from agent.agent import Agent
from config import MIN_THROTTLE, MAX_THROTTLE, REWARD_CRASH, CRASH_REWARD_WEIGHT, THROTTLE_REWARD_WEIGHT

from jetbot_env import JetbotEnv

from teleoperate.teleoperation import Teleoperator
from vae.vae import VAE

VARIANTS_SIZE = 32
image_channels = 3

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,layers=[32, 16],act_fun=tf.nn.elu,feature_extraction="mlp")

def calc_reward(action, e_i, done):
    if done:
        norm_throttle = (action[1] - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
        return REWARD_CRASH - (CRASH_REWARD_WEIGHT * norm_throttle)
    throttle_reward = THROTTLE_REWARD_WEIGHT * (action[1] / MAX_THROTTLE)
    return 1 + throttle_reward

if __name__ == '__main__':

    model_path = 'vae.torch'
    torch_device = 'cuda'
    vae = VAE(image_channels=image_channels, z_dim=VARIANTS_SIZE)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device(torch_device)))
    vae.to(torch.device(torch_device))
    vae.eval()

    env = JetbotEnv()
    teleop = Teleoperator()
    agent = Agent(env, vae, teleop=teleop, device=torch_device, reward_callback=calc_reward)
    #
    # model = SAC(CustomSACPolicy, agent, verbose=1, batch_size=64, buffer_size=30000, learning_starts=300,
    #             gradient_steps=600, train_freq=1, ent_coef='auto_0.1', learning_rate=3e-4)
    # model.learn(total_timesteps=30000, log_interval=1)

    agent.reset()
    for step in range(0,100):
        o,r,d,i = agent.step(agent.action_space.sample())
        if d:
            agent.reset()
    print("main ending.")
    exit(0)


    '''
    Normal SAC in stable baselines but code is changed to calculate gradient only when done episode.
    In gym_donkey, skip_frame parameter is 2 but modify to 1. 
    '''
    #model.save('donkey7')
