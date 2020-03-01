import argparse
import torch
from stable_baselines import SAC
from agent.agent import Agent
from config import MIN_THROTTLE, MAX_THROTTLE, REWARD_CRASH, CRASH_REWARD_WEIGHT, THROTTLE_REWARD_WEIGHT, VARIANTS_SIZE, \
    IMAGE_CHANNELS
from robot import JetbotEnv
from robot import JetRacerEnv
from teleoperate import Teleoperator
from vae.vae import VAE


robot_drivers = {'jetbot':JetbotEnv, 'jetracer':JetRacerEnv}

def calc_reward(action, e_i, done):
    if done:
        norm_throttle = (action[1] - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
        return REWARD_CRASH - (CRASH_REWARD_WEIGHT * norm_throttle)
    throttle_reward = THROTTLE_REWARD_WEIGHT * (action[1] / MAX_THROTTLE)
    return 1 + throttle_reward

def load_vae(model_path, variants_size, image_channels, device):
    vae = VAE(image_channels=image_channels, z_dim=variants_size)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    vae.to(torch.device(device)).eval()
    return vae

def create_agent(robot_driver, vae, torch_device):
    env = robot_driver()
    teleop = Teleoperator()
    agent = Agent(env, vae, teleop=teleop, device=torch_device, reward_callback=calc_reward)
    return agent

parser = argparse.ArgumentParser()

parser.add_argument('-vae', '--vae-path', help='Path to a trained vae model path.',
                    default='vae.torch', type=str)
parser.add_argument('-model', '--model-path', help='Path to a trained vae model path.',
                    default='model', type=str)
parser.add_argument('-device', '--device', help='torch device {"cpu" | "cuda"}',
                    default='cuda', type=str)
parser.add_argument('-robot', '--robot-driver', help='choose robot driver',
                    default='jetbot', type=str)
parser.add_argument('-steps', '--time-steps', help='total step.',
                    default='5000', type=int)


if __name__ == '__main__':

    args = parser.parse_args()

    torch_device = args.device

    vae = load_vae(args.vae_path, VARIANTS_SIZE, IMAGE_CHANNELS, torch_device)
    agent = create_agent(robot_drivers[args.robot_driver], vae, torch_device)

    model = SAC.load(args.model_path)

    obs = agent.reset()
    dones=False
    for step in range(args.time_steps): # 500ステップ実行
        if step % 100 == 0: print("step: ", step)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = agent.step(action)
