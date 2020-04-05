import torch
from agent.agent import Agent
from config import VARIANTS_SIZE, IMAGE_CHANNELS, VERBOSE, BATCH_SIZE, BUFFER_SIZE, LEARNING_STARTS, GRADIENT_STEPS, \
    TRAIN_FREQ, ENT_COEF, LEARNING_RATE, LOG_INTERVAL
from robot import JetbotEnv, JetRacerEnv
from sac import CustomSAC, CustomSACPolicy, reward
from teleoperate import Teleoperator
from vae.vae import VAE

robot_drivers = {'jetbot':JetbotEnv, 'jetracer':JetRacerEnv}

def _load_vae(model_path, variants_size, image_channels, device):
    vae = VAE(image_channels=image_channels, z_dim=variants_size)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    vae.to(torch.device(device)).eval()
    return vae


def _create_agent(robot_driver, vae, torch_device):
    env = robot_driver()
    teleop = Teleoperator()
    agent = Agent(env, vae, teleop=teleop, device=torch_device, reward_callback=reward)
    return agent


def _init_agent(args):
    torch_device = args.device
    vae = _load_vae(args.vae_path, VARIANTS_SIZE, IMAGE_CHANNELS, torch_device)
    agent = _create_agent(robot_drivers[args.robot_driver], vae, torch_device)
    return agent


def _generate_save_callbask(args):
    save_freq_episode = args.save_freq_episode
    path = args.save

    def _save_callback(locals, globals):
        num_episodes = len(locals['episode_rewards'])
        if locals['self'].num_timesteps > locals['self'].learning_starts and \
                num_episodes % save_freq_episode == 0 and locals['done']:
            locals['self'].save(path + '_' + str(num_episodes) + '.zip')

        return True

    return _save_callback

def command_train(args):

    agent = _init_agent(args)
    model = CustomSAC(CustomSACPolicy, agent, verbose=VERBOSE, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE,
                learning_starts=LEARNING_STARTS, gradient_steps=GRADIENT_STEPS, train_freq=TRAIN_FREQ,
                ent_coef=ENT_COEF, learning_rate=LEARNING_RATE)
    save_callback = _generate_save_callbask(args)

    model.learn(total_timesteps=args.time_steps, log_interval=LOG_INTERVAL, callback=save_callback)
    '''
    WARNING.
    Normal SAC in stable baselines but code is changed to calculate gradient only when done episode.
    In gym_donkey, skip_frame parameter is 2 but modify to 1. 
    '''
    model.save(args.save)


def command_demo(args):
    agent = _init_agent(args)
    model = CustomSAC.load(args.model_path)
    obs = agent.reset()
    for step in range(args.time_steps):
        if step % 100 == 0: print("step: ", step)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = agent.step(action)
