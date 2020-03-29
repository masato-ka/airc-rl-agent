import torch
from agent.agent import Agent
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


def _create_agent(robot_driver, vae, torch_device, config):
    env = robot_driver()
    teleop = Teleoperator()
    agent = Agent(env, vae, teleop=teleop, device=torch_device, reward_callback=reward, config=config)
    return agent


def _init_agent(args, config):
    torch_device = args.device
    vae = _load_vae(args.vae_path, config.sac_variants_size(), config.sac_image_channel(), torch_device)
    agent = _create_agent(robot_drivers[args.robot_driver], vae, torch_device, config)
    return agent


def command_train(args, config):
    agent = _init_agent(args, config)
    model = CustomSAC(CustomSACPolicy, agent, verbose=config.sac_verbose(), batch_size=config.sac_batch_size(),
                      buffer_size=config.sac_buffer_size(),
                      learning_starts=config.sac_learning_starts(), gradient_steps=config.sac_gradient_steps(),
                      train_freq=config.sac_train_freq(),
                      ent_coef=config.sac_ent_coef(), learning_rate=config.sac_learning_rate())
    model.learn(total_timesteps=args.time_steps, log_interval=config.sac_log_interval())
    '''
    WARNING.
    Normal SAC in stable baselines but code is changed to calculate gradient only when done episode.
    In gym_donkey, skip_frame parameter is 2 but modify to 1. 
    '''
    model.save(args.save)


def command_demo(args, config):
    agent = _init_agent(args)
    model = CustomSAC.load(args.model_path)
    obs = agent.reset()
    for step in range(args.time_steps):
        if step % 100 == 0: print("step: ", step)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = agent.step(action)
