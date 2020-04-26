import torch

from learning_racer import exce
from learning_racer.agent.agent import Agent
from learning_racer.exce.LearningRacerError import OptionsValueError
from learning_racer.robot import JetbotEnv, JetRacerEnv
from learning_racer.sac import CustomSAC, CustomSACPolicy, reward
from learning_racer.teleoperate import Teleoperator
from learning_racer.vae.vae import VAE
from learning_racer.robot.donkey_sim.donkey_sim_env import factory_creator
from logging import getLogger

logger = getLogger(__name__)

robot_drivers = {'jetbot': JetbotEnv, 'jetracer': JetRacerEnv, 'sim': factory_creator}

def _load_vae(model_path, variants_size, image_channels, device):
    vae = VAE(image_channels=image_channels, z_dim=variants_size)
    try:
        vae.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except FileNotFoundError:
        logger.error("Specify VAE model path can not find. Please specify correct vae path using -vae option.")
        raise OptionsValueError(
            "Specify VAE model path can not find. Please specify correct vae path using -vae option.")
    vae.to(torch.device(device)).eval()
    return vae


def _create_agent(robot_driver, vae, teleop, torch_device, config, train):
    env = robot_driver()
    agent = Agent(env, vae, teleop=teleop, device=torch_device, reward_callback=reward, config=config, train=train)
    return agent


def _init_agent(args, config, train=True):
    torch_device = args.device
    vae = _load_vae(args.vae_path, config.sac_variants_size(), config.sac_image_channel(), torch_device)
    print(args.robot_driver)
    if args.robot_driver in ['jetbot', 'jetracer']:
        teleop = Teleoperator()
        agent = _create_agent(robot_drivers[args.robot_driver], vae, teleop, torch_device, config, train=train)
    elif args.robot_driver == 'sim':
        agent = _create_agent(robot_drivers[args.robot_driver]
                              (args.sim_path, args.sim_host, args.sim_port, args.sim_track),
                              vae, None, torch_device, config, train=train)
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

def command_train(args, config):
    agent = _init_agent(args, config)
    if args.load_model == '':
        model = CustomSAC(CustomSACPolicy, agent, verbose=config.sac_verbose(), batch_size=config.sac_batch_size(),
                          buffer_size=config.sac_buffer_size(),
                          learning_starts=config.sac_learning_starts(), gradient_steps=config.sac_gradient_steps(),
                          train_freq=config.sac_train_freq(),
                          ent_coef=config.sac_ent_coef(), learning_rate=config.sac_learning_rate())
    else:
        model = CustomSAC.load(args.load_model, env=agent, verbose=config.sac_verbose(),
                               batch_size=config.sac_batch_size(),
                               buffer_size=config.sac_buffer_size(),
                               learning_starts=config.sac_learning_starts(), gradient_steps=config.sac_gradient_steps(),
                               train_freq=config.sac_train_freq(),
                               ent_coef=config.sac_ent_coef(), learning_rate=config.sac_learning_rate())
    save_callback = _generate_save_callbask(args)

    model.learn(total_timesteps=args.time_steps, log_interval=config.sac_log_interval(), callback=save_callback)

    model.save(args.save)


def command_demo(args, config):
    agent = _init_agent(args, config, train=False)
    agent._wrapped_env.viewer.set_car_config(args.sim_car, (128, 128, 128), args.sim_user, 20)
    model = CustomSAC.load(args.model_path)
    obs = agent.reset()
    for step in range(args.time_steps):
        if step % 100 == 0: print("step: ", step)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = agent.step(action)
