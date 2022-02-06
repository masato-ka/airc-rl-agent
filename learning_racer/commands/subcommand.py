import gym
import torch

from stable_baselines3 import SAC

from learning_racer.agent.simulator.simulator_auto_stop_env import SimulatorAutoStopEnv
from learning_racer.agent import StableBaselineCallback, TeleoperationEnv, SimulatorEnv, AutoStopEnv
from learning_racer.exce.LearningRacerError import OptionsValueError
from learning_racer.sac import CustomSAC
from learning_racer.teleoperate import Teleoperator
from learning_racer.vae.vae import VAE
from logging import getLogger

# gym regitration
from learning_racer.robot import JetbotEnv, JetRacerEnv
import gym_donkeycar

logger = getLogger(__name__)

env_config = {
    'jetbot': {
        'robot_name': 'jetbot-v0',
        'wrapped_env': 'learning_racer.agent.teleoperation_env:TeleoperationEnv',
        'conf': {},
        'parts': {'teleoperator': 'learning_racer.teleoperate:Teleoperator'},
    },
    'jetracer': {
        'robot_name': 'jetracer-v0',
        'wrapped_env': 'learning_racer.agent.teleoperation_env:TeleoperationEnv',
        'conf': {},
        'parts': {'teleoperator': 'learning_racer.teleoperate:Teleoperator'},
    },
    'jetbot-auto': {
        'robot_name': 'jetbot-v0',
        'wrapped_env': 'learning_racer.agent.auto_stop_env:AutoStopEnv',
        'conf': {},
        'parts': {},
    },
    'jetracer-auto': {
        'robot_name': 'jetracer-v0',
        'wrapped_env': 'learning_racer.agent.auto_stop_env:AutoStopEnv',
        'conf': {},
        'parts': {},
    },
    'sim': {
        'robot_name': 'donkey-generated-track-v0',
        'wrapped_env': 'learning_racer.agent.simulator_env:SimulatorEnv',
        'conf': {"exe_path": "remote", "port": 9091, "host": "127.0.0.1", "frame_skip": 1},
        'parts': {},
    },
    'sim-auto': {
        'robot_name': 'donkey-generated-track-v0',
        'wrapped_env': 'learning_racer.agent.simulator_env:SimulatorAutoStopEnv',
        'conf': {"exe_path": "remote", "port": 9091, "host": "127.0.0.1", "frame_skip": 1},
        'parts': {},
    }

}


def load_vae(model_path, variants_size, image_channels, device):
    vae = VAE(image_channels=image_channels, z_dim=variants_size)
    try:
        vae.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except FileNotFoundError:
        logger.error("Specify VAE model path can not find. Please specify correct vae path using -vae option.")
        raise OptionsValueError(
            "Specify VAE model path can not find. Please specify correct vae path using -vae option.")
    vae.to(torch.device(device)).eval()
    return vae


def load_pure_env(robot_driver: str):
    if robot_driver not in env_config:
        logger.error("Specify robot name can not find. Please specify correct robot name using -robot option.")
        raise OptionsValueError(
            "Specify robot name can not find. Please specify correct robot name using -robot option.")
    driver_name = env_config[robot_driver]['robot_name']
    env = gym.make(driver_name, conf=env_config[robot_driver]['conf'])
    return env


def load_wrapped_env(env_name, env, vae, config, train=True):
    wrapped_env = None

    if env_name == 'learning_racer.agent.teleoperation_env:TeleoperationEnv':
        teleoperator = Teleoperator()
        teleoperator.start_process()
        wrapped_env = TeleoperationEnv(teleoperator, env, vae, config)
    elif env_name == "learning_racer.agent.auto_stop_env:AutoStopEnv":
        teleoperator = Teleoperator()
        teleoperator.start_process()
        wrapped_env = AutoStopEnv(teleoperator, env, vae, config)
    elif env_name == 'learning_racer.agent.simulator_env:SimulatorEnv':
        wrapped_env = SimulatorEnv(env, vae, config)
    elif env_name == 'learning_racer.agent.simulator_env:SimulatorAutoStopEnv':
        wrapped_env = SimulatorAutoStopEnv(env, vae, config)
    else:
        logger.error("Specify env name can not find. Please specify correct env name in config.yaml file.")
        raise OptionsValueError(
            "Specify env name can not find. Please specify correct env name in config.yaml file.")
    if not train:
        wrapped_env.eval()
    return wrapped_env


def command_train(args, config):
    torch_device = args.device
    vae = load_vae(args.vae_path, config.sac_variants_size(), config.sac_image_channel(), torch_device)
    env = load_pure_env(args.robot_driver)
    wrapped_env = load_wrapped_env(env_config[args.robot_driver]['wrapped_env'], env, vae, config, train=True)
    callback = StableBaselineCallback(wrapped_env)
    model = CustomSAC(callback.wrapped_env, args, config)
    model.lean(callback=callback)
    model.save(args.save)


def command_demo(args, config):
    torch_device = args.device
    vae = load_vae(args.vae_path, config.sac_variants_size(), config.sac_image_channel(), torch_device)
    env = load_pure_env(args.robot_driver)
    wrapped_env = load_wrapped_env(env_config[args.robot_driver]['wrapped_env'], env, vae, config, train=False)
    model = SAC.load(args.model_path)
    obs = wrapped_env.reset()
    for step in range(args.time_steps):
        if step % 100 == 0: print("step: ", step)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = wrapped_env.step(action)
