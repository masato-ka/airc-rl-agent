import inspect
from importlib import import_module

import gym
import torch

from stable_baselines3 import SAC
from learning_racer.agent import StableBaselineCallback, BaseWrappedEnv
from learning_racer.exce.LearningRacerError import OptionsValueError
from learning_racer.sac import CustomSAC
from learning_racer.vae.vae import VAE
from logging import getLogger

# gym regitration
from learning_racer.robot import JetbotEnv, JetRacerEnv
import gym_donkeycar

logger = getLogger(__name__)


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


def load_pure_env(driver_name: str, driver_conf: dict):
    if driver_name is None:
        logger.error("Specify robot name can not find. Please specify correct robot name using -robot option.")
        raise OptionsValueError(
            "Specify robot name can not find. Please specify correct robot name using -robot option.")
    driver_name = driver_name
    env = gym.make(driver_name, conf=driver_conf)
    return env


def load_class(module_str, class_type):
    module_attr, class_attr = module_str.split(':')
    try:
        module = import_module(module_attr)
    except ImportError:
        raise OptionsValueError(
            "Specify module can not find. Please specify correct module in config.yml.")
    class_module = getattr(module, class_attr)
    if inspect.isclass(class_module) and issubclass(class_module, class_type):
        return class_module
    else:
        raise TypeError('Wrapped environment class must be a subclass of {}'.format(class_type))


def load_wrapped_env(env_name, parts, env, vae, config, train=True):
    wrapped_env_class = load_class(env_name, BaseWrappedEnv)
    args = {}
    for k in parts.keys():
        parts_class = load_class(parts[k], object)
        args[k] = parts_class()
    args['vae'] = vae
    args['config'] = config
    args['env'] = env
    wrapped_env = wrapped_env_class(**args)
    if not train:
        wrapped_env.eval()
    return wrapped_env


def command_train(args, config):
    torch_device = args.device
    vae = load_vae(args.vae_path, config.sac_variants_size(), config.sac_image_channel(), torch_device)
    env = load_pure_env(config.get_env_conf_robot_name(args.robot_driver),
                        config.get_env_conf_conf(args.robot_driver))
    wrapped_env = load_wrapped_env(config.get_env_conf_wrapped_env(args.robot_driver),
                                   config.get_env_conf_parts(args.robot_driver), env, vae, config, train=True)
    callback = StableBaselineCallback(wrapped_env)
    model = CustomSAC(callback.wrapped_env, args, config)
    model.lean(callback=callback)
    model.save(args.save)


def command_demo(args, config):
    torch_device = args.device
    vae = load_vae(args.vae_path, config.sac_variants_size(), config.sac_image_channel(), torch_device)
    env = load_pure_env(config.get_env_conf_robot_name(args.robot_driver),
                        config.get_env_conf_conf(args.robot_driver))
    wrapped_env = load_wrapped_env(config.get_env_conf_wrapped_env(args.robot_driver),
                                   config.get_env_conf_parts(args.robot_driver), env, vae, config, train=False)
    model = SAC.load(args.model_path)
    obs = wrapped_env.reset()
    for step in range(args.time_steps):
        if step % 100 == 0: print("step: ", step)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = wrapped_env.step(action)
