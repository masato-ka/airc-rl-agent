from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
import tensorflow as tf

from learning_racer.config.config import ConfigReader

config = ConfigReader()


class CustomSACPolicy(SACPolicy):

    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self). \
            __init__(*args, **kwargs, layers=[32, 16], act_fun=tf.nn.elu, feature_extraction="mlp")


def reward(action, e_i, done):
    if done:
        norm_throttle = (action[1] - config.agent_min_throttle()) / (
                    config.agent_max_throttle() - config.agent_min_throttle())
        return config.reward_reward_crash() - (config.reward_crash_reward_weight() * norm_throttle)
    throttle_reward = config.reward_throttle_reward_weight() * (action[1] / config.agent_max_throttle())
    return 1 + throttle_reward
