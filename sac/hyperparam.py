from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
import tensorflow as tf

from config import MIN_THROTTLE, MAX_THROTTLE, REWARD_CRASH, CRASH_REWARD_WEIGHT, THROTTLE_REWARD_WEIGHT


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).\
        __init__(*args, **kwargs,layers=[32, 16],act_fun=tf.nn.elu,feature_extraction="mlp")


def reward(action, e_i, done):
    if done:
        norm_throttle = (action[1] - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
        return REWARD_CRASH - (CRASH_REWARD_WEIGHT * norm_throttle)
    throttle_reward = THROTTLE_REWARD_WEIGHT * (action[1] / MAX_THROTTLE)
    return 1 + throttle_reward