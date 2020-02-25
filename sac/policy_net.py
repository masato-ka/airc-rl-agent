from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
import tensorflow as tf


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).\
        __init__(*args, **kwargs,layers=[32, 16],act_fun=tf.nn.elu,feature_extraction="mlp")
