import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


def _load_sac(agent, args, config, policy):
    model = None
    if args.load_model == '':
        model = SAC("MlpPolicy", policy_kwargs=policy,
                    env=agent, verbose=config.sac_verbose(), batch_size=config.sac_batch_size(),
                    buffer_size=config.sac_buffer_size(),
                    learning_starts=config.sac_learning_starts(), gradient_steps=config.sac_gradient_steps(),
                    train_freq=config.sac_train_freq(),
                    ent_coef=config.sac_ent_coef(), learning_rate=config.sac_learning_rate(),
                    tensorboard_log="tblog", gamma=0.99, tau=0.02, use_sde_at_warmup=True, use_sde=True,
                    sde_sample_freq=64, n_episodes_rollout=1
                    )
    else:
        model = SAC.load(args.load_model, env=agent,
                         policy_kwargs=policy,
                         verbose=config.sac_verbose(),
                         batch_size=config.sac_batch_size(),
                         buffer_size=config.sac_buffer_size(),
                         learning_starts=config.sac_learning_starts(), gradient_steps=config.sac_gradient_steps(),
                         train_freq=config.sac_train_freq(),
                         ent_coef=config.sac_ent_coef(), learning_rate=config.sac_learning_rate(),
                         tensorboard_log="tblog", gamma=0.99, tau=0.02, use_sde_at_warmup=True, use_sde=True,
                         sde_sample_freq=64, n_episodes_rollout=1)
    return model


class CustomSAC:

    def __init__(self, agent, args, config):
        self.agent = Monitor(agent)
        self.args = args
        self.config = config
        self.policy = dict(activation_fn=torch.nn.ReLU, net_arch=[32, 32], use_sde=True)
        self._load_sac(agent, args, config, self.policy)
        self.checkpoint_cb = \
            CheckpointCallback(save_freq=args.save_freq_steps,
                               save_path=args.save_model_path, name_prefix=args.save)

    def lean(self):
        self.model.learn(total_timesteps=self.args.time_steps,
                         log_interval=self.config.sac_log_interval(),
                         tb_log_name="racer_learnig_log",
                         callback=self.checkpoint_cb)

    def predict(self, obs):
        return self.model.predict(obs)
