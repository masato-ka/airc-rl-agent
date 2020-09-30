from typing import Optional

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv


def _load_sac(agent, args, config, policy):
    model = None
    if args.load_model == '':
        model = SAC("MlpPolicy", policy_kwargs=policy,
                    env=agent, verbose=config.sac_verbose(), batch_size=config.sac_batch_size(),
                    buffer_size=config.sac_buffer_size(),
                    learning_starts=config.sac_learning_starts(), gradient_steps=config.sac_gradient_steps(),
                    train_freq=config.sac_train_freq(),
                    ent_coef=config.sac_ent_coef(), learning_rate=config.sac_learning_rate(),
                    tensorboard_log="tblog", gamma=config.sac_gamma(), tau=config.sac_tau(),
                    use_sde_at_warmup=config.sac_use_sde_at_warmup(), use_sde=config.sac_use_sde(),
                    sde_sample_freq=config.sac_sde_sample_freq(), n_episodes_rollout=1
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
                         tensorboard_log="tblog", gamma=config.sac_gamma(), tau=config.sac_tau(),
                         use_sde_at_warmup=config.sac_use_sde_at_warmup(), use_sde=config.sac_use_sde(),
                         sde_sample_freq=config.sac_sample_freq(), n_episodes_rollout=1)
    return model


class CustomSAC:

    def __init__(self, agent, args, config):
        self.agent = Monitor(agent)
        self.args = args
        self.config = config
        self.policy = dict(activation_fn=torch.nn.ReLU, net_arch=[32, 32], use_sde=True)
        self.model = _load_sac(self.agent, self.args, self.config, self.policy)
        self.checkpoint_cb = \
            CheckpointCallback(save_freq=args.save_freq_steps,
                               save_path=args.save_model_path, name_prefix=args.save)

    def lean(self,
             callback: MaybeCallback = None,
             log_interval: int = 4,
             eval_env: Optional[GymEnv] = None,
             eval_freq: int = -1,
             n_eval_episodes: int = 5,
             tb_log_name: str = "run",
             eval_log_path: Optional[str] = None,
             reset_num_timesteps: bool = True, ):
        callback = CallbackList([self.checkpoint_cb, callback])
        self.model.learn(total_timesteps=self.args.time_steps,
                         log_interval=self.config.sac_log_interval(),
                         tb_log_name="racer_learnig_log",
                         callback=callback)
        return self.model

    def predict(self, obs):
        return self.model.predict(obs)

    def save(self, path):
        self.model.save(path)
