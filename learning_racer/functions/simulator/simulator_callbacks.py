import math

from functions.base import BaseAgentCallbacks


def reward_sim(done, speed, cte, crash_reward, crash_reward_weight, throttle_reward_weight):
    """'pos': (self.x, self.y, self.z), 'cte': self.cte,
               "speed": self.speed, "hit": self.hit"""
    if done:
        return crash_reward + crash_reward_weight * (speed / 18.0)
    throttle_reward = throttle_reward_weight * (speed / 18.0)
    return 1 + throttle_reward - math.fabs(cte / 5.0)


class SimulatorCallbacks(BaseAgentCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_training_end(self) -> None:
        self.env.close()

    def on_pre_step_callback(self, action):
        return action

    def on_post_step_callback(self, action, observe, reward, done, info, z):
        reward = reward_sim(done, info['speed'], info['cte'], self.config.reward_reward_crash(),
                            self.config.reward_crash_reward_weight(), self.config.reward_throttle_reward_weight())
        return action, observe, reward, done, info, z

    def on_pre_reset(self):
        pass

    def on_post_reset(self, observe):
        return observe
