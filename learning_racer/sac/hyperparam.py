import math

from learning_racer.config.config import ConfigReader

hit_counter = 0
speed_counter = 0
config = ConfigReader()


def reward(action, e_i, done):
    if done:
        norm_throttle = (action[1] - config.agent_min_throttle()) / (
                config.agent_max_throttle() - config.agent_min_throttle())
        return config.reward_reward_crash() - (config.reward_crash_reward_weight() * norm_throttle), done
    throttle_reward = config.reward_throttle_reward_weight() * (action[1] / config.agent_max_throttle())
    return 1 + throttle_reward, done


# For gym_donkey
def reward_sim(self, done):
    """'pos': (self.x, self.y, self.z), 'cte': self.cte,
                   "speed": self.speed, "hit": self.hit"""
    if done:
        return config.reward_reward_crash() + config.reward_crash_reward_weight() * (self.speed / 18.0)
    throttle_reward = config.reward_throttle_reward_weight() * (self.speed / 18.0)
    return 1 + throttle_reward - math.fabs(self.cte / 5.0)


# For gym_donkey
hit_counter = 0
speed_counter = 0
initial = False


def episode_over_sim(self):
    global hit_counter, speed_counter, initial
    #    print(self.speed)

    if not initial and self.speed > 3.0:
        initial = True

    if self.hit != "none":
        hit_counter += 1
        if hit_counter > 5:
            self.over = True
            hit_counter = 0
    elif self.speed < 0.03 and initial:
        speed_counter += 1
        if speed_counter > 10:
            self.over = True
            speed_counter = 0
    elif self.missed_checkpoint:
        self.over = True
    elif self.dq:
        self.over = True
