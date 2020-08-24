import math

from learning_racer.config.config import ConfigReader

config = ConfigReader()


def reward(action, e_i, done):
    if done:
        norm_throttle = (action[1] - config.agent_min_throttle()) / (
                config.agent_max_throttle() - config.agent_min_throttle())
        return config.reward_reward_crash() - (config.reward_crash_reward_weight() * norm_throttle), done
    throttle_reward = config.reward_throttle_reward_weight() * (action[1] / config.agent_max_throttle())
    return 1 + throttle_reward, done


def reward_sim(action, e_i, done):
    """'pos': (self.x, self.y, self.z), 'cte': self.cte,
                   "speed": self.speed, "hit": self.hit"""
    speed = e_i['speed']
    speed = min(18.0, speed)
    if done:
        # norm_throttle = (action[1] - config.agent_min_throttle()) / (
        #         config.agent_max_throttle() - config.agent_min_throttle())
        # if (e_i['cte'] * action[0]) > 0.0:
        #    norm_throttle + 0.3
        return config.reward_reward_crash(), done  # - (config.reward_crash_reward_weight() * norm_throttle)
    # throttle_reward = config.reward_throttle_reward_weight() * (action[1] / config.agent_max_throttle())
    throttle_reward = config.reward_throttle_reward_weight() * (speed / 18.0) - math.fabs(e_i['cte'] / 15.0)
    cte_reward = 0
    # cte_reward = -0.1 * (0.4 - ((math.exp(-(math.fabs(e_i['cte'] + 0.0)) ** 2 / 2)) / math.sqrt(2 * math.pi)))
    # if math.fabs(e_i['cte'] + 0.2) < 0.3:
    #     pass
    # else:
    #     cte_reward = -0.1
    return 1 + throttle_reward + cte_reward, done
