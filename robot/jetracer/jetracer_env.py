import numpy as np
from gym import Env, spaces

from .core.controller import RobotController
from .core.observer import Observer

#Camera settings
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

#Actuator settings
MIN_STEERING = 1.0
MAX_STEERING = -1.0
MIN_THROTTLE = 0.0
MAX_THROTTLE = 1.0

class JetRacerEnv(Env):

    def __init__(self):
        super(JetRacerEnv, self).__init__()
        self.controller = RobotController()
        self.observer = Observer(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=IMAGE_SIZE,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([MIN_STEERING, MIN_THROTTLE]),
                                       high=np.array([MAX_STEERING, MAX_THROTTLE]), dtype=np.float32)
        self.ie = {}

        self.observer.start()

    def step(self, action):
        self.controller.action(action[0], action[1])
        obs = self.observer.observation()
        reward = 1.0
        done = False
        return obs, reward, done, self.ie
        pass

    def reset(self):
        self.controller.action(0,0)
        obs = self.observer.observation()
        return obs

    def render(self, mode='human'):
        pass

    def seed(self,seed):
        pass

    def close(self):
        self.observer.stop()
        pass