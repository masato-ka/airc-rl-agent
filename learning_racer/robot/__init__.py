from .jetbot.jetbot_env import JetbotEnv
from .jetracer.jetracer_env import JetRacerEnv
# Register gym environments
from gym.envs.registration import register

register(id='jetbot-v0', entry_point='learning_racer.robot.jetbot.jetbot_env:JetbotEnv')
register(id='jetracer-v0', entry_point='learning_racer.robot.jetracer.jetracer_env:JetRacerEnv')
