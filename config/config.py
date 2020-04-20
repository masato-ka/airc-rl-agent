from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

class ConfigReader:

    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton == None:
            cls.__singleton = super(ConfigReader, cls).__new__(cls)
        return cls.__singleton

    def __init__(self):
        self.sac = None
        self.reward = None
        self.agent = None
        self.config = None


    def load(self,file_path='config.yml'):

        with open(file_path, 'r') as f:
            self.config = load(f, Loader=Loader)
        self.sac = self.config.get('SAC_SETTING')
        self.reward = self.config.get('REWARD_SETTING')
        self.agent = self.config.get('AGENT_SETTING')

    def sac_log_interval(self):
        return self.sac.get('LOG_INTERVAL')

    def sac_verbose(self):
        return self.sac.get('VERBOSE')

    def sac_learning_rate(self):
        return float(self.sac.get('LEARNING_RATE'))

    def sac_ent_coef(self):
        return self.sac.get('ENT_COEF')

    def sac_train_freq(self):
        return self.sac.get('TRAIN_FREQ')

    def sac_batch_size(self):
        return self.sac.get('BATCH_SIZE')

    def sac_gradient_steps(self):
        return self.sac.get('GRADIENT_STEPS')

    def sac_learning_starts(self):
        return self.sac.get('LEARNING_STARTS')

    def sac_buffer_size(self):
        return self.sac.get('BUFFER_SIZE')

    def sac_variants_size(self):
        return self.sac.get('VARIANTS_SIZE')

    def sac_image_channel(self):
        return self.sac.get('IMAGE_CHANNELS')

    def reward_reward_crash(self):
        return self.reward.get('REWARD_CRASH')

    def reward_crash_reward_weight(self):
        return self.reward.get('CRASH_REWARD_WEIGHT')

    def reward_throttle_reward_weight(self):
        return self.reward.get('THROTTLE_REWARD_WEIGHT')

    def agent_n_command_history(self):
        return self.agent.get('N_COMMAND_HISTORY')

    def agent_min_steering(self):
        return self.agent.get('MIN_STEERING')

    def agent_max_steering(self):
        return self.agent.get('MAX_STEERING')

    def agent_min_throttle(self):
        return self.agent.get('MIN_THROTTLE')

    def agent_max_throttle(self):
        return self.agent.get('MAX_THROTTLE')

    def agent_max_steering_diff(self):
        return self.agent.get('MAX_STEERING_DIFF')

ConfigReader()

if __name__ == '__main__':
    config = ConfigReader()
    config.load('../config.yml')


    print(config.sac_batch_size())

