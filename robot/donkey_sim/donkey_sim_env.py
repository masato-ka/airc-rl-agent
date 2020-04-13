import gym
import gym_donkeycar


def factory_creator(sim_path, host, port, sim_track):
    def create_simulator_agent():
        env = gym.make(sim_track,
                       exe_path=sim_path, host=host, port=port, frame_skip=1)
        return env

    return create_simulator_agent
