import gym
import gym_donkeycar


def factory_creator(sim_path, host, port, sim_track):
    def create_simulator_agent():
        conf = {"exe_path": sim_path, "port": port, "host": host, "frame_skip": 1}
        env = gym.make(sim_track, conf=conf)
        return env

    return create_simulator_agent
