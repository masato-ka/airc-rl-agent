import gym
import gym_donkeycar


def create_simulator_agent(donkey_sim_path="remote", host="127.0.0,1", port=9091):
    env = gym.make('donkey-generated-track-v0', exe_path=donkey_sim_path, host=host, port=port)
    return env
