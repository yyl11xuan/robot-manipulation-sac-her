import gymnasium as gym
import panda_gym


def make_env(env_id, seed=None, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)

    return env