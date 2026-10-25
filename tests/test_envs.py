import gymnasium as gym
import panda_gym
from robot_rl.envs import make_env



env = make_env("PandaReachDense-v3",67,"human")
print(env.spec.id)
env.close()