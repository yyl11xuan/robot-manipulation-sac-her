import gymnasium as gym
import panda_gym
import numpy as np

# 创建 PandaReach 密集奖励环境
env = gym.make("PandaReachDense-v3")

# 重置环境
obs, info = env.reset()

# 观察空间和动作空间
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

# 观察机器人状态
print("Observation:", obs["observation"])
print("Achieved Goal:", obs["achieved_goal"])
print("Desired Goal:", obs["desired_goal"])


#随机生成动作
action_space = env.action_space
action = action_space.sample()
print(action)
#计算动作执行前的goal距离
current_location = obs["achieved_goal"]
goal_location = obs["desired_goal"]
distance_before = np.linalg.norm(goal_location-current_location,ord=2)
print("现在距离是:",distance_before)

#执行动作

next_obs,reward,terminated,truncated,_ = env.step(action)

#执行动作后目标距离
current_location = next_obs["achieved_goal"]
goal_location = next_obs["desired_goal"]
distance_after = np.linalg.norm(goal_location-current_location,ord=2)
print(f"行动后现在距离是:{distance_after},奖励是:{reward},是否到达终点：{terminated},是否因时间限制等原因被截断：{truncated}")




env.close()