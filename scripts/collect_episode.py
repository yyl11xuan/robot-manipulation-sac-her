import gymnasium as gym
import panda_gym
import numpy as np

# 创建 PandaReach 密集奖励环境
env = gym.make("PandaReachDense-v3")

# 重置环境
obs, info = env.reset()
current_location = obs["achieved_goal"]
goal_location = obs["desired_goal"]
distance_before = np.linalg.norm(goal_location-current_location,ord=2)
count =1
episode_reward = 0
done = False

while done==False :
    action = env.action_space.sample()
    print(f"当前动作是：{action}")

    next_obs,reward,terminated,truncated,_ = env.step(action)
    obs = next_obs
    done = terminated or truncated

    current_location = next_obs["achieved_goal"]
    goal_location = next_obs["desired_goal"]
    distance_after = np.linalg.norm(goal_location-current_location,ord=2)
    print(f"现在是第{count}次行动了，行动后现在距离是:{distance_after},奖励是:{reward},是否到达终点：{terminated},是否因时间限制等原因被截断：{truncated}")
    episode_reward +=reward
    count+=1

print("\n===== Episode 结束 =====")
print(f"总步数：{count - 1}")
print(f"累计奖励：{episode_reward:.4f}")
print(f"是否成功：{terminated}")
print(f"是否超时截断：{truncated}")

env.close()


