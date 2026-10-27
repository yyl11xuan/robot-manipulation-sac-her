import gymnasium as gym
import panda_gym
import numpy as np
import copy
import random
from robot_rl.envs import make_env


def main():
    env = gym.make("PandaReach-v3")

    obs, info = env.reset(seed=42)
    #生成随机动作
    action_space = env.action_space

    action = action_space.sample()

    transitions = []



    while True:
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        transitions.append({
            "obs": {k: v.copy() for k, v in obs.items()},
            "action": action.copy(),
            "next_obs": {k: v.copy() for k, v in next_obs.items()},
            "reward": reward,
            "info": info.copy(),
        })

        obs = next_obs

        if terminated or truncated:
            break

    print("Episode Transition 数量：", len(transitions))

    N = len(transitions)
    idx = np.random.randint(N)
    transition = transitions[idx]#transtions是列表，存放的是字典
    print(f'这里随机抽到的是{transitions[idx]}')
    print(f'原始目标是{transition["obs"]["desired_goal"]}')

    print(f'实际到达{transition["next_obs"]["achieved_goal"]}')

    print(f'获得奖励是{transition["reward"]}')

    future_idx  = np.random.randint(idx, N)

    new_goal = transitions[future_idx]["next_obs"]["achieved_goal"].copy()
    print(f'新目标是{new_goal}')
#真实的HER，的新目标和当前的transition都是随机抽样的
    new_reward = env.unwrapped.compute_reward(
        transition["next_obs"]["achieved_goal"],
        new_goal,
        transition["info"],
    )

    print(f"新奖励是{new_reward}")


    env.close()


if __name__ == "__main__":
    main()
