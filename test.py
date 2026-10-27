from pathlib import Path
import numpy as np
import yaml

from stable_baselines3 import SAC
from robot_rl.envs import make_env


def main():
    with open("configs/sac_sparse.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_id = config["env"]["id"]
    seed = config["train"]["seed"]
    device = config["train"]["device"]
    total_timesteps = config["train"]["total_timesteps"]

    her_suffix = "_HER" if "her" in config else ""

    run_dir = Path("runs") / (
        f"{env_id}{her_suffix}_seed{seed}_steps{total_timesteps}"
    )


    eval_env = make_env(
        env_id=env_id,
        seed=seed + 2000,
        render_mode="rgb_array",
    )

    model_path = run_dir / "best_model.zip"  
#test.py 不需要额外导入 HerReplayBuffer。 它已经在训练时配置进模型，我们现在只需要加载保存的 SAC 模型。
    model = SAC.load(
        model_path,
        env=eval_env,
        device=device,
    )

    print("模型加载成功！")

    n_eval_episodes = 50
    episode_rewards = []
    success_results = []
   

    i=1
    for i in range(n_eval_episodes):
        obs,_= eval_env.reset()
        episode_reward =0
        done = False

        while done==False:
            action,_ = model.predict(obs,deterministic=True)
            next_obs,reward,terminated,truncated,info = eval_env.step(action)

            done = truncated or terminated

            episode_reward+=reward
            obs = next_obs

        episode_rewards.append(episode_reward)
        success_results.append(bool(info["is_success"]))

        print(f"epidose {i}:   Reward = {episode_reward}   Success = {success_results[i]}")

    print("===== Evaluation Results =====")
    mean_reward = np.mean(episode_rewards)
    success_rate = np.mean(success_results)
    print(f"Mean Reward:  {mean_reward}")
    print(f"Success Rate:  {success_rate}")


    eval_env.close()

if __name__ == "__main__":
    main()

        
        