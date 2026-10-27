from pathlib import Path
import yaml
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from robot_rl.envs import make_env


def main():
    # 1. 读取 YAML 配置
    with open("configs/sac_sparse.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_id = config["env"]["id"]
    render_mode = config["env"]["render_mode"]

    seed = config["train"]["seed"]
    device = config["train"]["device"]
    total_timesteps = config["train"]["total_timesteps"]

    eval_freq = config["evaluation"]["eval_freq"]
    n_eval_episodes = config["evaluation"]["n_eval_episodes"]

    # 2. 创建实验目录
    her_suffix = "_HER" if "her" in config else ""

    run_dir = Path("runs") / (
        f"{env_id}{her_suffix}_seed{seed}_steps{total_timesteps}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3. 创建训练环境
    train_env = make_env(
        env_id=env_id,
        seed=seed,
        render_mode=render_mode,
    )

    train_env = Monitor(
        train_env,
        filename=str(run_dir / "train.monitor.csv"),
        info_keywords=("is_success",),
    )

    # 4. 创建独立的训练期间评估环境
    eval_env = make_env(
        env_id=env_id,
        seed=seed + 1000,
        render_mode=render_mode,
    )

    # 评估数据由 EvalCallback 保存，无需额外 CSV
    eval_env = Monitor(eval_env)

    try:
        sac_extra = {}

        if "her" in config:
            sac_extra = {
                "replay_buffer_class": HerReplayBuffer,
                "replay_buffer_kwargs": config["her"],
            }
        # 5. 初始化 SAC
        model = SAC(
            policy="MultiInputPolicy",
            env=train_env,
            seed=seed,
            device=device,
            verbose=1,
            **config["sac"],#引用配置文件，标号为sac
            **sac_extra,
        )

        print(f"Replay Buffer 类型：{type(model.replay_buffer).__name__}")
        # 6. 设置定期评估
        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,#这个是评估频率
            n_eval_episodes=n_eval_episodes,#这个是每次评估几个episode
            deterministic=True,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),#evaluations.npz生成
            verbose=1,
        )

        # 7. 只调用一次 learn
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=1,
        )

        # 8. 保存最后一步模型
        save_path = run_dir / "sac_model"
        model.save(save_path)

        print(f"实际训练步数：{model.num_timesteps}")
        print(f"最终模型已保存：{save_path}.zip")

    finally:
        # 9. 关闭训练环境和评估环境
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
