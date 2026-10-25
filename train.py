import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from robot_rl.envs import make_env

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAC on PandaReach"
    )

    parser.add_argument(
        "--env-id",
        type=str,
        default="PandaReachDense-v3",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
    )

    return parser.parse_args()

def main():
    #以此函数可以做到在命令行修改环境参数
    args = parse_args()
    #设置环境参数
    env = make_env(
        env_id=args.env_id,
        seed=args.seed,
        render_mode="rgb_array",)
    env = Monitor(env)
    try:
        #设置SAC算法参数
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        print(f"模型使用的设备是{model.device}")
    finally:
        env.close()


if __name__ == "__main__":
    main()