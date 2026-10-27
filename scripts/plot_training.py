
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

run_dir = Path("runs/PandaReach-v3_seed44_steps50000")

# 中文字体设置（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 1. 读取训练 CSV ==========

df = pd.read_csv(
    run_dir / "train.monitor.csv",
    comment="#",
)

# 准备训练曲线的横纵坐标
steps = df["l"].cumsum()
rewards = df["r"]
smooth_rewards = rewards.rolling(20, min_periods=1).mean()


# ========== 2. 读取评估 NPZ ==========

data = np.load(run_dir / "evaluations.npz")

eval_steps = data["timesteps"]
eval_rewards = data["results"]
eval_successes = data["successes"]

mean_eval_rewards = eval_rewards.mean(axis=1)
success_rates = eval_successes.mean(axis=1) * 100


# ========== 3. 画训练奖励曲线 ==========

plt.figure()

plt.plot(steps, rewards, alpha=0.3, label="原始奖励")
plt.plot(steps, smooth_rewards, label="20 Episode 移动平均")

plt.xlabel("训练步数")
plt.ylabel("Episode 累计奖励")
plt.title("SAC 训练奖励曲线")
plt.legend()
plt.grid(True)

plt.savefig(run_dir / "training_reward.png", dpi=150)
plt.close()


# ========== 4. 画评估奖励曲线 ==========

plt.figure()

plt.plot(eval_steps, mean_eval_rewards, marker="o")

plt.xlabel("训练步数")
plt.ylabel("平均评估奖励")
plt.title("SAC 评估奖励曲线")
plt.grid(True)

plt.savefig(run_dir / "evaluation_reward.png", dpi=150)
plt.close()


# ========== 5. 画评估成功率曲线 ==========

plt.figure()

plt.plot(eval_steps, success_rates, marker="o")

plt.xlabel("训练步数")
plt.ylabel("成功率 (%)")
plt.title("SAC 评估成功率曲线")
plt.ylim(0, 100)
plt.grid(True)

plt.savefig(run_dir / "evaluation_success.png", dpi=150)
plt.close()

print("三张训练分析图已保存！")
