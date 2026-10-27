from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# file_path = (
#     Path("runs")
#     / "PandaReach-v3_seed42_steps50000"
#     / "evaluations.npz"
# )

# data = np.load(file_path)

# print("包含的字段：", data.files)
# '''
# timesteps	(10,)	10 个评估时间点
# results	(10, 20)	每次评估的 20 局累计奖励
# ep_lengths	(10, 20)	每次评估的 20 局执行步数
# successes	(10, 20)	每次评估的 20 局成功情况
# '''

def load_success_rates(seeds, use_her=False):
    all_success_rates = []

    her_suffix = "_HER" if use_her else ""

    for seed in seeds:
        file_path = (
            Path("runs")
            / f"PandaReach-v3{her_suffix}_seed{seed}_steps50000"
            / "evaluations.npz"
        )

        with np.load(file_path) as data:
            actual_steps = data["timesteps"]
            if not np.array_equal(actual_steps, timesteps):
                raise ValueError(f"评估时间轴不一致：{file_path}")
            rates = data["successes"].mean(axis=1) * 100

        all_success_rates.append(rates)

    all_success_rates = np.array(all_success_rates)

    return all_success_rates


timesteps = np.arange(5000, 50001, 5000)
path = Path("result/")
path.mkdir(parents=True, exist_ok=True)
sac_rates = load_success_rates([42, 43, 44], use_her=False)
her_rates = load_success_rates([42, 43, 44], use_her=True)

sac_mean_rates = sac_rates.mean(axis=0)
sac_std_rates = sac_rates.std(axis=0)
her_mean_rates = her_rates.mean(axis=0)
her_std_rates = her_rates.std(axis=0)

print("SAC 平均成功率：", sac_mean_rates)
print("SAC 标准差：", sac_std_rates)

print("HER 平均成功率：", her_mean_rates)
print("HER 标准差：", her_std_rates)

timesteps = np.arange(5000, 50001, 5000)


plt.figure()

plt.plot(timesteps, sac_mean_rates, label="Sparse SAC")
plt.plot(timesteps, her_mean_rates, label="SAC + HER")

plt.fill_between(
    timesteps,
    np.clip(sac_mean_rates - sac_std_rates, 0, 100),
    np.clip(sac_mean_rates + sac_std_rates, 0, 100),
    alpha=0.2,
)#阴影的垂直宽度： 不同 Seed 的结果有多分散？越宽，代表同一步数下的跨 Seed 差异越大
plt.fill_between(
    timesteps,
    np.clip(her_mean_rates - her_std_rates, 0, 100),
    np.clip(her_mean_rates + her_std_rates, 0, 100),
    alpha=0.2,
)#阴影的垂直宽度： 不同 Seed 的结果有多分散？越宽，代表同一步数下的跨 Seed 差异越大
plt.plot(timesteps, her_mean_rates, color="tab:orange", label="SAC + HER")
plt.xlabel("Training Timesteps")
plt.ylabel("Success Rate (%)")
plt.title("Sparse SAC vs SAC + HER (3 Seeds)")
plt.legend()
plt.grid(True, alpha=0.3)


plt.savefig(
    path / "success_rate_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()
plt.close()