# 基于 SAC + HER 的 Panda 机械臂目标到达任务：复现与多 Seed 实验

> 第一个完整的机器人学习算法工程实践。  
> 状态：**P01 训练与主要实验分析已完成；正进行文档/图表终验和 GitHub 发布。**
> **结项范围已确认：不再新增 HER `final` 消融或自创算法实验；这些属于可选后续研究，不影响 P01 结项。**

## 一、项目简介

本项目基于 **panda-gym / PyBullet** 机器人仿真环境和 **Stable-Baselines3（SB3）**，在 `PandaReach-v3` 目标到达任务中验证 SAC（Soft Actor-Critic）以及 HER（Hindsight Experience Replay，事后经验回放）的作用。

路线遵循 **环境接口理解 → SAC 基线 → 稀疏奖励对照 → HER 源码与目标重标记 → 多 Seed 评估 → 数据可视化与结论**，目标是独立掌握机器人强化学习实验工程流程。

**项目边界**：本项目研究的是**机械臂末端目标到达（Reach）**，不是抓取、复杂精密操作、真实机器人部署或新算法发明。HER 属于经典经验重标记方法；本项目的工作是实现接入、理解机制和验证样本效率。

详细学习过程、SAC 更新公式、HER `future/final`、关键代码、常见错误及复盘题：**[知识点.md](知识点.md)**。

GitHub 提交/本地项目清理的操作核验表：**[GitHub_交付检查清单.md](GitHub_交付检查清单.md)**。

## 二、项目目标

- [x] 复现 SAC 在 PandaReach 机器人目标到达任务中的训练流程。
- [x] 理解状态、动作、目标条件、奖励函数及 Gymnasium 交互接口。
- [x] 理解 SAC 的 Actor-Critic、Off-policy、Replay Buffer 和关键超参数。
- [x] 复现 SAC + HER，解释 `future` 目标采样与奖励重计算。
- [x] 建立配置、训练、定期评估、独立测试、数据分析的完整流程。
- [x] 通过 3 个 Seed 的受控实验分析 HER 的样本效率。
- [ ] 扰动鲁棒性、目标分布外泛化与复杂操作任务（**不在当前已完成范围**）。

## 三、技术栈

| 技术 / 工具 | 作用 |
|---|---|
| Python 3.10 / Conda | 独立开发环境 |
| PyTorch | SB3 深度学习计算后端 |
| Stable-Baselines3 2.4.1 | SAC、HerReplayBuffer、Monitor、EvalCallback |
| panda-gym 3.0.7 | PandaReach 机器人目标到达任务 |
| PyBullet | 物理仿真与 GUI |
| Gymnasium 0.29.1 | `reset` / `step` 环境接口 |
| NumPy / pandas / Matplotlib | 统计与可视化 |
| Git / GitHub / VS Code | 开发与版本管理 |

**开发平台**：Windows 11；当时使用 CUDA 环境（PyTorch 2.3.1+cu121）。具体安装依赖应以提交的环境清单/锁文件为准，不要仅凭本表盲目安装。

## 四、项目实施计划与验收进度

### 阶段一：开发环境与仿真环境

- [x] 配置独立的 Python / Conda 项目环境。
- [x] 安装并验证项目依赖、CUDA / PyBullet 可用性。
- [x] 了解 panda-gym 环境封装与 `PandaReach-v3`。
- [x] 运行 PandaReach 仿真与 GUI 可视化。
- [x] 理解 Observation、Action、`achieved_goal`、`desired_goal` 和 Dense/Sparse Reward。
- [x] 使用 `src/robot_rl/envs.py` 抽离环境创建接口。

### 阶段二：SAC 基线复现

- [x] 理解 SAC 的核心工作流：Actor、Critic、Replay Buffer、最大熵、Off-policy。
- [ ] 完整阅读并复述 SB3 SAC 各项网络损失及更新源码（非当前必需验收）。
- [x] 构建配置文件与机械臂目标到达任务训练脚本 `train.py`。
- [x] 完成 Dense SAC / Sparse SAC 的训练、模型保存与加载。
- [x] 编写独立策略测试脚本 `test.py`。
- [x] 记录并可视化训练奖励、评估奖励、评估成功率。

### 阶段三：SAC + HER 算法复现

- [x] 理解稀疏奖励任务的探索与训练信号困难。
- [x] 阅读 `HerReplayBuffer` 的 `sample` / `_sample_goals` / `_get_virtual_samples` 关键逻辑。
- [x] 完成 SAC + HER 训练及 50 局独立模型测试。
- [x] 自行编写简单目标重标记 Demo、单 Episode `future` 目标采样 Demo。
- [x] 对比 Sparse SAC 与 SAC + HER 的学习速度和评估成功率。

### 阶段四：对照实验与分析

- [x] 明确受控假设：HER 能提高本任务稀疏奖励条件下的前期样本效率。
- [x] 设计统一的训练预算、评估次数和 SAC 超参数。
- [x] 完成 Seed 42 / 43 / 44 × 普通 Sparse SAC / SAC+HER，共 **6 次训练**。
- [x] 从 `evaluations.npz` 读取成功率，计算跨 Seed 均值和标准差。
- [x] 绘制多 Seed 的平均成功率对照曲线（已完成初版，**双算法标准差阴影需最终检查**）。
- [x] 区分训练稳定性、随机种子敏感性与扰动鲁棒性的含义。
- [ ] **可选扩展，未开展**：HER `future` vs `final` 消融或原创采样改进（**不作为 P01 结项条件**）。
- [ ] **可选扩展，未开展**：系统验证扰动鲁棒性、目标分布外泛化及失败场景（**不作为 P01 结项条件**）。

### 阶段五：项目整理与成果交付

- [x] 完成训练、测试、HER Demo 和实验对比分析核心脚本。
- [x] 完成项目技术总结（见 [知识点.md](知识点.md)）。
- [x] 保存六组实验的训练输出和 NPZ 统计数据（本地已生成）。
- [ ] 检查 README 中运行命令与仓库当前代码完全一致。
- [ ] 将 `train.py` 的配置选择方式整理为不会误切换实验的形式，并保存完整依赖/配置快照。
- [ ] 将均值 ± 标准差两组阴影、正确时间轴和最终图片在本地验收。
- [ ] 可选：录制机械臂目标到达仿真演示（不作为结项门槛）。
- [ ] 将最终文档、代码、适合托管的实验数据与模型提交并推送 GitHub。
- [ ] 在新目录重新 clone，检查模型可加载、数据分析可复现，完成发布验收。

## 五、实验设计与主要结果

### 5.1 对照设置

- **环境**：`PandaReach-v3`（Sparse Reward）。
- **训练算法**：相同 SAC；实验变量为是否启用 `HerReplayBuffer`。
- **Seeds**：42、43、44。
- **训练预算**：每组 50,000 环境步。
- **定期评估**：每 5,000 步，20 个 Episode，使用确定性策略。
- **HER 配置**：`n_sampled_goal: 4`，`goal_selection_strategy: future`。

### 5.2 10k 步的平均成功率

| 方法 | 三 Seed 平均成功率 | 跨 Seed 标准差 |
|---|---:|---:|
| 普通 Sparse SAC | 25.00% | 8.16 个百分点 |
| SAC + HER | **96.67%** | 4.71 个百分点 |

### 5.3 首次在评估点达到 80% 成功率所需步数

| Seed | Sparse SAC | SAC + HER |
|---|---:|---:|
| 42 | 30k | 10k |
| 43 | 20k | 10k |
| 44 | 15k | 10k |
| **平均** | **21.67k** | **10k** |

本实验显示：在所采用的阈值与评估频率下，HER 平均所需交互步数减少约 **53.8%**。50k 步时两者在三个 Seed 上的评估成功率均为 **100%**。

### 5.4 正确的结论与限制

结果支持 **HER 在本 PandaReach 稀疏奖励任务中提高早期样本效率**，并不证明 HER 具有更高最终性能上限、对其他任务必然有效，或具备更强的物理扰动鲁棒性。

- 训练 Seed 只有 3 个；每次评估只有 20 局。
- 评估每 5k 步才发生一次；阈值到达时间是离散测量。
- 均值 ± 标准差阴影反映跨 Seed 离散程度，不是置信区间。
- 目前没有开展系统的目标分布外测试、碰撞扰动实验或其他 HER 超参消融。

原始明细、NumPy Shape、目标重标记 Demo、错误分析参见 [知识点.md](知识点.md)。

## 六、核心文件和复现入口

```text
train.py                        # 按 YAML 训练普通 SAC / SAC + HER
test.py                         # 独立测试 best / final 模型
src/robot_rl/envs.py            # make_env()
configs/sac_sparse.yaml         # Sparse SAC 配置
configs/sac_her.yaml            # SAC + HER 配置
scripts/plot_training.py        # 单实验训练 / 评估图
scripts/her_relabel_demo.py     # HER 目标重标记小实验
scripts/compare_experiments.py  # 多 Seed 评估成功率对照
runs/.../evaluations.npz        # 原始评估数组
runs/.../train.monitor.csv      # 原始训练日志
```

> 实际 GitHub 目录需由最终提交核对；上面列的是本项目对话中确实开发或运行过的核心路径。

### 基本环境准备（Windows PowerShell）

在项目根目录和已安装的 Python/Conda 环境中：

```powershell
python -m pip install -e .
```

同时根据依赖清单安装与本次环境兼容的 panda-gym、SB3、Gymnasium、PyBullet 等。**单独 `pip install -e .` 是否安装全部第三方依赖取决于项目 `pyproject.toml`，发布前应验证。**

### 训练 / 测试 / 分析

```powershell
# 首先确认 train.py 读取的 YAML 是目标实验配置，检查 seed 和 HER 选项
python train.py

# 先确认 test.py 读取的 YAML、model_path 与测试 seed 指向预期模型
python test.py

# 读取 runs/ 下 6 组 evaluations.npz 并绘制对照图
python scripts/compare_experiments.py
```

**重要**：截至当前代码交流，`train.py` 曾通过修改 `with open("configs/sac_sparse.yaml", ...)` 来选择配置，`test.py` 也曾手动切换 best/final 模型路径；这些命令不是一键自动切换所有实验的保证。建议提交前补充配置选择和目录核对，避免重新训练时覆盖已有实验文件。

### 保存/展示结果的建议

建议提交六组 `evaluations.npz`、必要的 CSV、对照 PNG，以及便于复现的 YAML。训练好的模型 ZIP 可视文件大小与 GitHub 仓库策略决定是否托管在 Git / Git LFS / Releases。

### 关于 HER `future` 与 `final` 的取舍

本项目**仅实测了 `future`**，并通过简化 Demo 理解其采样逻辑；未训练/比较 `final`。`future` 可在当前 Transition 到本 Episode 末尾的实际到达状态中采样新目标；`final` 仅取本 Episode 的末尾实际到达目标。哪种策略更好依赖具体任务，**不应把机制上的猜测冒充消融结果**。详见《知识点.md》中的比较小节。

## 七、参考开源项目及论文

- **panda-gym：** https://github.com/qgallouedec/panda-gym
- **Stable-Baselines3：** https://github.com/DLR-RM/stable-baselines3
- **RL Baselines3 Zoo：** https://github.com/DLR-RM/rl-baselines3-zoo
- **SAC 论文：** https://arxiv.org/abs/1801.01290
- **HER 论文：** https://arxiv.org/abs/1707.01495
- **SB3 v2.4.1 HER 文档：** https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/her.html

本项目为个人学习、经典算法复现与对照实验；第三方代码、论文及研究成果归属于其原作者，并遵循各自许可证。

## 八、当前项目进度

- [x] 明确项目定位、学习目标和仓库方向。
- [x] 完成 Windows、Conda 环境与依赖配置。
- [x] 已创建 GitHub 项目仓库：**https://github.com/yyl11xuan/robot-manipulation-sac-her**。
- [x] PandaReach 仿真、Dense/Sparse SAC、HER 训练与测试。
- [x] HER 原理及 `future` 采样最小实验。
- [x] 六组多 Seed 对照实验与统计分析初版。
- [x] 项目知识点总结文档。
- [ ] 最终绘图、配置与仓库文件完整性验收。
- [ ] 提交/推送本次最新文档与实验结果（**本次文档整理并不等于已推送**）。
- [ ] 重新克隆并验证可以复现统计及模型推理。

---
### 实验记录说明

Seed 42 的普通 Sparse SAC 训练 Monitor CSV 仅保留了
1,584 步对应的 Episode 记录，并非完整 50,000 步日志，
因此未纳入 GitHub 正式交付。

该实验的模型权重及 evaluations.npz 已完整保留。
多随机种子成功率分析基于各实验的 evaluations.npz。

**下一步（独立于 P01）**：先完成 GitHub 核验，再利用已经训练好的 SAC+HER 策略生成示范，开展轻量级 PyTorch **Behavior Cloning（BC）** 项目。新的“自主设计→独立实现→分级提示→复盘”教学方式从 BC 开始，之后通过 P02 MuJoCo 桥梁进入 P03 LeRobot/ACT 与 P04 VLA 策略适配。
