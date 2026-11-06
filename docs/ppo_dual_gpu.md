# 双卡 PPO 训练说明

双卡版使用两个进程，每个进程在一张 GPU 上保存完整模型。两张卡分别采样不同电路，再通过 DDP 同步梯度，共同更新同一个模型。两张卡的显存不会合并，每张卡都需要容纳模型、梯度和 Adam 优化器状态。

## 文件与启动入口

| 用途 | 单卡（原文件） | 双卡（新增文件） |
| --- | --- | --- |
| 配置 | `configs/ppo.yaml` | `configs/ppo_dual_gpu.yaml` |
| 启动入口 | `scripts/train_ppo.py` | `scripts/train_ppo_dual_gpu.py` |
| 训练循环 | `circuit_transformer/ppo_training.py` | `circuit_transformer/ppo_dual_gpu.py` |

双卡版复用原有电路采样、编码和文件工具，原单卡文件无需修改。

## 配置

编辑 `configs/ppo_dual_gpu.yaml`：

```yaml
device: cuda          # 由进程自动绑定第一、第二张可见显卡
world_size: 2
mixed_precision: true # FP16 + 动态梯度缩放，利用 V100 Tensor Core
num_circuits: null    # 完整遍历 data_dir 直接包含的全部 JSON
data_epochs: 1       # 完整遍历一次
rollout_batch_size: 128 # 两卡合计，每卡通常处理 64 个电路
sampling_batch_size: 32 # 每张卡同时生成的轨迹数
samples_per_circuit: 2 # 每个电路采样 2 次，计算同题基线
minibatch_size: 32     # 两卡合计有效训练 batch，必须为正偶数
train_microbatch_size: 16 # 每卡一次前向/反向同时处理的轨迹数
ppo_epochs: 2          # 每批采样轨迹重复更新 2 轮
save_every_batches: 10 # 每完成 10 批采样及更新保存一次；遍历结束也保存
log_every_batches: 10  # 每 10 批打印一次进度和性能
log_every_updates: 100 # 每 100 次参数更新打印一次指标
log_every_checkpoints: 10 # 每保存 10 次检查点打印一次提示
```

`checkpoint` 默认指向原实验的 `model-0010.pt`，`model_config` 使用原实验的 `resolved_config.yaml`。PPO 只读取训练 JSON 的原始电路，测试集留作评估。输出目录默认以 `ppo-dual-gpu-` 开头，区分单卡实验。

## 启动

在项目根目录激活训练环境，先检查配置：

```bash
python scripts/train_ppo_dual_gpu.py --dry-run
```

正式启动，两张卡通过 `CUDA_VISIBLE_DEVICES` 指定：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u scripts/train_ppo_dual_gpu.py
```

脚本会自动启动两个 DDP 进程；在 IDE 中选择训练环境的解释器运行该脚本也可以，默认使用第一、第二张可见 GPU。不要同时启动两次脚本。

如果习惯手动使用 torchrun，等价命令为：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_ppo_dual_gpu.py
```

`--dry-run` 只检查配置和路径，不检查 GPU 或 NCCL 是否可用，也不扫描全部文件。正式运行需要两张可见 CUDA GPU；启动时会检查双卡通信能否返回正确的归约结果。

## 数据流与输出

```text
双卡 YAML + 第 10 轮权重
  → rank 0 创建文件清单，分批广播路径
  → GPU 0 / GPU 1 分别批量采样不同电路，每步同时预测多条轨迹的 token
  → 完成的轨迹退出采样 batch，其余轨迹保留 KV 缓存继续生成
  → 多条轨迹补齐长度后批量前向/反向，PAD 不计入损失
  → FP16 + 动态梯度缩放；必要时按 microbatch 累积梯度
  → DDP 同步梯度，按实际参与的轨迹总数取平均
  → 两卡各自更新参数，保持模型一致
  → rank 0 写入日志和检查点
  → 下一批，直到完整遍历数据
```

最后一批不足 128 个电路或两卡分配不均时，按实际轨迹数加权，不丢弃或补重复电路；没有样本的进程仍参与梯度同步。采样阶段两卡不必同时结束，更新阶段会等待彼此。

```text
~/runs/circuit-transformer/
└── ppo-dual-gpu-<时间戳>_all_data/
    ├── resolved_ppo_dual_gpu.yaml # 本次双卡实际配置
    ├── data_manifest.jsonl        # 固定的训练文件顺序
    ├── rollouts.jsonl             # 采样奖励、门数、成功情况；rank 表示采样进程
    ├── metrics.json               # 两卡合计的 PPO 更新指标（每次更新均记录）
    ├── performance.jsonl          # 每批采样/更新耗时、吞吐量与各卡显存峰值
    ├── trainer-latest.pt          # 模型、优化器、进度和两进程随机数状态
    ├── model-latest.pt            # 最近保存的推理权重
    ├── model-ppo.pt               # 全部遍历完成后保存的推理权重
    └── summary.json               # 全部遍历完成后的进度汇总
```

真正影响单卡并行度的参数是 `sampling_batch_size`（采样）和 `train_microbatch_size`（训练）。有效 batch 由 `minibatch_size` 控制：默认两卡合计 32 条，每卡 16 条，单次批量计算 16 条。若降低训练 microbatch 到 8，每卡分两次累积梯度，参数更新频率和有效 batch 大小保持一致。microbatch 超过每卡有效 batch 时，也只会处理实际存在的轨迹。

显存不足时，先降低 `train_microbatch_size`，采样阶段不足则降低 `sampling_batch_size`；显存有余量时可逐步提高这两个参数。若要将每卡训练并行度提高到 32，需要同时将 `minibatch_size` 提高到至少 64。改变有效 batch 会改变参数更新次数；单纯调整 microbatch 不改变每次更新的有效样本数。

控制台打印与文件保存分别控制。默认每 10 个采样批打印进度/性能、每 100 次参数更新打印指标、每保存 10 次检查点打印提示。首批、末批、启动保存及遍历结束会显示必要提示。`metrics.json` 和 `performance.jsonl` 仍记录完整数据。`save_every_batches` 控制实际保存频率，不受打印间隔影响。

`performance.jsonl` 中，`sampling_seconds` / `update_seconds` 为采样和更新阶段耗时，`sampling_trajectories_per_second` / `update_trajectories_per_second` 为对应吞吐量，`peak_allocated_mib` 和 `peak_reserved_mib` 分别记录两张卡的 PyTorch 已分配/已保留显存峰值（MiB）。阶段耗时不包括检查点保存，不是整次训练的总耗时。FP16 指标中的 `loss_scale` 表示梯度缩放因子；`optimizer_step_skipped: true` 表示溢出导致跳过本次更新并调整缩放因子。

`rollout_batch_size` 和 `minibatch_size` 均为两卡总量。每个参数更新的实际轨迹数记录在 `metrics.json` 的 `trajectories` 字段，最后不足一批时可能更少。日志中的 `rank: 0/1` 是训练进程编号，对应第一、第二张可见 GPU。

## 断点续训与评估

把下面路径替换成实际的双卡实验目录：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u scripts/train_ppo_dual_gpu.py \
  --resume "$HOME/runs/circuit-transformer/ppo-dual-gpu-<时间戳>_all_data/trainer-latest.pt"
```

恢复时继续使用状态文件中保存的配置和原输出目录；仍需两个进程。程序从最近保存的完整采样批次恢复，恢复内容包含 FP16 梯度缩放状态；未保存的批次会重新采样和更新，对应日志回退后重写。双卡训练状态与单卡训练状态不能互相直接续训。

`model-ppo.pt` 和 `model-latest.pt` 使用原有推理权重格式，可以直接用于 `scripts/evaluate.py`。在 `configs/evaluate.yaml` 中修改 `checkpoint` 和结果输出路径即可，模型结构配置仍指向原实验的 `resolved_config.yaml`。

## 运行说明

当前默认配置已启用批量采样、批量训练和 FP16。按用户要求未追加吞吐量测试，实际加速幅度以运行时记录为准；完整模型的全量训练尚未启动。
