# TensorFlow → PyTorch 迁移与验收

本项目的运行后端已改为 PyTorch。运行目标仅为 Linux 服务器＋Python 3.12，不新增 Windows 兼容处理。本机仅用于修改代码和数值对照。原始版本基线：`0fc85a5798575334cbd18dab32545c4fcbcc2b1d`。

## 语义保持范围

- 电路 token 编码、树位置编码输入、动作可行性约束、don't-care、节点共享、奖励、MCTS 选择／回传和溢出处理保持原算法。测试通过 AST 对比检查主要方法未改动。
- Transformer 保留 pre-norm、ReLU、attention 无偏置、前馈层有偏置、1e-6 LayerNorm、最终归一化、词嵌入缩放和解码器输出权重共享。
- 保留训练时目标 token 与树位置右移，候选动作掩码不右移；推理缓存的 K/V 布局仍为“批次、序列、头数、头维度”。
- 保留 PAD=0 的损失／准确率屏蔽规则、warmup 学习率（第 0 步为 0）、Keras Adam 的 epsilon 放置与偏差修正，以及混合精度动态 loss scaling。溢出跳过更新时，迭代计数仍按 Keras 规则增加。
- 公开的 `CircuitTransformer`、`train`、`optimize`、`optimize_batch`、`load` 和 `load_from_hf` 入口保留；新增 `device`、`save` 和 `num_workers`。
- 原版 `distributed=True` 会另建模型，只恢复 `ckpt_path`，不会把训练后的独立模型赋回当前对象。迁移保留这一行为，训练后请重新 `load` 保存的权重。

## 训练数据读取：按用户要求保留原行为

默认 `num_workers=8`。继续使用 1024 个文件分块、`imap_unordered(..., chunksize=128)`、跨 epoch 持续存在的队列，以及“计数先加 1、再判断计数 < 训练样本数”的原始条件。若训练样本数为 N，每轮返回 N−1 个，下一项留在队列中进入后续轮次。没有改成完整遍历，也没有新增 worker 随机种子或顺序排序。

直接调用 `utils.py` 中原封未动的 `MPDataset`；没有新增 Windows 兼容层，也没有改写原多进程生产者。Linux 多进程回归测试仅在 Linux 执行，在其他系统上会明确跳过。

`num_workers=0` 或 `1` 对应原 MPDataset 的单进程分支，每轮返回全部样本；它是显式调试选项，不是迁移后的默认值。异步读取与随机增强本来就不能保证固定训练轨迹。

## Linux 服务器使用

先按服务器 CUDA 环境安装对应 PyTorch，再安装项目：

```bash
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

远端提交 `cd4b2eb` 的 `configs/train_deepsyn.yaml`、`scripts/train.py` 和
`scripts/download_data.sh` 已合入。训练入口继续使用原来的数据目录、时间戳运行目录
和日志布局；输出权重改为 `ckpt/model-0001.pt`。先检查配置中的服务器路径，再执行：

```bash
python scripts/train.py --config configs/train_deepsyn.yaml --dry-run
python scripts/train.py --config configs/train_deepsyn.yaml
```

`--resume /path/model-0010.pt` 从模型权重开始新训练；与原版训练入口相同，
优化器状态和学习率步数不会恢复。`initial_epoch` 只决定循环起点，不恢复优化器。
配置文件默认 `device: cuda`、`num_workers: 8`；训练脚本支持命令行覆盖。

```python
import circuit_transformer as ct

if __name__ == '__main__':
    model = ct.CircuitTransformer(device='cuda', mixed_precision=False)
    model.train(
        train_data_dir='/data/circuits',
        ckpt_save_path='/data/checkpoints',
        batch_size=128,
        epochs=10,
        num_workers=8,
    )
```

FP32 便于迁移验收。原默认 `mixed_precision=True` 仍保留，开启后使用 FP16 计算和 FP32 参数／归一化，不改变模型结构。

## 旧权重转换

支持项目官方发布的 TensorFlow object checkpoint（`.index` 与 `.data-*`）。按明确变量路径转换每个张量，转换 attention 头维布局与线性层转置；缺少、重复、额外模型变量或尺寸错误会直接报错，不会静默跳过。optimizer slots 不属于权重转换范围，原 `train()` 本身也会新建 optimizer。

在转换环境执行一次：

```bash
python -m pip install -e '.[conversion]'
python -m circuit_transformer.convert_checkpoint /path/deepsyn_reinforced /path/deepsyn_reinforced.pt
```

自定义结构需传递对应参数，可用 `--help` 查看。把 `.pt` 复制到服务器后，日常训练／推理不需要 TensorFlow：

本次已将官方 `deepsyn_reinforced` 权重转换为仓库下的 `checkpoints/deepsyn_reinforced.pt`（约 353 MB，367 个参数张量），并通过保存／加载逐张量完全一致检查。该大文件已加入 Git 忽略规则，部署时需要单独复制。

```python
model = ct.CircuitTransformer(ckpt_path='/path/deepsyn_reinforced.pt', device='cuda')
optimized = model.optimize(aigs)
model.save('/path/another-copy.pt')
```

`load_from_hf()` 保留原模型名称与来源。上游目前发布 TF 权重，第一次下载／转换需要 TensorFlow；同一缓存目录已存在转换后的 `.pt` 时直接读取 PyTorch 权重。也可以在本机转换后上传，避免服务器安装 TensorFlow。

## 验证与边界

本机环境：Python 3.12；PyTorch 2.14.0+cpu；TensorFlow／tf-keras／tf-models-official 2.18.0 固定参考版本。参考源码放在 `tests/reference`，不会作为运行后端导入。

验收包括：同权重前向分数、动作掩码、encoder 输出、逐步 KV 缓存、loss／accuracy、每个参数的梯度、连续训练更新、Adam 微小梯度／溢出计数、beam search、greedy／MCTS 动作及奖励、don't-care 真值表、权重往返、训练后再推理。原生 Linux 多进程 epoch 边界和队列延续测试已提供，待服务器执行。

FP32 官方 12 层预训练模型在 README 示例电路上的已验证结果：两版均把 25 个 AND 节点变为 16 个；greedy 及 MCTS（2 次 playout、2 个并行叶节点）动作和奖励序列一致，完整真值表一致。对照输出分数最大绝对误差约 3.6×10⁻⁵。该设置不等于 README 中 10 次 playout 的示例，不能据此声称已复现其 14 节点结果。

同一完整预训练模型的混合精度对照也已通过：两版 greedy／MCTS 都得到 16 个 AND 节点，动作、奖励和真值表一致；分数最大绝对误差分别为 0.162109375 和 0.18359375。FP32 与混合精度的 MCTS 调用次数不同，因此这里证明的是各自精度模式内的跨框架样例一致性。

合并训练入口后的常规测试：**21 通过、6 跳过**。新增测试通过 YAML 入口实际完成一轮混合精度训练并保存 `.pt`；它使用小样本与单进程读取，只验证入口和权重保存。6 个跳过项分别为 2 个 Linux 多进程测试、2 个需显式启用的原仓库大型示例，以及 2 个完整预训练权重测试。后两项已在独立启用权重的运行中完成，结果为 **2 通过**（CPU，约 35 分钟），并非将跳过计作通过。

**数值接近、测试样例动作相同，不是对所有输入逐位一致的证明。** 不同计算库的浮点归约顺序会产生误差；接近并列的动作可能交换次序，ReLU 零点附近也可能出现不同梯度。小模型固定样例的混合精度误差约 0.007，完整模型约 0.184，均不能推广成任意训练过程的误差界。严格验收建议先采用 FP32、同一 checkpoint 和固定输入；混合精度应单独验收动作与电路等价性，不以分数逐位一致为标准。

本机未验证 Linux 服务器的 CUDA 内核、多卡执行和完整数据集训练。多卡路径按原版“先各卡归一化有效 token 损失，再平均各卡标量”的方式归约；不能把单卡／CPU 测试替代真实多卡验收。新模型随机初始化分布保持相同意图，但 PyTorch 与 TensorFlow 相同 seed 不会产生相同权重；同权重对照应加载转换后的 checkpoint。

运行测试：

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
# 安装参考环境后运行真正的跨框架对照：
python -m pytest tests/test_tensorflow_parity.py -q -s
# 完整官方权重验收（较慢，尤其 CPU FP16）：
CT_TF_CHECKPOINT=/path/deepsyn_reinforced python -m pytest tests/test_pretrained_parity.py -q -s
# Linux GPU 上进行相同验收，需要两个框架均支持该服务器的 CUDA：
CT_TEST_DEVICE=cuda CT_TF_CHECKPOINT=/path/deepsyn_reinforced python -m pytest tests/test_pretrained_parity.py -q -s
```

参考框架的固定安装方式见 `tests/reference/README.md`。GPU 对照会检查两个框架都能使用 GPU，关闭 TF32 并让 TensorFlow 按需分配显存，避免静默退回 CPU；两个完整模型同时驻留仍需足够显存。服务器应先运行常规测试（包含 Linux 多进程），再运行同权重 GPU 对照，最后使用实际数据进行短程训练、保存和重新加载验收。多卡训练需在实际卡数和 batch size 下单独检查。

原仓库测试中需要联网／大型训练集的示例改为显式环境变量启用；它们被跳过时，不应计为通过。完整预训练验收与服务器验收结果应单独记录。
