# Circuit Transformer

这是 ICLR 2025论文 **Circuit Transformer: A Transformer That Preserves Logical Equivalence（电路Transformer：一种可保持逻辑等价性的Transformer）** 的官方实现版本。

模型权重与数据集托管在HuggingFace的独立仓库：

- 模型权重：<[https://huggingface.co/snowkylin/circuit-transformer](https://huggingface.co/snowkylin/circuit-transformer)>
- 数据集：<[https://huggingface.co/datasets/snowkylin/circuit-transformer](https://huggingface.co/datasets/snowkylin/circuit-transformer)>

交互式演示页面：<[https://huggingface.co/spaces/snowkylin/circuit-transformer-demo](https://huggingface.co/spaces/snowkylin/circuit-transformer-demo)>

## 环境安装

Circuit Transformer 目前使用 **PyTorch** 作为运行后端，支持 Python 3.12。
请先安装适配你的Linux服务器CUDA版本的PyTorch，再执行本地项目安装：

```
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

等价性测试、训练加载器行为、数值限制、服务器校验说明可查阅 [`MIGRATION.md`](MIGRATION.md)。程序默认保留8进程数据加载行为。

上游预训练文件是TensorFlow格式权重。你需要在装有TensorFlow的环境中执行一次权重转换，再将转换得到的原生`.pt`权重文件拷贝到你的服务器：

```
python -m pip install -e '.[conversion]'
python -m circuit_transformer.convert_checkpoint /path/deepsyn_reinforced /path/deepsyn_reinforced.pt
```

加载原生权重：
`ct.CircuitTransformer(ckpt_path='/path/deepsyn_reinforced.pt', device='cuda')`

原生`.pt`格式的训练与推理**不需要导入TensorFlow**。加载旧版TensorFlow权重（包括`load_from_hf()`首次下载转换）时才需要TensorFlow；已有转换后的`.pt`缓存时，`load_from_hf()`直接加载PyTorch权重。从零训练无需转换旧权重。

## Linux下启动训练

从数据集仓库下载 `8_inputs_2_outputs_random_deepsyn.zip` 并解压。
修改配置文件 `configs/train_deepsyn.yaml`，将`paths.data_dir`设置为存放`.json`数据文件的文件夹路径。
在同一个yaml配置文件中，设置`paths.output_root`输出根目录、模型参数以及训练超参。
提供的默认配置中`model.ckpt_path: null`，代表使用随机初始化PyTorch权重从头开始训练。

先预览解析后的路径与权重来源，再启动训练：

```
python scripts/train.py --config configs/train_deepsyn.yaml --dry-run
python scripts/train.py --config configs/train_deepsyn.yaml
```

每一轮实验的输出都会保存在配置的输出根目录下，包含：
`ckpt/model-0001.pt`（模型权重）、`resolved_config.yaml`（合并后的完整配置）、`train.log`（训练日志）。

默认8进程数据读取保留原版异步加载逻辑。测试脚本生成的权重存放于临时目录，不会被自动加载。
`--resume /path/model-0010.pt` 参数可以覆盖`model.ckpt_path`，从指定权重初始化模型；**但优化器、学习率调度器会重新初始化，无法做到和原实验完全无缝接续训练**，和原版`train()`行为保持一致。
更多细节参考 [`MIGRATION.md`](MIGRATION.md)，包含服务器验收测试。

如果你需要可视化网络结构，需要额外安装GraphViz：

```
conda install graphviz
```

## 使用示例

Colab交互式demo：[[点击跳转]](https://colab.research.google.com/drive/1r0tS_ZbxUf0ojkazViT94ENMBq4r7B_t?usp=sharing)

Circuit Transformer的实际效果取决于训练数据集。使用默认权重时，该模型用于对**AIG（与反图）**做规模最小化；支持输入不超过8个输入、2个输出的电路。模型是在`(原始AIG，经过最小化后的AIG)`配对数据上做监督训练得到。
经验上，默认权重对AND节点数≤30的AIG电路效果较好。

作为Transformer模型，**批量处理效率更高**：单次API调用同时优化多个AIG电路，会比逐个优化快很多。因此`CircuitTransformer.optimize`默认接收AIG列表作为输入，同样返回优化后的AIG列表。实践表明，单次传入AIG数量大于128时，GPU利用率会更高。

```
import circuit_transformer as ct
# 读取Aiger格式逻辑网络
# 格式参考：[https://fmv.jku.at/aiger/](https://fmv.jku.at/aiger/)
aig, info = ct.read_aiger(aiger_str="""aag 33 8 0 2 25
2\n4\n6\n8\n10\n12\n14\n16\n58\n67
18 13 16\n20 19 7\n22 21 15\n24 3 9\n26 25 11
28 27 17\n30 3 6\n32 29 31\n34 29 32\n36 23 35
38 7 36\n40 10 29\n42 41 32\n44 13 15\n46 42 45
48 47 21\n50 39 49\n52 4 45\n54 25 53\n56 54 5
58 51 57\n60 45 12\n62 18 61\n64 63 19\n66 48 64
""")
circuit_transformer = ct.CircuitTransformer(ckpt_path='/path/deepsyn_reinforced.pt')
aigs = [aig]    # 实际使用建议一次传入多个AIG
optimized_aigs = circuit_transformer.optimize(aigs) # 批量接口，接收AIG列表
optimized_aig = optimized_aigs[0]
print("Optimized AIG: \n%s\n#(AND) from %d to %d, equivalence check: %r" %
      (ct.write_aiger(optimized_aig), 
       ct.count_num_ands(aig), 
       ct.count_num_ands(optimized_aig), 
       ct.cec(aig, optimized_aig)))
```

输出示例（部分换行替换为`\n`）：

```
Optimized AIG: 
aag 24 8 0 2 16
2\n4\n6\n8\n10\n12\n14\n16\n44\n49
18 13 15\n20 19 11\n22 13 16\n24 23 7\n26 21 25
28 3 9\n30 3 6\n32 31 11\n34 29 32\n36 16 7
38 35 37\n40 27 39\n42 29 5\n44 41 43\n46 41 23
48 46 6
#(AND) from 25 to 16, equivalence check: True
```

### 蒙特卡洛树搜索 MCTS

可以启用蒙特卡洛树搜索进一步提升电路规模最小化效果（会带来额外时间开销）。最终效果可以和ABC工具中`&deepsyn`能力相当。

```
optimized_aigs_with_mcts = circuit_transformer.optimize(
    aigs=aigs,
    num_mcts_steps=1,               # 开启MCTS
    num_mcts_playouts_per_step=10   # 每一步MCTS模拟次数
)
optimized_aig_with_mcts = optimized_aigs_with_mcts[0]
print("Optimized AIG: \n%s\n#(AND) from %d to %d, equivalence check: %r" %
      (ct.write_aiger(optimized_aig_with_mcts), 
       ct.count_num_ands(aig),
       ct.count_num_ands(optimized_aig_with_mcts), 
       ct.cec(aig, optimized_aig)))
deepsyn_optimized_aig, info = ct.sequential_synthesis(aig, command='deepsyn')
print("deepsyn optimized AIG: #(AND) = %d" % ct.count_num_ands(deepsyn_optimized_aig))
```

输出示例：

```
Optimized AIG: 
aag 22 8 0 2 14
2\n4\n6\n8\n10\n12\n14\n16\n42\n45
18 3 9\n20 19 5\n22 19 11\n24 23 17\n26 3 6
28 27 11\n30 13 15\n32 28 31\n34 13 16\n36 35 7
38 33 37\n40 25 39\n42 21 41\n44 38 35
#(AND) from 25 to 14, equivalence check: True
deepsyn optimized AIG: #(AND) = 14
```

### 本仓库AIG数据结构

如果你需要手动操作AIG电路，仓库使用两个类来表示AIG：`Node` 和 `NodeWithInv`

- `Node(var, left, right)`：代表二输入与门节点，或者原始输入端口
  - `var=None`：二输入与门；`left`、`right` 接收`NodeWithInv`实例，代表两个输入
  - `var`为非负整数：代表第`var`号原始输入端口
- `NodeWithInv(node, inverted)`：包装类，包裹一个Node节点，`inverted`标记是否取反（`True`代表取反）

一个完整AIG电路就是输出节点的列表。

示例：构造2输入2输出电路

- $o_0 = \neg x_0 \land x_1$
- $o_1 = \neg(x_0 \land \neg x_1)$

```
import circuit_transformer as ct
x_0 = ct.Node(var=0, left=None, right=None)
x_1 = ct.Node(var=1, left=None, right=None)
and_0 = ct.Node(var=None, 
                left=ct.NodeWithInv(x_0, inverted=True), 
                right=ct.NodeWithInv(x_1, inverted=False))
and_1 = ct.Node(var=None, 
                left=ct.NodeWithInv(x_0, inverted=False), 
                right=ct.NodeWithInv(x_1, inverted=True))
aig = [ct.NodeWithInv(and_0, inverted=False), ct.NodeWithInv(and_1, inverted=True)]
print(ct.count_num_ands(aig))
print(ct.compute_tts(aig, num_inputs=2))
```

输出与门数量和真值表：

```
2
[bitarray('0010'), bitarray('1011')]
```

常用AIG工具函数：

- `ct.read_aiger(filename=None, aiger_str=None)`：读取Aiger文件或者Aiger字符串，转为内部AIG对象；返回`(aig, info)`，`aig`是输出节点列表，`info`保存Aiger格式头部5个数字。
- `ct.write_aiger(aig, filename)`：把内部AIG对象转为Aiger格式字符串；指定filename时会写入磁盘文件。
- `ct.count_num_ands(aig)`：统计AIG内部AND门数量。
- `ct.compute_tts(aig, num_inputs)`：计算多输出AIG每个输出的真值表；单节点真值表用`ct.compute_tt`。

基于ABC后端的工具函数：

- `ct.sequential_synthesis(aig, command)`：调用ABC工具优化电路；支持`resyn2`或者`deepsyn`命令。
- `ct.cec(aig1, aig2)`：调用ABC的cec命令做电路等价性检查，等价返回True。

### 无关项 Don't Cares

Circuit Transformer原生支持**无关项**。即只要求部分输入组合保持和原电路逻辑等价。
使用`CircuitTransformer.optimize`参数`care_set_tts`传入每个电路对应的关心集合。
每个关心集合是`bitarray.bitarray`对象，长度为$2^N$；比特位为1代表该输入组合需要保证逻辑等价。

> 
> 示例：2输入电路，`bitarray.bitarray('0110')`，代表仅要求$(x_0=1,x_1=0)$和$(x_0=0,x_1=1)$这两组输入满足等价，其余输入组合不作约束。

### 从零训练 Circuit Transformer

调用`CircuitTransformer.train`可以从零训练模型，需要配置：

- `train_data_dir`：训练数据集目录，目录存放多个`.json`文件。每个json内部是列表，每条样本包含：原始AIG字符串、原始电路与门数量、目标优化后AIG字符串、目标电路与门数量。> 
> 示例json片段：

```
[
    "aag 41 8 0 2 33\n2\n4\n6\n8\n10\n12\n14\n16\n42\n83\n18 7 12\n20 15 12\n22 18 20\n24 15 23\n26 6 8\n28 2 27\n30 25 28\n32 16 2\n34 11 33\n36 8 2\n38 34 36\n40 38 20\n42 31 40\n44 3 11\n46 45 27\n48 5 13\n50 46 49\n52 50 15\n54 51 8\n56 34 5\n58 55 57\n60 52 59\n62 11 36\n64 8 20\n66 62 65\n68 10 15\n70 36 45\n72 68 71\n74 57 63\n76 72 74\n78 67 77\n80 79 58\n82 61 80\n", 
    33, 
    "aag 24 8 0 2 16\n2\n4\n6\n8\n10\n12\n14\n16\n30\n49\n18 8 2\n20 15 12\n22 21 11\n24 23 18\n26 17 11\n28 26 6\n30 28 24\n32 13 8\n34 33 27\n36 35 5\n38 15 10\n40 39 19\n42 8 6\n44 43 41\n46 44 25\n48 46 37\nc\nn\n", 
    16
  ]
```

数据集`8_inputs_2_outputs_random_deepsyn.zip`可在huggingface数据集仓库下载，约500万条样本。原始电路为随机生成8输入2输出AIG；目标电路是ABC中`&deepsyn -T 1`优化后的结果。

- `ckpt_save_path`：模型权重保存文件夹
- `epochs`：训练轮数
- `batch_size`：单批次电路样本数量

训练完成，使用 `CircuitTransformer.load(ckpt_path)` 加载权重。

### 测试数据集

随机电路、IWLS基准测试集放在数据集仓库的`test_data.zip`，格式和训练集一致；评估时只使用原始输入电路。

## 打包构建

```
python setup.py sdist
```

本地安装打包包：

```
pip install dist/circuit-transformer-x.x.tar.gz
```

上传PyPI：

```
twine upload dist/circuit-transformer-x.x.tar.gz
```

## 引用

```
@inproceedings{li2025circuit,
    title={Circuit Transformer: A Transformer That Preserves Logical Equivalence},
    author={Xihan Li and Xing Li and Lei Chen and Xing Zhang and Mingxuan Yuan and Jun Wang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={[https://openreview.net/forum?id=kpnW12Lm9p](https://openreview.net/forum?id=kpnW12Lm9p)}
}
```

# 项目结构与训练流程

| 文件路径 | 功能说明 |
| --- | --- |
| `configs/train_deepsyn.yaml` | 设置数据路径、模型参数、训练超参；`model.ckpt_path: null`=从零初始化训练 |
| `scripts/train.py` | 读取yaml配置，接收命令行参数，预览参数并启动训练 |
| `circuit_transformer/model.py`、`pytorch_transformer.py`、`training.py` | 电路算法对外接口、PyTorch模型主体、训练循环 |
| `circuit_transformer/checkpoints.py` | 保存/加载`.pt`权重；支持转换旧版TensorFlow权重 |
| `tests/` | 小规模训练、权重读写、迁移对照单元测试 |

> 
> yaml仅保存参数，实际计算全部由PyTorch执行。从零训练不会自动加载测试生成的权重。

`--dry-run` 参数仅打印路径、设备、轮数、权重来源，不会生成目录，不会开始训练。

正式运行输出存放于 `paths.output_root/<时间戳>-<run.name>/`：

```
ckpt/model-0001.pt   模型权重
resolved_config.yaml 合并完整配置
train.log            训练日志
tensorboard/         开启性能分析后生成的追踪文件
```

建议数据集、实验输出放在项目仓库外面。`paths.data_dir`必须直接指向json文件所在文件夹。

## 训练启动步骤

以下命令均在项目根目录执行。

1. 安装适配服务器CUDA环境的PyTorch，再运行`python -m pip install -e .`安装项目依赖。运行`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`确认环境；默认配置使用`cuda`。
2. 下载并解压前文的数据集`8_inputs_2_outputs_random_deepsyn.zip`。`paths.data_dir`直接指向存放JSON文件的目录，不要指向压缩包或外层目录，也不要混入其他文件。当前加载器按每个JSON文件一条样本读取，内容为`[原始AIG字符串, 原始AND数量, 目标AIG字符串, 目标AND数量]`。
3. 编辑`configs/train_deepsyn.yaml`：设置`paths.data_dir`、`paths.output_root`和`run.name`，按需要调整模型和训练参数。默认从零训练（`model.ckpt_path: null`），训练10轮、批大小128、验证比例0.1、8进程读取数据。
4. 检查路径与参数，然后正式启动：

```bash
python scripts/train.py --config configs/train_deepsyn.yaml --dry-run
python scripts/train.py --config configs/train_deepsyn.yaml
```

`--dry-run`会检查数据目录并显示运行计划，不导入PyTorch，也不会验证GPU是否可用或逐个解析JSON内容。正式训练的运行目录以正式启动时生成的时间戳为准。

如需从已有PyTorch权重开始新一轮训练，使用下面的命令，并保证配置中的模型结构与权重匹配：

```bash
python scripts/train.py --config configs/train_deepsyn.yaml --resume /path/model-0010.pt
```

`--resume`只恢复模型参数，优化器和学习率进度重新开始；`train.initial_epoch`只控制轮次起点，不恢复训练状态。

## 文件执行顺序与训练流程

只需启动`scripts/train.py`，其余Python文件由入口调用，YAML配置文件不需要单独运行：

```text
scripts/train.py
  → 读取 configs/train_deepsyn.yaml，合并命令行覆盖参数，展开并检查路径
  → 创建实验目录，设置 train.log
  → circuit_transformer/model.py：创建 CircuitTransformer
      → circuit_transformer/pytorch_transformer.py：构建PyTorch Transformer
      → circuit_transformer/checkpoints.py：加载指定权重（从零训练时跳过）
  → 保存 resolved_config.yaml
  → model.py 的 CircuitTransformer.train()
      → training.py 的 train_circuit()：组织数据加载和训练循环
          → utils.py 的 MPDataset：划分训练/验证数据并读取样本
          → model.py 的 load_and_encode_formatted()：解析和编码电路
          → pytorch_transformer.py：前向计算、损失和准确率计算
          → training.py：反向传播、更新参数、验证和记录指标
          → checkpoints.py：保存 .pt 权重
```

训练先将数据文件名排序，以固定随机种子打乱，再按`validation_split`划分训练集和验证集；默认约90%用于训练、10%用于验证。划分只在内存中进行，不会复制JSON文件。

每条样本中的原始电路和目标电路会进行一致的随机输入取反、输入置换和输出取反增强，然后编码为token序列、树位置和合法动作掩码。模型以原始电路为输入、优化后的电路为监督目标；训练时将目标序列右移作为解码器输入，计算忽略PAD的损失和准确率，再反向传播更新参数。优化器使用PyTorch实现的`KerasAdam`以保留原版Adam更新语义，学习率由`CustomSchedule`控制，混合精度由`model.mixed_precision`控制。

每轮先训练，再验证；验证阶段不更新参数。默认8进程异步加载延续原版行为：训练集有N条样本时，每轮返回N−1条，预取队列跨轮延续；`num_workers: 0`或`1`使用单进程，每轮读取全部训练样本。默认`latest_ckpt_only: false`，每轮结束保存一个权重文件，并将训练/验证损失和准确率打印到终端及`train.log`。

## 数据与输出目录

按当前配置，目录结构如下（JSON文件名仅作示意）：

```text
Circuit Transformer/                       # 项目目录
├── readme.md
├── configs/
│   └── train_deepsyn.yaml                  # 训练配置
├── scripts/
│   └── train.py                            # 唯一需要启动的训练入口
└── circuit_transformer/
    ├── model.py                            # 电路接口与编码
    ├── pytorch_transformer.py              # PyTorch模型、损失和优化器
    ├── training.py                         # 训练循环
    ├── utils.py                            # 数据加载及电路工具
    └── checkpoints.py                      # 权重读写

~/data/circuit-transformer/train/
└── 8_inputs_2_outputs_random_deepsyn/       # paths.data_dir，下载解压的输入数据
    ├── sample_000001.json
    ├── sample_000002.json
    └── ...

~/runs/circuit-transformer/                 # paths.output_root
└── <时间戳>-deepsyn_8i2o/                   # 一次实验，后缀来自 run.name
    ├── resolved_config.yaml               # 本次实际使用的配置
    ├── train.log                          # 训练期间的标准输出和标准错误
    ├── ckpt/
    │   ├── model-0001.pt                   # 第1轮结束的模型权重
    │   ├── model-0002.pt
    │   └── ...                            # 默认训练完成后到 model-0010.pt
    └── tensorboard/                       # 启动时创建；profile: true时写入性能追踪
```

训练使用已有配对数据，不会自动运行ABC生成新数据集，也不会将增强后的样本、编码张量或优化后的AIG另存为文件。训练指标记录在`train.log`，同时保留在当前模型对象的`training_history`中；不会自动生成单独的指标CSV。`.pt`包含模型配置和参数，不包含优化器状态。

训练结束后，以与训练相同的模型结构创建`CircuitTransformer`，通过`ckpt_path`或`load()`加载`ckpt/`下的权重，再调用`optimize()`优化电路。优化结果作为AIG列表返回；需要落盘时，显式调用`ct.write_aiger(optimized_aig, filename='/path/result.aag')`指定保存位置。

