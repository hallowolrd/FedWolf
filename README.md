# FedWolf

这是一个基于 CIFAR10/CIFAR100 的 FL + MoE / Switch Transformer 聚合实验项目。当前主工程包含标准 baseline 聚合方法，以及面向 expert 参数的 FedWoLF 聚合方法。

`references/` 只作为论文代码和映射说明的参考区，不是生产代码路径。主工程不应直接 import `references/`。

## 环境准备

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate fedwolf
```

如果环境已经存在，直接激活即可：

```bash
conda activate fedwolf
```

## 配置方式

项目现在使用一个 `config.yaml` 作为唯一配置来源，不再依赖 `configs/args.py + argparse`。
配置加载使用真正的 `PyYAML` 解析，也就是 `yaml.safe_load(...)`，不再使用手写的逐行 flat parser。

`config.yaml` 顶层必须包含三个 section：

- `data`
  - 数据集、划分协议、随机种子等
- `model`
  - 模型结构和优化器超参数等
- `train`
  - 联邦训练轮数、设备、实验输出根目录和实验名等

项目入口会调用 `configs/__init__.py` 中的 `load_args()`，将 `data/model/train` 三层配置 flatten 成一个扁平的 `args` 对象，因此项目内部仍然继续使用 `args.xxx` 访问配置，而不是 `args.data.xxx`。

`data/model/train` 三个 section 都必须是 key-value mapping。为避免 flatten 时产生歧义，三个 section 里的 key 必须全局唯一；如果出现重复 key，`load_args()` 会直接报错，而不是静默覆盖。
默认假设从项目根目录运行；不传 `--config` 时会读取 `configs/config.yaml`，也可以通过 `--config` 手动指定其他实验配置。

默认配置：

```bash
python train.py
```

手动指定其他配置：

```bash
python train.py --config configs/test1/config.yaml
```

输出目录由 `config.yaml` 的 `train.save_root` 和 `train.run_name` 自动派生：

- `save/{run_name}/data`
- `save/{run_name}/model`
- `save/{run_name}/result`

开新实验时只需要改 `run_name`。如果同名输出目录已经存在且非空，`allow_overwrite: false` 会直接报错；确认要复用旧目录时再设置 `allow_overwrite: true`。

## 支持的聚合方法

在 `config.yaml` 的 `train.agg_method` 中切换：

- `fedavg`
  - 全模型按客户端训练样本数加权平均。
- `expert_fedavg`
  - shared/backbone/router/classifier 按客户端样本数 FedAvg。
  - expert 参数按 expert usage / token usage 加权。
- `fedwolf_fisher_only`
  - shared/backbone/router/classifier 按客户端样本数 FedAvg。
  - expert 参数按客户端上传的 Fisher raw score `s` 加权。
  - 不使用 `mu/P`，不使用 IMQ 权重，不使用 gamma。
  - 这是 FedWoLF 的 Fisher-only 消融。
- `fedwolf`
  - 完整 FedWoLF。
  - 客户端训练后额外计算 expert Fisher evidence：`s` 和 `z = log(1+s)`。
  - 服务端为每层每个 expert 维护 `mu/P` 状态。
  - 使用 WoLF-IMQ 权重对残差大的 evidence 降权。
  - 使用 `gamma = sigmoid(mu)` 在 old global expert 和 Fisher weighted expert average 之间插值。

当前没有实现 `fedwolf_kf`。如果需要 Fisher + 普通 scalar filter 的消融入口，建议后续单独补充。

## FedWoLF 流程

客户端：

1. 从 `server.pth` 同步当前全局模型。
2. 在本地 `client_train` 上训练。
3. 训练后额外做一次 evidence pass。
4. 对每层每个 expert 参数块累计梯度平方，得到 Fisher raw score `s`。
5. 计算 `z = log(1+s)`。
6. 将 `expert_fisher_score_by_layer` 和 `expert_fisher_log_score_by_layer` 随 `client_stats` 返回给 server。

服务端：

1. shared/backbone/router/classifier 继续普通 FedAvg。
2. 对每层每个 expert，按客户端顺序用 `z/s` 更新 scalar filter state：
   - `mu_pred = mu_prev`
   - `P_pred = P_prev + q`
   - `R = sigma_e2 / (s + eps)`
   - `residual = z - mu`
   - `w = (1 + residual^2 / c^2)^(-0.5)`
   - `obs_precision = w^2 / R`
3. 用 Fisher raw score `s` 得到 `Theta_bar`。
4. 对 `fedwolf`，计算 `gamma = sigmoid(mu)`。
5. 最终 expert 更新：

```text
expert_new = (1 - gamma) * expert_old + gamma * Theta_bar
```

如果某个 expert 本轮 Fisher 总权重为 0，则保留旧 global expert，不用随机客户端参数覆盖。

## FedWoLF 配置

FedWoLF 参数放在 `config.yaml` 的 `train` section：

- `agg_method`
  - 可选：`fedavg`、`expert_fedavg`、`fedwolf_fisher_only`、`fedwolf`
- `fedwolf_eps`
  - 数值稳定项，用于 Fisher 权重归一化和观测噪声分母，默认 `1e-8`
- `fedwolf_process_noise_q`
  - expert 可信度状态的过程噪声，默认 `0.01`
- `fedwolf_sigma_e2`
  - 观测噪声基础强度，`R = sigma_e2 / (s + eps)`，默认 `1.0`
- `fedwolf_imq_c`
  - WoLF-IMQ 鲁棒权重尺度，越小越容易对残差大的 evidence 降权，默认 `1.0`

当前代码没有使用 `fedwolf_gamma_temperature`，因此配置文件中不提供该字段。gamma 直接使用 `sigmoid(mu)`。

## 实验切换方式

- 切 CIFAR10 / CIFAR100：修改当前 `config.yaml` 的 `data.data_name`
- 改 `alpha`：修改当前 `config.yaml` 的 `data.alpha`
- 改客户端数量：修改当前 `config.yaml` 的 `data.num_clients`
- 切聚合方法：修改当前 `config.yaml` 的 `train.agg_method`，可选 `fedavg`、`expert_fedavg`、`fedwolf_fisher_only`、`fedwolf`
- 开新实验：复制一个 `config.yaml`，并修改 `train.run_name`
- 故意覆盖旧实验：保留同一个 `run_name`，并设置 `train.allow_overwrite: true`
- 切模型：修改当前 `config.yaml` 的 `model.model_type`
  - `hybrid_switch_transformer`：CNN stem + Transformer
  - `switch_transformer`：patch embedding + Transformer
  - `resnet20_switch_transformer`：ResNet-20 style backbone + Switch Transformer
  - `resnet32_switch_transformer`：ResNet-32 style backbone + Switch Transformer
  - `switch_transformer` 现在支持显式 `patch_size`；该字段只对标准 Switch 生效
  - `hybrid_switch_transformer` 仍然使用 `token_grid_size` 控制 token 网格
  - 结果文件名现在会区分 `model_type`；对 `switch_transformer` 还会进一步区分 `patch_size`
- 改完会影响数据划分的配置后，可以直接运行 `train.py`。例如 `data.data_name`、`data.alpha`、`data.num_clients`、`data.seed`、`data.min_datasize`、`data.data_path` 变化时，`train.py` 会检测旧 partition 是否与当前 config 匹配，不匹配就自动重新生成。
- 如果希望无论是否匹配都重新划分，加 `--force_repartition`。
- 如果只改 `model` 或 `train` 中不影响数据划分的参数，partition 会被复用。

默认配置写法：

```bash
python train.py
```

手动指定其他配置：

```bash
python train.py --config configs/test1/config.yaml
```

这些模型当前都提供一致的 MoE 辅助接口，包括 expert/router state dict 提取和 parameter groups。

## 运行顺序

推荐直接启动训练。`train.py` 会在训练前自动检查数据划分文件是否存在、是否和当前配置匹配；如果需要，会自动生成或重新生成 `partition_meta.pt` 和 `partition_stats.json`。

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/test1/config.yaml
```

强制重新划分数据：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/test1/config.yaml \
  --force_repartition
```

关闭自动数据准备，恢复以前行为：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/test1/config.yaml \
  --no_auto_prepare_data
```

如果只想手动生成数据划分，也可以单独运行：

```bash
python -m data.data --config configs/test1/config.yaml
```

正常训练时不再必须手动先运行 `data.py`。

## 最小 Smoke Test

建议复制一份小规模 `config.yaml`，或临时修改当前 `config.yaml` 的 `train` section：

```yaml
num_clients: 2
server_epochs: 2
client_epochs: 1
batch_size: 64
allow_overwrite: false
```

依次测试：

```yaml
agg_method: fedavg
run_name: smoke_fedavg
```

```yaml
agg_method: expert_fedavg
run_name: smoke_expert_fedavg
```

```yaml
agg_method: fedwolf_fisher_only
run_name: smoke_fedwolf_fisher_only
```

```yaml
agg_method: fedwolf
run_name: smoke_fedwolf
```

如果修改了 `train.run_name`，输出目录会变化；直接运行 `train.py` 时会为该 run 自动检查并生成 partition。

## 数据协议

当前项目使用的是 index-based partition 协议：

- official `train` 全部通过 Dirichlet non-IID 划分得到各客户端的 `client_train_indices`
- official `test` 直接作为 `global_test`，由 server 持有
- `partition_meta.pt` 只保存索引和元信息，不保存原始图像数据
- `partition_stats.json` 保存各 split 的样本规模和类别统计
- 这些文件可以由 `train.py` 自动生成，不一定需要用户手动执行 `data.py`

这意味着训练阶段仍然需要 `data_path` 下存在原始 CIFAR 数据文件。`data/loader.py` 会基于：

- raw CIFAR dataset
- saved indices
- split-specific transforms

动态构造 `Dataset` / `DataLoader`。

如果训练时报原始 CIFAR 缺失，请检查当前 `config.yaml` 的 `data.data_path`，或者重新运行：

```bash
python train.py --config configs/test1/config.yaml --force_repartition
```

## 训练与评估协议

- client 只训练自己的 `client_train`
- server 不再划分和使用验证集
- `global_test` 不参与训练
- server 每轮聚合后会在 `global_test` 上评估一次用于曲线监控
- 训练结束后再做一次最终 `global_test`
- 不根据 `global_test` 保存 best model

## 输出文件

### 数据划分

- `save/{run_name}/data/partition_meta.pt`
  - 索引划分协议和元信息
- `save/{run_name}/data/partition_stats.json`
  - 各 split 的样本数量和类别分布统计

### 模型文件

- `save/{run_name}/model/server.pth`
  - 当前轮 / 最后一轮服务端模型的纯 `state_dict`
- `save/{run_name}/model/{client_id}.pth`
  - 每个客户端当前模型的纯 `state_dict`

### 结果与日志

- `save/{run_name}/result/detail/*.csv`
  - client 侧逐轮训练明细
- `save/{run_name}/result/server/*.csv`
  - server 侧 `round_test` 和 `final_test` 结果
- `save/{run_name}/result/logs/*.log`
  - 本次实验的完整日志

CSV 和日志文件名都会包含：

- `data_name`
- `num_clients`
- `alpha`
- `seed`
- `agg_method`
- `run_name`

FedWoLF 日志中可观察：

- `--expert_fisher_score_by_layer`
- `--expert_fisher_log_score_by_layer`
- `--fedwolf_filter_state_summary`
  - `mu`
  - `P`
  - `mean_weight`
  - `mean_abs_residual`
  - `mean_kalman_gain`
  - `gamma`
  - `total_fisher_weight`

## references 边界

- `references/fedfisher_ref/` 只借鉴“客户端训练后计算 Fisher evidence”的思想。
- `references/wolf_ref/` 只借鉴“服务器端鲁棒 scalar filtering”的思想。
- 主工程不直接 import `references/`。
- 不迁移 JAX / `rebayes_mini` 代码。
- 不复现参考论文原始实验框架。

## 注意事项

- expert Fisher evidence 会在每个客户端训练后额外做一次 forward/backward，训练开销会增加。
- 某些 expert 的 Fisher score 可能为 0；此时该 expert 保留旧 global 参数。
- `mu/P` 当前只存在内存中；断点续训如果只恢复 `server.pth`，filter state 会丢失。
- `gamma = sigmoid(mu)` 可能过早饱和；可优先调 `fedwolf_process_noise_q`、`fedwolf_sigma_e2`、`fedwolf_imq_c`。
- `fedwolf_fisher_only` 和 `fedwolf` 语义不同：前者是消融，后者是完整方法。
- `train.py` 的自动数据准备只会覆盖 `save/{run_name}/data` 下的 `partition_meta.pt` 和 `partition_stats.json`。
- 如果 `train.allow_overwrite: false`，`train.py` 仍然会检查 `model/result` 输出目录是否非空，避免误覆盖训练结果。
- 自动数据准备不会改变模型结构、训练参数或聚合逻辑。
- 如果只改 `model` 或 `train` 中不影响数据划分的参数，partition 会被复用。
- 如果改了 `data` section 中影响划分的参数，partition 会自动重新生成。

## 常用命令

检查核心依赖版本：

```bash
python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
```

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
```
