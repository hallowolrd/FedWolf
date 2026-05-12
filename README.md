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
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/test1/config.yaml
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
  - 不使用 `mu/P`、不使用 IMQ 权重。
  - 这是 FedWoLF 的 Fisher-only 消融。
- `fedwolf`
  - 完整 FedWoLF。
  - 客户端训练后额外计算 expert Fisher evidence：`s`、`z = log(1+s)` 和 activation support `n`。
  - 服务端为每个 expert 维护 `mu/P`，使用 activation-support-derived observation noise、IMQ 鲁棒权重、Fisher salience、leave-one-out consistency 和 client-expert precision fusion。
  - 不再使用旧插值步骤。

当前没有实现 `fedwolf_kf`。如果需要 Fisher + 普通 scalar filter 的消融入口，建议后续单独补充。

## FedWoLF 流程

客户端：

1. 从 `server.pth` 同步当前全局模型。
2. 在本地 `client_train` 上训练。
3. 训练后额外做一次 evidence pass；默认使用 deterministic evidence loader，并以 eval-mode forward 关闭训练态随机行为。
4. 对每层每个 expert 参数块累计梯度平方，得到 Fisher raw score `s`。
5. 计算 `z = log(1+s)`。
6. 将 `expert_fisher_score_by_layer` 和 `expert_fisher_log_score_by_layer` 随 `client_stats` 返回给 server。

服务端：

1. shared/backbone/router/classifier 继续普通 FedAvg。
2. 对每个 expert：
   - 先做 filter predict，得到 `mu_pred`, `P_pred`
   - 用 activation support 计算 `R`
   - 用 `z = log1p(s)` 和 `R` 做 standardized residual / IMQ / filter precision update
   - 用 Fisher salience、leave-one-out consistency 得到 `lambda_clients`
   - 用 `lambda0` 和 `lambda_clients` 做 client-expert precision fusion
3. 如果某个 expert 本轮没有有效 evidence，则保留旧 global expert。

## FedWoLF 配置

FedWoLF 参数放在 `config.yaml` 的 `train` section：

- `agg_method`
  - 可选：`fedavg`、`expert_fedavg`、`fedwolf_fisher_only`、`fedwolf`
- `aggregation_device`
  - 可选：`cpu`、`cuda`、`cuda:<index>`，默认 `cpu`
  - `cpu`：在 CPU 上聚合，显存占用更稳，和旧版本行为一致
  - `cuda`：将每个参数 key 的浮点聚合临时放到 GPU 上完成，可能减少 CPU 聚合开销
  - 聚合结果仍然转回 CPU state_dict，不会长期保存 GPU 版 client state_dict
- `fedwolf_evidence_loader_mode`
  - 默认 `deterministic`
- `fedwolf_evidence_model_mode`
  - 默认 `eval`
- `fedwolf_fisher_score_mode`
  - 默认 `trace_per_active_sample`
- `fedwolf_fisher_max_samples`
  - 默认 `512`
- `fedwolf_fisher_max_batches`
  - 默认 `null`
- `fedwolf_fisher_debug_batches`
  - 默认 `0`
- `fedwolf_fisher_debug`
  - 默认 `false`
- `fedwolf_eps`
  - 数值稳定项，用于权重归一化和观测噪声分母，默认 `1e-8`
- `fedwolf_process_noise_q`
  - expert 可信度状态的过程噪声，默认 `0.01`
- `fedwolf_sigma_e2`
  - 观测噪声基础强度；FedWoLF 主方法中由 activation support 计算 observation noise
- `fedwolf_imq_c`
  - WoLF-IMQ 鲁棒权重尺度，越小越容易对残差大的 evidence 降权，默认 `1.0`
- `fedwolf_consistency_min`
  - leave-one-out update consistency 的下界，默认 `0.05`
- `fedwolf_lambda_min` / `fedwolf_lambda_max`
  - client-expert precision 的归一化后裁剪范围，默认分别为 `0.05` / `5.0`

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
  - `resnet18_switch_transformer`：ResNet-18 style backbone + Switch Transformer
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
- `--fedwolf_filter_summary`
  - `aggregation_weight_mode`
  - `num_experts`
  - `num_valid_experts`
  - `lambda0`
  - `mean_lambda_filter`
  - `mean_lambda_raw`
  - `mean_lambda_final`
  - `mean_R`
  - `mean_rho`
  - `mean_std_residual`
  - `mean_abs_standardized_residual`
  - `mean_fisher_salience`
  - `mean_update_consistency`
  - `mean_mu`
  - `mean_P`
  - `skipped_observations`

  其中 `mean_s_agg` / `total_s_agg_weight` 以及各类 `min/max` 诊断字段也会保留在 summary 字典里，用于更细的 expert 权重分析，但默认日志行只展开上面的核心字段。

## references 边界

- `references/fedfisher_ref/` 只借鉴“客户端训练后计算 Fisher evidence”的思想。
- `references/wolf_ref/` 只借鉴“服务器端鲁棒 scalar filtering”的思想。
- 主工程不直接 import `references/`。
- 不迁移 JAX / `rebayes_mini` 代码。
- 不复现参考论文原始实验框架。

## 注意事项

- expert Fisher evidence 会在每个客户端训练后额外做一次 forward/backward，训练开销会增加。
- 默认 evidence pass 是 deterministic loader + eval-mode forward；这里的 eval-mode evidence 不是 inference / `no_grad`，而是在关闭训练态随机行为后仍然计算梯度的 Fisher evidence。
- evidence pass 不使用 `torch.no_grad()`，不执行 `optimizer.step()`，不会更新模型参数；结束后会恢复进入 evidence 前的 `model.training` 状态。
- 当前 expert evidence 使用 supervised cross-entropy loss 计算 per-sample empirical Fisher；`router_aux_loss` / `router_z_loss` 是 batch-level auxiliary losses，不纳入 expert Fisher。直接把 batch-level scalar 加到每个 sample loss 会重复计入 batch size 次并破坏 Fisher 尺度，而且当前 evidence 只针对 `blocks.*.ffn.experts.*` expert 参数。
- Fisher score mode 只改变客户端上传的 scalar `s` 的尺度，不改变 server-side FedWoLF 的 `R`、IMQ、`mu/P` 或 expert precision fusion。
- 如果日志里 `mean_s` 只有 `1e-12` 到 `1e-9`、`mean_R` 达到 `1e7` 以上、`mean_kalman_gain` 接近 0、`mu` 长期接近 0，可以做 score mode 尺度消融。`trace_per_active_sample` 对 top-1 MoE 通常更适合作为优先消融，因为每个 expert 只在部分样本上被路由激活。
- 某些 expert 的 Fisher score 可能为 0；此时该 expert 保留旧 global 参数。
- `mu/P` 当前只存在内存中；断点续训如果只恢复 `server.pth`，filter state 会丢失。
- FedWoLF 当前只输出 precision fusion 诊断字段；旧插值相关超参已经删除，当前只使用 client-expert precision fusion。
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
