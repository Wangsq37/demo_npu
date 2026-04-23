# TorchEasyRec 昇腾 Demo 适配说明

## 1. 目标

这个目录是一个**独立于 `TorchEasyRec` 源码**的昇腾版 demo，仅复刻原 `demo.sh` 对应的四段流程：

1. `train_eval`
2. `eval`
3. `export`
4. `predict`

它面向 **鲲鹏 CPU + 单卡昇腾 NPU**，模型结构对齐原始 `multi_tower_din_taobao_local.config` 的核心思路：  
`多塔 Deep 特征 + DIN 序列注意力 + 二分类 CTR 训练/评估/export/predict`。

## 2. 目录结构

```text
demo_npu/
├── configs/
│   └── multi_tower_din_taobao_local_npu.yaml
├── easyrec_npu/
│   ├── config.py
│   ├── data.py
│   ├── device.py
│   ├── eval.py
│   ├── export.py
│   ├── model.py
│   ├── predict.py
│   ├── runtime.py
│   └── train_eval.py
├── demo.sh
└── requirements.txt
```

## 3. 与原 demo 的映射关系

原始 `TorchEasyRec/demo.sh` 等价映射如下：

| 原流程 | 新流程 |
|---|---|
| `python -m tzrec.train_eval` | `python -m easyrec_npu.train_eval` |
| `python -m tzrec.eval` | `python -m easyrec_npu.eval` |
| `python -m tzrec.export` | `python -m easyrec_npu.export` |
| `python -m tzrec.predict` | `python -m easyrec_npu.predict` |

配置文件从 protobuf 风格 `.config` 改成了更易维护的 YAML：  
`configs/multi_tower_din_taobao_local_npu.yaml`

## 4. 环境建议

先完成 CANN 与 Ascend Extension for PyTorch 环境初始化，再安装 Python 依赖。

参考官方文档：

- Ascend Extension for PyTorch 文档入口：<https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html>
- `torch_npu` 快速安装说明：<https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html/>
- GPU 到 NPU 手工迁移说明：<https://www.hiascend.com/document/detail/zh/Pytorch/720/ptmoddevg/trainingmigrguide/PT_LMTMOG_0016.html>

典型启动方式：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd demo_npu
pip install -r requirements.txt
bash demo.sh
```

## 5. 数据路径

默认配置直接复用当前仓库中的 demo 数据：

```yaml
paths:
  train_input_path: "../TorchEasyRec/data/taobao_data_train/*.parquet"
  eval_input_path: "../TorchEasyRec/data/taobao_data_eval/*.parquet"
```

如果你把数据移动到了别的位置，直接修改 `configs/multi_tower_din_taobao_local_npu.yaml` 即可。

## 6. 已做的昇腾适配

### 6.1 设备层

- 用 `easyrec_npu/device.py` 统一设备选择；
- 优先选择 `torch_npu + torch_npu.npu.set_device("npu:0")`；
- 如果当前机器暂时没有安装 `torch_npu`，代码会自动退回 CPU，方便本地静态开发和冒烟验证。

### 6.2 模型层

- 不再依赖 `torchrec / fbgemm-gpu / CUDA`；
- 使用原生 `torch.nn.Embedding + MLP + DIN attention` 重写多塔 DIN；
- 保留原 demo 的核心特征：
  - 用户稀疏特征
  - 商品稀疏特征
  - `price` 分桶 embedding
  - `pid` 哈希 embedding
  - `click_50_seq__*` 序列行为特征

### 6.3 训练/评估层

- 训练损失使用 `BCEWithLogitsLoss`；
- 指标使用 `AUC`；
- 训练阶段分离 embedding 参数与 dense 参数，分别走：
  - `Adagrad`（近似对齐原 sparse optimizer）
  - `Adam`（对齐原 dense optimizer）

### 6.4 导出/预测层

- `export.py` 会导出：
  - `model_state.pt`
  - 尝试生成 `model.pt`（TorchScript）
  - `pipeline.yaml`
- `predict.py` 优先读取 `model.pt`；
- 若 TorchScript 不可用，则自动回退到 `model_state.pt`。

## 7. 数据读取瓶颈是怎么处理的

原始 TorchEasyRec 在 CUDA 生态下有很多高性能配套组件，这里改成昇腾独立实现后，重点做了三件事：

1. **流式 Parquet 读取**  
   不把全部 parquet 一次性读入内存，而是按 `iter_batches` 分批流式读取。

2. **DataLoader 多进程并行**  
   每个 worker 自动按文件切片，避免重复读同一个 parquet 文件。

3. **批级预取 + 序列解析缓存**  
   `PrefetchIterator` 在后台线程里预取 Arrow batch；  
   同时对序列字符串解析用了 LRU cache，减少重复 `split("|")` 的开销。

这套设计虽然不依赖 CUDA/FBGEMM，但对单卡昇腾 demo 已经足够实用。

## 8. 为什么这样适配原 demo

原 `multi_tower_din_taobao_local.config` 的关键难点不在 MLP，而在：

- 稀疏大表 embedding
- 行为序列 DIN 注意力
- parquet 输入管线
- 训练、评估、导出、预测四段链路闭环

这次重写时，我保留了**流程语义**和**模型语义**，但把原先强依赖 CUDA 的实现替换成了：

- 原生 PyTorch 模型
- `torch_npu` 设备管理
- 独立的 parquet 流式 reader
- 自己维护的 checkpoint / export / predict 逻辑

因此它不是“打补丁式兼容”，而是一套**可在昇腾上独立演示完整 demo 流程**的最小工程。

## 9. 没有 1:1 复刻的部分，以及这里的对应处理

这次适配的目标是“在昇腾上独立跑通原 demo 流程”，不是把 `TorchEasyRec` 全仓完全搬过来，所以有几类地方没有做字节级等价复刻，但都做了明确的替代处理。

### 9.1 FG / 特征处理

`TorchEasyRec` 原始流程里有自己的 FG 配置体系和运行时逻辑；这里没有继续依赖那套 CUDA/原框架能力，而是把 demo 用到的特征直接固化到数据管线里：

- `id_feature`：按字段读出后做空值填充，再映射到 embedding id；
- `raw_feature(price)`：按原 config 的 boundary 列表做分桶，再走 embedding；
- `pid`：原 config 是 `hash_bucket_size: 20`，这里使用稳定哈希映射到 20 桶；
- `sequence_feature(click_50_seq__*)`：按 `|` 切分，截断到 `sequence_length=100`，不足部分补 0，并同时生成 `seq_mask`。

也就是说，FG 没有照搬原框架实现，但**demo 真正依赖到的特征语义都保留下来了**。

### 9.2 大桶 sparse id 的处理

原始配置里很多特征桶非常大，比如：

- `user_id: 1141730`
- `adgroup_id: 846812`
- `brand: 461498`

这里仍然按这些桶大小创建 embedding 表；但在样本值落表前，需要特别注意一个和 TorchEasyRec 对齐的点：

- **`0` 不是通用 padding id，而是很多特征里的合法类别值**

例如当前 taobao demo 数据里：

- `cms_segid` 大量样本本身就是 `0`
- `occupation` 绝大多数样本本身就是 `0`
- 历史序列 `click_50_seq__brand` 里也有大量 token 等于 `0`

因此这里最终采用的规则是：

- 空值/非法值映射到 `0`
- 合法 id 直接保留原值
- 超出 bucket 范围的异常值再回退到 `0`

这个处理的目的不是改语义，而是保证：

- 不依赖 TorchEasyRec 的内部字典逻辑；
- 对 demo 数据能稳定落到 embedding；
- 不会把真实的类别 `0` 错误地吞掉；
- 在昇腾单卡上保持实现简单、可控。

这点非常关键：如果把 `0` 误当成统一 padding，会直接损伤模型效果。

### 9.3 DIN 内部实现

原框架里的 `multi_tower_din` 内部细节并没有完整公开为可直接复用的独立模块，因此这里采用了标准 DIN 思路重写：

- query 由当前 item 的 `adgroup_id/cate_id/brand` embedding 拼接；
- keys 由历史序列 `click_50_seq__adgroup_id/cate_id/brand` embedding 拼接；
- attention 输入使用 `[q, k, q-k, q*k]`；
- 经过 attention MLP 后做 softmax 加权求和，得到用户兴趣表示；
- 再和 deep tower 输出拼接，接最终 MLP 做 CTR 二分类。

所以这里是**结构语义对齐**，不是**算子级完全一致**。

### 9.4 参数初始化

原始 `TorchEasyRec` 各层的默认初始化策略，这里没有逐项追溯和完全复刻。当前实现使用的是 **PyTorch 原生模块默认初始化**：

- `nn.Embedding`
- `nn.Linear`

这样做的原因是：

- 昇腾 demo 的核心目标是流程跑通与结构适配；
- 默认初始化足以支持该 demo 训练；
- 如果后续需要贴近原始数值表现，再单独对齐初始化策略会更合理。

因此，**训练结果趋势应可复现，数值不会承诺与原框架逐步一致**。

### 9.5 优化器实现

原 config 中有：

- sparse optimizer：`Adagrad(lr=0.001)`
- dense optimizer：`Adam(lr=0.001)`

这里保留了同样的优化器类型和学习率，并把 embedding 参数与非 embedding 参数分组优化。  
但这仍然不是对原框架 optimizer 包装层的逐项复刻，例如：

- 是否有额外 lr scheduler 包装
- 是否有框架内部状态管理
- 是否有 fused/kernel 级优化

这些没有照搬，不过**优化器语义和超参已经对齐原 demo**。

### 9.6 数据读取与性能路径

原始 TorchEasyRec 在 CUDA 生态里可能会依赖更成熟的高性能数据组件；这里没有复刻那些依赖，而是换成更适合当前昇腾独立工程的方案：

- `pyarrow.parquet.ParquetFile.iter_batches` 流式读取；
- DataLoader 多 worker 按文件切片；
- 后台线程预取 Arrow batch；
- 序列字符串解析加 LRU cache。

因此，**性能路径不一样**，但目标是一致的：尽量缓解 CPU 侧读数与特征解析瓶颈，避免 NPU 等待。

### 9.7 export / predict 产物形式

原框架 export 后的产物组织和加载方式，这里也没有完全照搬。当前导出策略是：

- 优先导出 `TorchScript`：`model.pt`
- 同时保留 `model_state.pt`
- 额外导出 `pipeline.yaml`

预测时优先读取 `model.pt`，失败则回退到 `model_state.pt`。  
这保证 demo 流程闭环，但**文件组织形式与 TorchEasyRec 原生产物并不要求完全一致**。

### 9.8 结论

可以把这次适配理解为：

- **配置语义对齐**
- **模型结构对齐**
- **训练/评估/export/predict 流程对齐**
- **昇腾运行方式对齐**

但不是：

- **TorchEasyRec 内部实现逐文件搬运**
- **CUDA 依赖逐项替换后的全仓兼容**
- **所有中间数值与原框架逐步 bitwise 对齐**

对当前“只适配 demo 流程”的目标来说，这样的处理是刻意的：优先保证独立、可运行、可解释、可继续演进。

## 10. 运行建议

当前默认配置按原 `multi_tower_din_taobao_local.config` 对齐：

- `batch_size: 8192`
- `num_workers: 8`
- `sequence_length: 100`
- `num_epochs: 1`
- `sparse_lr: 0.001`
- `dense_lr: 0.001`

如果只是想快速冒烟，可以临时把 `max_train_steps`、`max_eval_steps`、`max_train_files`、`max_eval_files` 改成较小值。
