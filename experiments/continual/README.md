# SQuAD 持续内化实验

本目录用于加载作者的主实验检查点，比较逐篇重置与累计 chunk LoRA。
第一阶段只做 SQuAD 1.1，不重新训练超网络。

## 实验二：20 篇、三种顺序及加权对照

具体定义见工作区根目录的 `Doc-to-LoRA_持续内化实验二方案.md`。新入口先取
原 seed 11 流的前 20 篇，固定这个集合，再用 seed 22/33 分别打乱；不要把
100 篇流的三个不同前缀当作同一个集合。每五篇分别评估正常累计、加强每组
首篇、每组重置，以及整体缩放和固定目标缩放。只放大 A，且共享 bias 只加一次。

在仓库根目录，先用纯计划模式确认冻结清单和工作量（不需要 PyTorch 或 GPU）：

```bash
python3 -m experiments.continual.run_experiment2 \
  --stream-dir results/continual/squad100 \
  --checkpoint trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
  --output-dir results/continual/experiment2-plan --plan-only
```

在已配置的 GPU 环境，把 `python` 换成该环境的解释器；按原环境需要配置
`PYTHONPATH=src`、模型缓存和兼容脚本。先执行真实模型验收：

如果沿用 Matpool 的 Torch 2.1 / FlashAttention 2.6 环境，将以下命令中的
`-m experiments.continual.run_experiment2` 替换为
`-m experiments.continual.run_experiment2_compat`；它应用与首轮相同的兼容补丁。

```bash
PYTHONPATH=src python -m experiments.continual.run_experiment2 \
  --stream-dir results/continual/squad100 \
  --checkpoint trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
  --output-dir results/continual/experiment2-preflight --preflight-only
```

通过后可使用新的输出目录追加 `--pilot`，先实际运行 seed 11 的第一个五篇
节点并记录耗时。试跑标记为 `pilot_completed`，完整评分工具会拒绝把它当成
正式结果；当前不支持把试跑预测导入后续正式运行。

然后使用**新的输出目录**启动正式运行（入口也会先重新做一次验收）：

```bash
PYTHONPATH=src python -m experiments.continual.run_experiment2 \
  --stream-dir results/continual/squad100 \
  --checkpoint trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
  --output-dir results/continual/experiment2-full

python3 -m experiments.continual.score_experiment2 \
  --run-dir results/continual/experiment2-full
```

输出目录不能已存在；中断后目前不支持续跑，已有 JSONL 和 `status.json` 留作
排错，重跑时使用新目录。GPU 验收失败会停止正式评测，不会生成完成状态。
评测逐题保存原始 token ID、参考答案、状态与文档系数；完全相同的有序状态
和问题会复用生成结果，`prediction_reused` 标识逻辑记录的复用。实际生成成本
应查看 `resources.jsonl`，不要将复用记录的 `seconds` 重复相加。

`config.json` 保存三种顺序、任务、依赖版本和检查点哈希，尽力解析本地模型
权重文件并记录哈希；`resolved: false` 表示没有确认文件，不能声称已完成模型
文件指纹核验。`preflight.json` 保存真实模型验收结果。评分拒绝不完整、重复或
状态不匹配的预测；`summary.csv` 包含每个 seed 的节点和完整 20 篇汇总，
`document_scores.csv` 列示逐篇结果，`paired_by_document.csv` 为同题配对差值，`seed_ranges.csv` 列示三个顺序的
范围。按五篇组的首篇与其余四篇分列；最后一个状态不包含额外历史回测。

从仓库根目录运行。`python` 应替换为已配置 D2L 依赖的环境解释器。

## 1. 固定文档流

```bash
python -m experiments.continual.prepare_streams \
  --num-docs 100 --sample-seed 42 --order-seeds 11 22 33 \
  --output-dir results/continual/squad100
```

默认完整加载 `rajpurkar/squad` 的 validation，然后按上下文精确归组再抽段落。
离线时可传 `--input /path/to/dev-v1.1.json`（官方原始格式），或包含
`id/title/context/question/answers` 的扁平 JSON/JSONL。也可用
`--dataset-source data/raw_datasets/squad` 指定仓库已有数据来源。
下载来源可用 `--revision COMMIT_SHA` 固定；本地文件记录 SHA-256。

输出 `documents.jsonl` 和 `manifest.json`，保留所选段落全部问题与参考答案。
同一问题的重复标注合并，来源 ID 全保留。三个排列使用相同的段落集合；
5、20 段试跑使用固定 100 段流的前缀，不重新抽样。输出目录必须不存在。

## 2. GPU 试跑

先准备主实验权重（`gemma_2b_d2l/checkpoint-20000`）与 Gemma 缓存，
按仓库原有环境安装依赖。首次建议单卡 24GB，先测 5 个段落：

```bash
PYTHONPATH=src python -m experiments.continual.run_stream \
  --stream-dir results/continual/squad100 \
  --checkpoint trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
  --num-docs 5 --order-seed 11 \
  --output-dir results/continual/run5-seed11
```

确认输出与显存后，用不同输出目录依次运行 `--num-docs 20` 和 `100`。
默认运行 Base、ICL、Reset、Seq 四组；默认每层迭代生成 LoRA，chunk 串行编码，
每次更新后组合一次，问题间复用绑定状态。共享 head bias 只加一次。
所有回答都是独立单轮、greedy 解码，问题与答案不用于更新。

`--checkpoints 1 5 10 20 50 100` 控制历史回测，超过文档数的点忽略，
最终点始终包含。正式完整比较保留默认四组；`--groups` 仅供局部调试。
不支持从部分输出恢复，失败运行的输出目录保留以便排错，重跑使用新目录。
默认遇到输入越界、OOM 或无 CUDA 会报错，不截断 SQuAD 文档或丢弃历史。

输出包含固定数据快照、运行配置、逐题预测、累计状态、分阶段显存与时延，
以及明确的 `status.json`。原始 LoRA 的 chain hash 表示按到达顺序链接的
各增量内容指纹，不是重新计算的整个累计矩阵校验值。

## 3. 评分与图表

```bash
python -m experiments.continual.score_stream \
  --run-dir results/continual/run5-seed11 --plots
```

评分本身只需要标准库；`--plots` 额外需要 matplotlib。
它会拒绝未完成的运行、缺题、重复预测以及被修改过的数据快照。
EM/F1 使用 SQuAD 的小写、删除标点、删除冠词、合并空白规则，
每道题对所有参考答案分别评分后取最高值。

| 输出 | 内容 |
| --- | --- |
| `scores.csv` | 每道题的原始预测和 EM/F1（0～1） |
| `document_scores.csv` | 每篇文档在每个评测状态下的平均分（0～1） |
| `summary.csv` | EM/F1 各自的文档宏平均、问题微平均（0～100） |
| `retention.csv` | 各 checkpoint 下全部已见文档的分数及同批 Reset 参照 |
| `resource_summary.csv` | 各阶段耗时、PyTorch 峰值 allocated/reserved，显存单位 GiB |
| `score_summary.json` | 汇总、预测文件指纹、置信区间口径 |
| `figures/` | 已见文档得分曲线、文档×步骤得分矩阵；未测单元留空 |

`summary.csv` 的主要列：

- `seq_online`：每篇刚加入时回答该篇，汇总全部即时回答。
- `seq_final`：最终累计状态重新回答全部文档。
- `immediate_gap`：`seq_online - reset`。
- `final_gap`：`seq_final - reset`。
- `bwt`：每篇旧文档“最终分数 − 刚加入时分数”的平均，排除最后一篇。
- `checkpoint_forgetting`：旧文档在已测状态的最好分数到最终分数的下降。

差距单位为绝对分，不是相对百分比。文档宏平均每篇等权；问题微平均每题等权。
宏平均 BWT 对旧文档等权，微平均 BWT 按旧文档题数加权。
95% 配对 bootstrap 以源文章标题分组，默认 2,000 次，固定随机种子。
区间仅描述固定文档流和累计状态下的评测波动，不代表重新采样知识流的变异。
单篇运行的 BWT 和遗忘量留空；未运行的对照组留空，不记为 0。

三个排列完成后分别保留结果。由于累计规则是独立增量相加，同一文档集合的
最终状态在精确算术下与顺序无关，不能把三个排列的最终得分当独立样本。

## 当前实现边界

本阶段实现累计原始 chunk LoRA 并复用作者的组合与 forward；没有实现权重稠密
合并、固定 rank 压缩或回放。100 个单块文档的 FP32 原始加组合适配器约 1.8 GiB，
仍需额外容纳基础模型、编码器、超网络、激活和加载峰值，因此先测 5 段。
GPU 测量记录的是 PyTorch allocated/reserved，不是 NVML 进程总显存；
生成时间为整个 `generate()` 的耗时，尚不单独测首 token 延迟。

参考：[SQuAD 1.1 官方验证集](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)、
[SQuAD 1.1 评分程序的 Hugging Face 实现](https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py)。

## 开发验证

```bash
PYTHONPATH=src python -m unittest discover -s experiments/continual/tests -v
```

数据归组与 protocol 测试只需标准库。张量测试使用 CPU PyTorch 和仓库依赖，
验证单篇等价、共享 bias、累计增量、重复绑定、多题只读、重置恢复与换序结果。
这些开发测试不需要 GPU 或模型权重，也不替代真实 Gemma 问答与显存验收。

2026-11-21 本地验证：14 项测试通过；实际官方 validation 归组得到 10,570 题、
2,067 段，固定种子选出的 100 段含 488 题（seed 11 的 5/20 段前缀为 26/95 题）。
使用仓库 Gemma tokenizer 离线检查，100 段均为单块，带模板文档长 46～416 token。
对真实 488 题构造 2,440 个预测案例，与 Hugging Face 保留的 SQuAD 1.1
参考评分脚本比较，EM/F1 一致；另以明确标记的合成预测验证评分及画图流程。
尚未在 GPU 上执行真实模型问答，因此这些检查没有产生模型准确率或实测显存结论。
