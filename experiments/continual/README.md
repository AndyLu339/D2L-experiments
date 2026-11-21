# SQuAD 持续内化实验

本目录用于加载作者的主实验检查点，比较逐篇重置与累计 chunk LoRA。
第一阶段只做 SQuAD 1.1，不重新训练超网络。

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

## 开发验证

```bash
PYTHONPATH=src python -m unittest discover -s experiments/continual/tests -v
```

数据归组与 protocol 测试只需标准库。张量测试使用 CPU PyTorch 和仓库依赖，
验证单篇等价、共享 bias、累计增量、重复绑定、多题只读、重置恢复与换序结果。
这些开发测试不需要 GPU 或模型权重，也不替代真实 Gemma 问答与显存验收。
