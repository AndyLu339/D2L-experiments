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

## 开发验证

```bash
python -m unittest discover -s experiments/continual/tests -v
```

数据归组与 manifest 验证仅需 Python 标准库，不需要 GPU 或模型权重。
