
 # Finetune
在这个例子中，我们将展示如何使用您的数据对baai-general-embedding进行微调。

## 1. 安装
* **使用pip**
```
pip install -U FlagEmbedding
```

* **从源代码**
```
git clone https://github.com/FlagOpen/FlagEmbedding.git 
cd FlagEmbedding
pip install  .
```
对于开发，以可编辑模式安装：
```
pip install -e .
```

Transformers的新版本可能会对微调造成问题。如果您遇到问题，可以尝试降级到版本4.33-4.36。

## 2. 数据格式
训练数据应该是一个json文件，每行都是一个类似于以下的字典：

```
{"query": str, "pos": List[str], "neg":List[str]}
```

`query`是查询，`pos`是正面文本列表，`neg`是负面文本列表。
如果您没有查询的负面文本，可以从整个语料库中随机抽样一些作为负面文本。

查看[toy_finetune_data.jsonl](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/toy_finetune_data.jsonl) 以获取一个玩具数据文件。

### 硬负样本

硬负样本是一种广泛使用的方法，用于提高句子嵌入的质量。
您可以按照以下命令挖掘硬负样本：
```bash
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file toy_finetune_data.jsonl \
--output_file toy_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 
```

- `input_file`: 用于微调的json数据。这个脚本将为每个查询检索前k个文档，并从这些文档中随机抽样负面文本（不包括正面文档）。
- `output_file`: 保存带有挖掘出的硬负样本的微调JSON数据的路径。
- `negative_number`: 抽样的负面文本数量。
- `range_for_sampling`: 抽样负面的文档范围。例如，`2-100`表示从前2到前200个文档中抽样`negative_number`个负面。**您可以设置更大的值来降低负面的难度（例如，设置为`60-300`从前60到前300个段落中抽样负面）**
- `candidate_pool`: 检索池。默认值是None，这个脚本将从`input_file`中的所有`neg`组合中检索。这个文件的格式与[预训练数据](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain#2-data-format)相同。如果输入一个`candidate_pool`，这个脚本将从这个文件中检索负面。
- `use_gpu_for_searching`: 是否使用faiss-gpu进行检索。

## 3. 训练
```
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-zh-v1.5 \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size {large batch size; set 1 for toy data} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" 
```

**一些重要的参数**：
- `per_device_train_batch_size`: 训练中的批次大小。在大多数情况下，更大的批次大小会带来更强的性能。您可以通过启用`--fp16`、`--deepspeed ./df_config.json`（df_config.json可以参考[ds_config.json](./ds_config.json)）、`--gradient_checkpointing`等来扩展它。
- `train_group_size`: 训练中每个查询的正面和负面数量。由于总是有一个正面，所以这个参数将控制负面的数量（#negatives=train_group_size-1）。
请注意，负面数量不应大于数据`"neg":List[str]`中的负面数量。
除了这个组中的负面，批次内的负面也将用于微调。
- `negatives_cross_device`: 在所有GPU之间共享负面。这个参数将扩展负面的数量。
- `learning_rate`: 为您的模型选择一个合适的学习率。对于大型/基础/小型模型，推荐使用1e-5/2e-5/3e-5。
- `temperature`: 它将影响相似度分数的分布。
- `query_max_len`: 查询的最大长度。请根据您的数据中查询的平均长度设置。
- `passage_max_len`: 段落的最大长度。请根据您的数据中段落的平均长度设置。
- `query_instruction_for_retrieval`: 查询的指令，将添加到每个查询中。您也可以将其设置为`""`以不添加任何内容。
- `use_inbatch_neg`: 使用同一批中的段落作为负面。默认值为True。

有关更多训练参数，请参考[transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)

### 4. 通过[LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)进行模型合并

有关更多细节，请参考[LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)。

微调基础bge模型可以提高其在目标任务上的性能，
但可能会导致模型在目标域之外的一般能力严重退化（例如，在c-mteb任务上的性能降低）。
通过合并微调模型和基础模型，LM-Cocktail可以在保持其他无关任务性能的同时显著提高下游任务的性能。

```python
from LM_Cocktail import mix_models, mix_models_with_data

# 混合微调模型和基础模型；然后将它保存到输出路径：./mixed_model_1
model = mix_models(
    model_names_or_paths=["BAAI/bge-large-en-v1.5", "your_fine-tuned_model"], 
    model_type='encoder', 
    weights=[0.5, 0.5],  # 您可以更改权重以获得更好的权衡。
    output_path='./mixed_model_1')
```

如果您有新任务，并且没有数据或资源可用于微调，
您可以尝试使用LM-Cocktail合并现有模型（来自开源社区或您在其他任务上微调的模型）以产生特定于任务的模型。
这样，您只需要构建一些示例数据，而不需要微调基础模型。
例如，您可以使用您的任务的示例数据合并[huggingface](https://huggingface.co/Shitao)上的模型：
```python
from LM_Cocktail import mix_models, mix_models_with_data

example_data = [
    {"query": "How does one become an actor in the Telugu Film Industry?", "pos": [" How do I become an actor in Telugu film industry?"], "neg": [" What is the story of Moses and Ramesses?", " Does caste system affect economic growth of India?"]}, 
    {"query": "Why do some computer programmers develop amazing software or new concepts, while some are stuck with basic programming work?", "pos": [" Why do some computer programmers develops amazing softwares or new concepts, while some are stuck with basics programming works?"], "neg": [" When visiting a friend, do you ever think about what would happen if you did something wildly inappropriate like punch them or destroy their furniture?", " What is the difference between a compliment and flirting?"]}
]

model = mix_models_with_data(
    model_names_or_paths=["BAAI/bge-base-en-v1.5", "Shitao/bge-hotpotqa", "Shitao/bge-quora"], 
    model_type='encoder', 
    example_ata=example_data,
    temperature=5.0,
    max_input_length=512,
    neg_number=2)
```
**由于这个[repo](https://huggingface.co/Shitao)中只有9个`bge-*`模型，当您的任务与所有9个微调任务不同时，性能可能不会令人满意。
您可以在更多任务上微调基础模型并将它们合并，以在您的任务上实现更好的性能。**

### 5. 加载您的模型
微调BGE模型后，您可以像[这里](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding#usage)一样轻松加载它。

如果您在微调时为超参数`--query_instruction_for_retrieval`设置了不同的值，请替换`query_instruction_for_retrieval`。

### 6. 评估模型
我们提供了[一个简单的脚本](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding/finetune/eval_msmarco.py) 来评估模型在MSMARCO上的性能，这是一个广泛使用的检索基准。

首先，安装`faiss`，这是一个流行的近似最近邻搜索库：
```bash
conda install -c conda-forge faiss-gpu
```

接下来，您可以查看[msmarco语料库](https://huggingface.co/datasets/namespace-Pt/msmarco-corpus)和[评估查询](https://huggingface.co/datasets/namespace-Pt/msmarco)的数据格式。

最后，运行以下命令：

```bash
python -m FlagEmbedding.baai_general_embedding.finetune.eval_msmarco \
--encoder BAAI/bge-base-en-v1.5 \
--fp16 \
--add_instruction \
--k 100
```

**一些重要的参数：**
- `encoder`: 指定编码器模型，可以是huggingface上的模型或本地模型。
- `fp16`: 使用半精度进行推理。
- `add_instruction`: 添加检索指令（`Represent this sentence for searching relevant passages: `）。
- `k`: 指定为每个查询检索的最近邻数量。

结果应该类似于：
```python
{
    'MRR@1': 0.2330945558739255, 
    'MRR@10': 0.35786976395142633, 
    'MRR@100': 0.3692618036917553, 
    'Recall@1': 0.22606255969436478, 
    'Recall@10': 0.6412965616045848, 
    'Recall@100': 0.9012774594078318
}
```

脚本的工作流程简要概述：
1. 通过[DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)在所有可用的GPU上加载模型。
2. 在`faiss` Flat索引中编码语料库并将嵌入卸载。默认情况下，`faiss`也会在所有可用的GPU上转储索引。
3. 对查询进行编码并为每个查询搜索`100`个最近邻。
4. 计算Recall和MRR指标。
