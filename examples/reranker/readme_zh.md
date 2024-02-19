 # Finetune 交叉编码器
在这个例子中，我们将展示如何使用您的数据对交叉编码器重排器进行微调。

## 1. 安装
* **使用 pip**
```bash
pip install -U FlagEmbedding
```

* **从源代码**
```bash
git clone https://github.com/FlagOpen/FlagEmbedding.git 
cd FlagEmbedding
pip install .
```
为了开发，以可编辑模式安装：
```bash
pip install -e .
```

## 2. 数据格式

重排器的数据格式与[嵌入微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#data-format)相同。 
此外，我们强烈建议[挖掘硬负样本](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#hard-negatives)来微调重排器。

## 3. 训练

```bash
torchrun --nproc_per_node {gpu数量} \
-m FlagEmbedding.reranker.run \
--output_dir {保存模型的路径} \
--model_name_or_path BAAI/bge-reranker-base \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size {训练中的批次大小；对于玩具数据设置为1} \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--train_group_size 16 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 10 
```

**一些重要的参数**:
- `per_device_train_batch_size`: 训练中的批次大小。
- `train_group_size`: 训练中每个查询的正样本和负样本数量。总是有一个正样本，所以这个参数将控制负样本的数量 (#negatives=train_group_size-1)。
请注意，负样本的数量不应大于数据中负样本的数量 `"neg":List[str]`。
除了这个组中的负样本外，批次中的负样本也会在微调中使用。

更多训练参数请参考 [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) 

### 4. 通过 [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail) 合并模型

更多细节请参考 [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)。

微调基础的 bge 模型可以提高其在目标任务上的性能，
但可能会导致模型在目标域之外的一般能力严重退化（例如，在 c-mteb 任务上性能降低）。
通过合并微调模型和基础模型，LM-Cocktail 可以在保持其他不相关任务性能的同时显著提高下游任务的性能。

```python
from LM_Cocktail import mix_models, mix_models_with_data

# 混合微调模型和基础模型；然后保存到输出路径: ./mixed_model_1
model = mix_models(
    model_names_or_paths=["BAAI/bge-reranker-base", "你的微调模型路径"], 
    model_type='reranker', 
    weights=[0.5, 0.5],  # 你可以改变权重以获得更好的权衡。
    output_path='./mixed_model_1')
```

### 5. 加载你的模型

#### 使用 FlagEmbedding

```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True) # 使用 fp16 可以加速计算

score = reranker.compute_score(['查询', '段落'])
print(score)

scores = reranker.compute_score([['什么是熊猫？', '嗨'], ['什么是熊猫？', '大熊猫（Ailuropoda melanoleuca），有时被称为熊猫熊或简称熊猫，是一种仅在中国发现的熊科动物。']])
print(scores)
```

#### 使用 Huggingface transformers

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
model.eval()

pairs = [['什么是熊猫？', '嗨'], ['什么是熊猫？', '大熊猫（Ailuropoda melanoleuca），有时被称为熊猫熊或简称熊猫，是一种仅在中国发现的熊科动物。']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```
