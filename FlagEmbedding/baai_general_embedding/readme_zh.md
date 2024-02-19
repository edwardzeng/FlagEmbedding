 ```markdown
# 嵌入模型

## 常见问题

**由于使用不当导致的非常差的结果**

与其他使用均值池化的嵌入模型不同，BGE使用`[cls]`的最后一个隐藏状态作为句子嵌入：`sentence_embeddings = model_output[0][:, 0]`。
如果你使用均值池化，性能将显著下降。
因此，请确保使用正确的方法获取句子向量。你可以参照我们提供的使用方法。

**1. 如何微调bge嵌入模型？**

按照这个[示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) 准备数据并微调你的模型。
一些建议：
- 按照这个[示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#hard-negatives)挖掘硬负样本，这可以提高检索性能。
- 一般来说，较大的超参数`per_device_train_batch_size`会带来更好的性能。你可以通过启用`--fp16`、`--deepspeed df_config.json`（df_config.json可以参考[ds_config.json](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/ds_config.json)）、`--gradient_checkpointing`等来扩展它。
- 如果你想在微调你的数据时保持其他任务的性能，你可以使用[LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)来合并微调模型和原始bge模型。此外，如果你想在多个任务上进行微调，你也可以通过模型合并近似多任务学习，如[LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)。
- 如果你在你数据上预训练bge，预训练模型不能直接用于计算相似度，必须通过对比学习进行微调才能计算相似度。
- 如果微调模型的准确性仍然不高，建议使用/微调交叉编码器模型（bge-reranker）来重新排序top-k结果。硬负样本也是微调reranker所需的。

我们微调`bge-large-zh-v1.5`的方法：
微调数据集包括t2ranking、dulreader、mmarco、cmedqav2、mulit-cpr、nli-zh、ocmnli和cmnli。
对于t2ranking、dulreader和mmarco，我们挖掘硬负样本；
对于nli-zh、ocmnli和cmnli，我们使用标签等于0的对作为负样本；
对于cmedqav2和mulit-cpr，我们随机采样负样本。
微调设置：train_group_size=2, learning_rate=1e-5, max_epoch=5。
我们训练了两个模型：一个使用`--query_instruction_for_retrieval "为这个句子生成表示以用于检索相关文章："`微调，
另一个模型使用`--query_instruction_for_retrieval ""`微调，
然后我们将两个变体合并成一个模型，使最终模型既可以使用指令也可以不使用指令。

<details>
  <summary>2. 两个不相似句子之间的相似度分数高于0.5</summary>

  **建议使用bge v1.5，它缓解了相似度分布的问题。**

  由于我们通过对比学习微调模型，温度为0.01，
  当前BGE模型的相似度分布在区间[0.6, 1]。
  所以相似度分数大于0.5并不意味着两个句子相似。

  对于下游任务，如段落检索或语义相似度，
  重要的是分数的相对顺序，而不是绝对值。
  如果你需要根据相似度阈值过滤相似句子，
  请根据你的数据上的相似度分布选择适当的相似度阈值（例如0.8, 0.85, 或者甚至0.9）。

</details>

<details>
  <summary>3. 查询指令何时需要使用</summary>

  对于`bge-*-v1.5`，我们在不使用指令的情况下提高了检索能力。
  不使用指令相比使用指令在检索性能上只有轻微的下降。
  所以你可以为了方便在所有情况下不使用指令生成嵌入。

  对于使用短查询找到长相关文档的检索任务，
  建议为这些短查询添加指令。
  **决定是否为查询添加指令的最佳方法是选择在你的任务上表现更好的设置。**
  在所有情况下，文档/段落不需要添加指令。

</details>

## 使用

### 使用FlagEmbedding

安装：
```
git clone https://github.com/FlagOpen/FlagEmbedding.git 
cd FlagEmbedding
pip install -e .
```
或者：
```
pip install -U FlagEmbedding
```
 

```python
from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # 设置use_fp16为True可以加速计算，但性能略有下降
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# 对于s2p（短查询到长段落）检索任务，建议使用encode_queries()，它会自动为每个查询添加指令
# 检索任务中的语料库仍然可以使用encode()或encode_corpus()，因为它们不需要指令
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
```

关于`query_instruction_for_retrieval`参数的值，请参考[模型列表](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)。

默认情况下，FlagModel在编码时会使用所有可用的GPU。请设置`os.environ["CUDA_VISIBLE_DEVICES"]`来选择特定的GPU。
你也可以设置`os.environ["CUDA_VISIBLE_DEVICES"]=""`来使所有GPU不可用。

### 使用Sentence-Transformers

你也可以使用`bge`模型与[sentence-transformers](https://www.SBERT.net)：

```
pip install -U sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```
对于s2p（短查询到长段落）检索任务，
每个短查询应该以指令开头（指令请参考[模型列表](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)）。
但语料库不需要指令。

```python
from sentence_transformers import SentenceTransformer
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
instruction = "为这个句子生成表示以用于检索相关文章："

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```

### 使用Langchain

你可以这样在langchain中使用`bge`：

```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # 设置为True以计算余弦相似度
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："
```

### 使用HuggingFace Transformers

使用transformers包，你可以这样使用模型：首先，你将输入通过transformer模型，然后选择第一个标记（即[CLS]）的最后一个隐藏状态作为句子嵌入。

```python
from transformers import AutoTokenizer, AutoModel
import torch
# 我们想要句子嵌入的句子
sentences = ["样例数据-1", "样例数据-2"]

# 从HuggingFace Hub加载模型
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

# 对句子进行编码
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# 对于s2p（短查询到长段落）检索任务，查询应添加指令（对语料库不需要添加指令）
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# 计算标记嵌入
with torch.no_grad():
    model_output = model(**encoded_input)
    # 执行池化。在这种情况下，使用cls池化。
    sentence_embeddings = model_output[0][:, 0]
# 归一化嵌入
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("句子嵌入:", sentence_embeddings)
```

## 评估

`baai-general-embedding`模型在MTEB和C-MTEB排行榜上取得了**最先进的性能**！
更多详情和评估工具请参考我们的[脚本](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md)
- **MTEB**:

| 模型名称 | 维度 | 序列长度 | 平均（56） | 检索（15） | 聚类（11） | 对分类（3） | 重排序（4） | STS（10） | 总结（1） | 分类（12） |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)  | 1024 | 512 | **64.23** | **54.29** | 46.08 | 87.12 | 60.03 | 83.11 | 31.61 | 75.97 |  
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)  | 768 | 512 | 63.55 | 53.25 | 45.77 | 86.55 | 58.86 | 82.4 | 31.07 | 75.53 |  
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)  | 384 | 512 | 62.17 | 51.68 | 43.82 | 84.92 | 58.36 | 81.59 | 30.12 | 74.14 |  
| [bge-large-en](https://huggingface.co/BAAI/bge-large-en)  | 1024 | 512 | 63.98 | 53.9 | 46.98 | 85.8 | 59.48 | 81.56 | 32.06 | 76.21 | 
| [bge-base-en](https://huggingface.co/BAAI/bge-base-en)  | 768 | 512 | 63.36 | 53.0 | 46.32 | 85.86 | 58.7 | 81.84 | 29.27 | 75.27 | 
| [gte-large](https://huggingface.co/thenlper/gte-large)  | 1024 | 512 | 63.13 | 52.22 | 46.84 | 85.00 | 59.13 | 83.35 | 31.66 | 73.33 |
| [gte-base](https://huggingface.co/thenlper/gte-base)  	| 768 | 512 | 62.39 | 51.14 | 46.2 | 84.57 | 58.61 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2)  | 1024| 512 | 62.25 | 50.56 | 44.49 | 86.03 | 56.61 | 82.05 | 30.19 | 75.24 |
| [bge-small-en](https://huggingface.co/BAAI/bge-small-en)  | 384 | 512 | 62.11 | 51.82 | 44.31 | 83.78 | 57.97 | 80.72 | 30.53 | 74.37 |  
| [instructor-xl](https://huggingface.co/hkunlp/instructor-xl)  | 768 | 512 | 61.79 | 49.26 | 44.74 | 86.62 | 57.29 | 83.06 | 32.32 | 61.79 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)  | 768 | 512 | 61.5 | 50.29 | 43.80 | 85.73 | 55.91 | 81.05 | 30.28 | 73.84 |
| [gte-small](https://huggingface.co/thenlper/gte-small)  | 384 | 512 | 61.36 | 49.46 | 44.89 | 83.54 | 57.7 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings)  | 1536 | 8192 | 60.99 | 49.25 | 45.9 | 84.89 | 56.32 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2)  | 384 | 512 | 59.93 | 49.04 | 39.92 | 84.67 | 54.32 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl)  | 768 | 512 | 59.51 | 42.24 | 43.72 | 85.06 | 56.42 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)  	| 768 | 514 	| 57.78 | 43.81 | 43.69 | 83.04 | 59.36 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco)  	| 4096 | 2048 | 57.59 | 48.22 | 38.93 | 81.9 | 55.65 | 77.74 | 33.6 | 66.19 |

- **C-MTEB**:
我们创建了C-MTEB基准，用于中文文本嵌入，由6个任务的31个数据集组成。
请参考[C_MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md) 以获取详细介绍。

| 模型 | 嵌入维度 | 平均 | 检索 | STS | 对分类 | 分类 | 重排序 | 聚类 |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|
