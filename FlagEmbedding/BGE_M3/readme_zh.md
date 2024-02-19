 # BGE-M3 ([论文](https://arxiv.org/pdf/2402.03216.pdf), [代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3))
在这个项目中，我们介绍了BGE-M3，它以其多功能性、多语言性和多粒度性而著称。
- 多功能性：它能够同时执行嵌入模型的三种常见检索功能：密集检索、多向量检索和稀疏检索。
- 多语言性：它支持超过100种工作语言。
- 多粒度性：它能够处理不同粒度的输入，从短句到长达8192个令牌的长文档。

**在RAG中的检索流程建议：**
我们建议使用以下流程：混合检索 + 重排序。
- 混合检索利用了各种方法的优势，提供了更高的准确性和更强的泛化能力。
一个经典的例子是：同时使用嵌入检索和BM25算法。
现在，你可以尝试使用BGE-M3，它支持嵌入和稀疏检索。
这允许你在生成密集嵌入时无需额外成本即可获得令牌权重（类似于BM25）。
- 作为跨编码器模型，重排序器比双编码器嵌入模型展示了更高的准确性。
在检索后利用重排序模型（例如，[bge-reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker), [cohere-reranker](https://txt.cohere.com/rerank/)）可以进一步过滤选定的文本。

## 新闻：
- 2/6/2024: 我们发布了[MLDR](https://huggingface.co/datasets/Shitao/MLDR)（一个覆盖13种语言的长文档检索数据集）和[评估流程](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB/MLDR)。
- 2/1/2024: **感谢Vespa的优秀工具。** 你可以轻松地使用BGE-M3的多种模式，按照这个[笔记本](https://github.com/vespa-engine/pyvespa/blob/master/docs/sphinx/source/examples/mother-of-all-embedding-models-cloud.ipynb)操作。

## 规格

- 模型

| 模型名称 | 维度 | 序列长度 | 介绍 |
|:----:|:---:|:---:|:---:|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)  | 1024 | 8192 | 多语言；从bge-m3-unsupervised统一微调（密集、稀疏和colbert）|
| [BAAI/bge-m3-unsupervised](https://huggingface.co/BAAI/bge-m3-unsupervised)  | 1024 | 8192 | 多语言；从bge-m3-retromae进行对比学习 |
| [BAAI/bge-m3-retromae](https://huggingface.co/BAAI/bge-m3-retromae)  | -- | 8192 | 多语言；将[xlm-roberta](https://huggingface.co/FacebookAI/xlm-roberta-large) 的最大长度扩展到8192，并通过[retromae](https://github.com/staoxiao/RetroMAE)进一步预训练 |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)  | 1024 | 512 | 英语模型 |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)  | 768 | 512 | 英语模型 |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)  | 384 | 512 | 英语模型 |

- 数据集

| 数据集 | 介绍 |
|:----:|:---:|
| [MLDR](https://huggingface.co/datasets/Shitao/MLDR)  | 文档检索数据集，覆盖13种语言 |

## FAQ

**1. 不同检索方法的介绍**

- 密集检索：将文本映射为单个嵌入，例如，[DPR](https://arxiv.org/abs/2004.04906), [BGE-v1.5](https://github.com/FlagOpen/FlagEmbedding) 
- 稀疏检索（词汇匹配）：一个大小等于词汇表的向量，大多数位置设置为零，只为文本中出现的令牌计算权重。例如，BM25, [unicoil](https://arxiv.org/pdf/2106.14807.pdf), 和 [splade](https://arxiv.org/abs/2107.05720) 
- 多向量检索：使用多个向量来表示文本，例如，[ColBERT](https://arxiv.org/abs/2004.12832)。

**2. 与BGE-v1.5和其他单语言模型的比较**

BGE-M3是一个多语言模型，其在单语言嵌入检索方面的能力可能不会超过专门为单一语言设计的模型。
然而，我们仍然推荐尝试BGE-M3，因为它的多功能性（支持多种语言和长文本）。
此外，它可以同时生成多个表示，并且一起使用它们可以提高准确性和泛化能力，
与大多数现有模型不同，它们只能执行密集检索。

在开源社区中，有许多优秀的模型（例如，jina-embedding, colbert, e5等），
用户可以根据实际考虑（是否需要多语言或跨语言支持，以及是否需要处理长文本）选择适合自己需求的模型。

**3. 如何在其他项目中使用BGE-M3？**

对于嵌入检索，你可以像使用BGE一样使用BGE-M3模型。
唯一的区别是，BGE-M3模型不再需要在查询中添加指令。
对于稀疏检索方法，大多数开源库目前不支持直接利用BGE-M3模型。
欢迎社区的贡献。

在我们的实验中，我们使用[Pyserini](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB/MLDR#hybrid-retrieval-dense--sparse) 和 Faiss 进行混合检索。
**现在你可以尝试在[Vespa](https://github.com/vespa-engine/pyvespa/blob/master/docs/sphinx/source/examples/mother-of-all-embedding-models-cloud.ipynb)中尝试BGE-M3的混合模式。**
**感谢@jobergum。**

**4. 如何微调bge-M3模型？**

你可以按照这个[例子](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)来微调密集嵌入。
我们的代码和数据用于统一微调（密集、稀疏和多向量）将在不久后发布。


## 使用

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

### 生成文本嵌入

- 密集嵌入
```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True) # 设置use_fp16为True可以加速计算，但性能略有下降

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_1 = model.encode(sentences_1, 
                            batch_size=12, 
                            max_length=8192, # 如果你不需要这么长的长度，可以设置一个较小的值来加速编码过程。
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]
```
你也可以使用sentence-transformers和huggingface transformers来生成密集嵌入。
有关详细信息，请参阅[baai_general_embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding#usage)。

- 稀疏嵌入（词汇权重）
```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) # 设置use_fp16为True可以加速计算，但性能略有下降
 ```python
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
output_2 = model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=False)

# 你可以查看每个令牌的权重：
print(model.convert_id_to_token(output_1['lexical_weights']))
# [{'What': 0.08356, 'is': 0.0814, 'B': 0.1296, 'GE': 0.252, 'M': 0.1702, '3': 0.2695, '?': 0.04092}, 
#  {'De': 0.05005, 'fin': 0.1368, 'ation': 0.04498, 'of': 0.0633, 'BM': 0.2515, '25': 0.3335}]

# 通过词汇匹配计算分数
lexical_scores = model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][0])
print(lexical_scores)
# 0.19554901123046875

print(model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_1['lexical_weights'][1]))
# 0.0
```

- 多向量（ColBERT）
```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) 

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=True)
output_2 = model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=True)

print(model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][0]))
print(model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][1]))
# 0.7797
# 0.4620
```

### 计算文本对的分数
输入一组文本对，你可以得到不同方法计算的分数。
```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) 

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]

print(model.compute_score(sentence_pairs, 
                          max_passage_length=128, # 更小的最大长度会导致更低的延迟
                          weights_for_different_modes=[0.4, 0.2, 0.4])) # weights_for_different_modes(w)用于加权求和：w[0]*dense_score + w[1]*sparse_score + w[2]*colbert_score

# {
#   'colbert': [0.7796499729156494, 0.4621465802192688, 0.4523794651031494, 0.7898575067520142], 
#   'sparse': [0.195556640625, 0.00879669189453125, 0.0, 0.1802978515625], 
#   'dense': [0.6259765625, 0.347412109375, 0.349853515625, 0.67822265625], 
#   'sparse+dense': [0.482503205537796, 0.23454029858112335, 0.2332356721162796, 0.5122477412223816], 
#   'colbert+sparse+dense': [0.6013619303703308, 0.3255828022956848, 0.32089319825172424, 0.6232916116714478]
# }
```

## 评估

我们将BGE-M3与一些流行的方法进行了比较，包括BM25、OpenAI嵌入等。

- 多语言（Miracl数据集）

![avatar](./imgs/miracl.jpg)

- 跨语言（MKQA数据集）

![avatar](./imgs/mkqa.jpg)

- 长文档检索
  - MLDR:
  ![avatar](./imgs/long.jpg)
  请注意，[MLDR](https://huggingface.co/datasets/Shitao/MLDR) 是我们通过LLM构建的文档检索数据集，
  覆盖13种语言，包括测试集、验证集和训练集。
  我们利用MLDR的训练集来增强模型的长文档检索能力。
  因此，与`Dense w.o.long`（未进行长文档数据集微调的基线）进行比较更为公平。
  此外，这个长文档检索数据集将开源，以解决目前缺乏开源多语言长文本检索数据集的问题。
  我们相信这些数据将对开源社区在训练文档检索模型方面有所帮助。

  - NarritiveQA:  
  ![avatar](./imgs/nqa.jpg)

- 与BM25的比较

我们使用Pyserini实现了BM25，测试结果可以通过这个[脚本](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB/MLDR#bm25-baseline)复现。
我们测试了两种不同的分词器：一种使用Lucene Analyzer，另一种使用与M3相同的分词器（即xlm-roberta的分词器）。
结果表明，BM25仍然是一个有竞争力的基线模型，
特别是在长文档检索方面。

![avatar](./imgs/bm25.jpg)

## 训练
- 自我知识蒸馏：将不同检索模式的多个输出组合作为奖励信号，以增强单个模式的性能（特别是对于稀疏检索和多向量（ColBERT）检索）
- 高效批处理：在长文本微调时提高效率。
小批量策略简单但有效，也可用于微调大型嵌入模型。
- MCLS：一种在没有足够资源进行长文本微调时提高性能的简单方法。

有关更多详细信息，请参阅我们的[报告](https://arxiv.org/pdf/2402.03216.pdf)。

**微调代码和数据集将在不久的将来开源。**

## 致谢

感谢开源数据集的作者，包括Miracl、MKQA、NarritiveQA等。
感谢开源库如[Tevatron](https://github.com/texttron/tevatron), [pyserial](https://github.com/pyserial/pyserial)。

## 引用

如果你发现这个仓库有用，请考虑给它一个星标：star，并引用

```
@misc{bge-m3,
      title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation}, 
      author={Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2402.03216},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
