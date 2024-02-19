 ### Q&A 示例

向量数据库可以帮助大型语言模型（LLMs）访问外部知识。
你可以加载 baai-general-embedding 作为编码器来生成向量。
以下是一个使用中文维基百科构建能够回答问题的机器人的示例。

使用旗标嵌入和大型语言模型（LLM）进行问答对话场景的描述：

1. **数据预处理和索引：**
   - 下载中文维基百科数据集。
   - 使用旗标嵌入对中文维基百科文本进行编码。
   - 使用 BM25 构建索引。
2. **使用大型语言模型（LLM）增强查询：**
   - 利用大型语言模型（LLM）根据聊天历史增强和丰富原始用户查询。
   - LLM 可以执行文本补全和改写等任务，使查询更加健壮和全面。
3. **文档检索：**
   - 使用 BM25 根据新增强的查询从本地存储的中文维基数据集检索 top-n 文档。
4. **嵌入检索：**
   - 对检索到的 top-n 文档使用暴力搜索进行嵌入检索，以获取 top-k 文档。
5. **使用语言模型（LLM）检索答案：**
   - 将问题、检索到的 top-k 文档和聊天历史呈现给大型语言模型（LLM）。
   - LLM 可以利用其对语言和上下文的理解，为用户提供准确和全面的答案。

遵循这些步骤，问答系统可以利用旗标嵌入、BM25 索引和大型语言模型来提高系统的准确性和智能性。这些技术的整合可以创建一个更复杂、更可靠的问答系统，为用户提供全面的信息，有效回答他们的问题。

### 安装

```shell
sudo apt install default-jdk
pip install -r requirements.txt
conda install -c anaconda openjdk
```

### 准备数据

```shell
python pre_process.py --data_path ./data
```

这个脚本将下载数据集（中文维基百科），构建 BM25 索引，推理嵌入，然后将它们保存到 `data_path`。

## Q&A 使用

### 直接运行

```shell
export OPENAI_API_KEY=...
python run.py --data_path ./data
```

这个脚本将构建一个问答对话场景。

### 快速开始

```python
# encoding=gbk
from tool import LocalDatasetLoader, BMVectorIndex, Agent
loader = LocalDatasetLoader(data_path="./data/dataset",
                            embedding_path="./data/emb/data.npy")
index = BMVectorIndex(model_path="BAAI/bge-large-zh",
                      bm_index_path="./data/index",
                      data_loader=loader)
agent = Agent(index)
question = "上次有人登月是什么时候"
agent.Answer(question, RANKING=1000, TOP_N=5, verbose=False)
```
