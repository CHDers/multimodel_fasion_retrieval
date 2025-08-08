# 时尚多模态检索模型向量化工具

这个工具包提供了加载已训练的时尚多模态检索模型，并对图片和文本进行向量化的功能。

## 文件结构

```
data_embedding/
├── model_vectorization.py    # 主要的向量化模型类
├── example_usage.py          # 使用示例
├── README.md                 # 本文档
├── create_schema.ipynb       # 数据库schema创建
└── dataset_embedding.ipynb   # 数据集嵌入分析
```

## 主要功能

### 1. FashionEmbeddingModel 类

核心的向量化模型类，提供以下功能：

- **单个图片向量化**: `encode_image(image)` - 将单张图片转换为256维向量
- **单个文本向量化**: `encode_text(text)` - 将文本转换为256维向量
- **批量图片向量化**: `encode_batch_images(image_paths, batch_size)` - 批量处理图片
- **批量文本向量化**: `encode_batch_texts(texts, batch_size)` - 批量处理文本
- **相似度计算**: `compute_similarity(image_embedding, text_embedding)` - 计算图片和文本的相似度
- **图片检索**: `find_similar_images(query_text, image_embeddings, image_paths, top_k)` - 根据文本查找相似图片
- **文本检索**: `find_similar_texts(query_image, text_embeddings, texts, top_k)` - 根据图片查找相似文本

### 2. 数据集嵌入生成

`create_dataset_embeddings()` 函数可以为整个数据集生成嵌入向量并保存到文件中，包括：

- 图片嵌入向量 (`image_embeddings.npy`)
- 文本嵌入向量 (`text_embeddings.npy`)
- 元数据信息 (`metadata.pkl`)

## 使用方法

### 1. 基本初始化

```python
from model_vectorization import FashionEmbeddingModel

# 初始化模型
model_path = "../../training_results_20250705_165112/model_weights/best_model.pth"
processor_path = "/root/autodl-fs/blip-image-captioning-base/"

model = FashionEmbeddingModel(model_path, processor_path)
```

### 2. 单个图片和文本向量化

```python
# 图片向量化
image_embedding = model.encode_image("path/to/image.jpg")
print(f"图片嵌入向量维度: {image_embedding.shape}")  # (256,)

# 文本向量化
text_embedding = model.encode_text("red dress for women")
print(f"文本嵌入向量维度: {text_embedding.shape}")  # (256,)

# 计算相似度
similarity = model.compute_similarity(image_embedding, text_embedding)
print(f"相似度: {similarity:.4f}")
```

### 3. 批量处理

```python
# 批量处理图片
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
image_embeddings = model.encode_batch_images(image_paths, batch_size=16)
print(f"批量图片嵌入形状: {image_embeddings.shape}")  # (3, 256)

# 批量处理文本
texts = ["red dress", "blue jeans", "black jacket"]
text_embeddings = model.encode_batch_texts(texts, batch_size=32)
print(f"批量文本嵌入形状: {text_embeddings.shape}")  # (3, 256)
```

### 4. 相似度搜索

```python
# 文本到图片检索
query_text = "black leather jacket"
similar_images = model.find_similar_images(
    query_text, image_embeddings, image_paths, top_k=5
)

for image_path, score in similar_images:
    print(f"{image_path}: {score:.4f}")

# 图片到文本检索
query_image = "path/to/query_image.jpg"
similar_texts = model.find_similar_texts(
    query_image, text_embeddings, texts, top_k=5
)

for text, score in similar_texts:
    print(f"{text}: {score:.4f}")
```

### 5. 生成整个数据集的嵌入

```python
from model_vectorization import create_dataset_embeddings

# 生成数据集嵌入
csv_path = "../../dataset/DeepFashion-MultiModal/labels_front.csv"
image_embeddings, text_embeddings, metadata = create_dataset_embeddings(
    csv_path, model_path, processor_path
)

# 嵌入会保存到 embeddings_output/ 目录
# - image_embeddings.npy: 图片嵌入矩阵
# - text_embeddings.npy: 文本嵌入矩阵
# - metadata.pkl: 元数据信息
```

### 6. 加载预计算的嵌入

```python
import numpy as np
import pickle

# 加载预计算的嵌入
image_embeddings = np.load("embeddings_output/image_embeddings.npy")
text_embeddings = np.load("embeddings_output/text_embeddings.npy")

with open("embeddings_output/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# 使用预计算的嵌入进行搜索
query_text = "red dress"
similar_images = model.find_similar_images(
    query_text, image_embeddings, metadata['image_paths'], top_k=3
)
```

## 运行示例

### 1. 生成数据集嵌入

```bash
cd data_embedding
python model_vectorization.py
```

这会：
- 加载训练好的模型
- 处理整个数据集
- 生成并保存所有图片和文本的嵌入向量
- 输出到 `embeddings_output/` 目录

### 2. 运行使用示例

```bash
cd data_embedding
python example_usage.py
```

这会演示：
- 单个图片和文本的向量化
- 批量处理
- 相似度计算
- 各种搜索功能

## 技术细节

### 模型架构
- 基于 BLIP (Bootstrapping Language-Image Pre-training) 
- 使用 LoRA (Low-Rank Adaptation) 进行微调
- 图片和文本都被编码为256维向量
- 使用余弦相似度进行相似度计算

### LoRA 配置
- Rank: 16
- Alpha: 32
- 目标模块: ["query", "value", "vision_proj", "text_proj"]

### 性能优化
- 支持批量处理以提高效率
- 使用GPU加速（如果可用）
- 自动处理图片加载错误

## 注意事项

1. **模型路径**: 确保模型权重文件 `best_model.pth` 存在
2. **处理器路径**: 确保BLIP处理器路径正确
3. **内存使用**: 批量处理时注意内存使用，可以调整batch_size
4. **GPU**: 建议使用GPU以获得更好的性能
5. **图片格式**: 支持PIL可以处理的所有图片格式

## 错误处理

- 如果图片加载失败，会自动创建黑色占位符图片
- 所有向量都会进行归一化处理
- 自动处理设备迁移（CPU/GPU）

## 扩展功能

可以根据需要扩展以下功能：
- 添加更多的相似度计算方法
- 支持更多的图片预处理选项
- 添加向量索引以加速大规模搜索
- 支持更多的输出格式 