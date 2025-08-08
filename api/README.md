# 时尚多模态检索API

这是一个基于FastAPI的时尚多模态检索API，支持图片检索、文本检索和混合检索功能。

## 功能特性

- 🔍 **文本检索**: 根据文本描述检索相似时尚商品
- 🖼️ **图片检索**: 根据上传的图片检索相似时尚商品  
- 🔄 **混合检索**: 结合文本和图片进行加权混合检索
- 🎯 **过滤功能**: 支持按性别、产品类型等条件过滤
- 📊 **健康检查**: 提供API健康状态监控

## 安装依赖

```bash
pip install -r ../requirements.txt
```

## 启动服务器

```bash
# 方法1: 直接运行main.py
cd api
python main.py

# 方法2: 使用启动脚本
python start_server.py
```

服务器将在 `http://localhost:8000` 启动

## API文档

启动服务器后，可以访问以下地址查看交互式API文档：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API端点

### 1. 健康检查
```
GET /api/health
```

### 2. 文本检索
```
POST /api/search/text
```

**参数:**
- `query` (str, 必需): 搜索文本
- `top_k` (int, 可选): 返回结果数量，默认5
- `gender` (str, 可选): 性别过滤 (men/women)
- `product_type` (str, 可选): 产品类型过滤

**示例:**
```bash
curl -X POST "http://localhost:8000/api/search/text" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=red dress&top_k=5&gender=women&product_type=dress"
```

### 3. 图片检索
```
POST /api/search/image
```

**参数:**
- `file` (file, 必需): 上传的图片文件
- `top_k` (int, 可选): 返回结果数量，默认5
- `gender` (str, 可选): 性别过滤
- `product_type` (str, 可选): 产品类型过滤

**示例:**
```bash
curl -X POST "http://localhost:8000/api/search/image" \
  -F "file=@image.jpg" \
  -F "top_k=5" \
  -F "gender=men" \
  -F "product_type=jeans"
```

### 4. 混合检索
```
POST /api/search/hybrid
```

**参数:**
- `text_query` (str, 必需): 文本查询
- `file` (file, 必需): 上传的图片文件
- `alpha` (float, 可选): 文本权重 (0-1)，默认0.5
- `top_k` (int, 可选): 返回结果数量，默认5
- `gender` (str, 可选): 性别过滤
- `product_type` (str, 可选): 产品类型过滤

**示例:**
```bash
curl -X POST "http://localhost:8000/api/search/hybrid" \
  -F "text_query=blue jeans" \
  -F "file=@image.jpg" \
  -F "alpha=0.6" \
  -F "top_k=5" \
  -F "gender=men"
```

## 响应格式

所有API都返回JSON格式的响应：

```json
{
  "success": true,
  "query": "red dress",
  "total_results": 5,
  "results": [
    {
      "id": "123",
      "caption": "red summer dress",
      "product_type": "dress",
      "gender": "women",
      "score": 0.85
    }
  ]
}
```

## 测试API

使用提供的测试客户端：

```bash
python test_client.py
```

## 配置说明

在 `main.py` 中可以修改以下配置：

- `MILVUS_URI`: Milvus数据库连接地址
- `MILVUS_TOKEN`: Milvus访问令牌
- `MODEL_PATH`: 模型权重文件路径
- `PROCESSOR_PATH`: 模型处理器路径
- `COLLECTION_NAME`: Milvus集合名称

## 错误处理

API会返回适当的HTTP状态码和错误信息：

- `200`: 成功
- `400`: 请求参数错误
- `500`: 服务器内部错误

## 注意事项

1. 确保模型文件路径正确
2. 确保Milvus数据库连接正常
3. 上传的图片文件必须是有效的图片格式
4. 权重参数alpha必须在0-1之间
5. 临时上传的文件会自动清理

## 开发说明

- API使用异步处理提高性能
- 支持CORS跨域请求
- 自动生成API文档
- 包含完整的错误处理
- 支持文件上传和表单数据处理 