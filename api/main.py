#!/usr/bin/env python3
"""
时尚多模态检索API
提供图片检索、文本检索和混合检索功能
"""

import os
import sys
import uuid
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pymilvus import MilvusClient
import uvicorn

# 添加项目路径
sys.path.append('/root/autodl-tmp/multimodel_fasion_retrieval')
from data_embedding.model_vectorization import FashionEmbeddingModel
from vector_db.retrieval_util import (
    search_text_embedding, 
    search_image_embedding, 
    hybrid_search_weighted
)

# 创建FastAPI应用
app = FastAPI(
    title="时尚多模态检索API",
    description="提供图片检索、文本检索和混合检索功能",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
client = None
model = None

# 配置
MILVUS_URI = "https://in03-338d6950a393b2c.serverless.gcp-us-west1.cloud.zilliz.com"
MILVUS_TOKEN = "xxxx"
# 模型路径
MODEL_PATH = "/root/autodl-fs/training_results_20250705_165112/model_weights/best_model.pth"
PROCESSOR_PATH = "/root/autodl-fs/blip-image-captioning-base/"
# 向量数据库
COLLECTION_NAME = "fasion_multimodel_embedding"
# 上传目录
UPLOAD_DIR = "uploads"
# 图片目录
IMAGES_DIR = "../../dataset/DeepFashion-MultiModal/selected_images"

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 挂载静态文件目录
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型和数据库连接"""
    global client, model
    
    try:
        # 初始化Milvus客户端
        client = MilvusClient(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN
        )
        print("✅ Milvus连接成功")
        
        # 初始化模型
        model = FashionEmbeddingModel(MODEL_PATH, PROCESSOR_PATH)
        print("✅ 模型加载成功")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        raise e

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "时尚多模态检索API",
        "version": "1.0.0",
        "endpoints": {
            "text_search": "/api/search/text",
            "image_search": "/api/search/image", 
            "hybrid_search": "/api/search/hybrid",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查Milvus连接
        if client is None:
            raise Exception("Milvus客户端未初始化")
        
        # 检查模型
        if model is None:
            raise Exception("模型未初始化")
        
        return {
            "status": "healthy",
            "milvus_connected": True,
            "model_loaded": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

@app.post("/api/search/text")
async def text_search(
    query: str = Form(..., description="搜索文本"),
    top_k: int = Form(5, description="返回结果数量"),
    gender: Optional[str] = Form(None, description="性别过滤"),
    product_type: Optional[str] = Form(None, description="产品类型过滤")
):
    """
    文本检索API
    
    Args:
        query: 搜索文本
        top_k: 返回结果数量
        gender: 性别过滤条件
        product_type: 产品类型过滤条件
    
    Returns:
        检索结果列表
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="模型未初始化")
        
        # 编码文本
        query_emb = model.encode_text(query)
        
        # 执行检索
        results = search_text_embedding(
            client=client,
            query_text_emb=query_emb,
            top_k=top_k,
            collection_name=COLLECTION_NAME,
            gender=gender,
            product_type=product_type
        )
        
        # 格式化结果
        formatted_results = []
        for result in results:
            image_id = result.get("pk")
            image_url = f"/images/{image_id}.jpg"
            formatted_results.append({
                "id": image_id,
                "caption": result.get("caption"),
                "product_type": result.get("product_type"),
                "gender": result.get("gender"),
                "score": result.get("distance", 0.0),
                "image_url": image_url
            })
        
        return {
            "success": True,
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本检索失败: {str(e)}")

@app.post("/api/search/image")
async def image_search(
    file: UploadFile = File(..., description="上传的图片文件"),
    top_k: int = Form(5, description="返回结果数量"),
    gender: Optional[str] = Form(None, description="性别过滤"),
    product_type: Optional[str] = Form(None, description="产品类型过滤")
):
    """
    图片检索API
    
    Args:
        file: 上传的图片文件
        top_k: 返回结果数量
        gender: 性别过滤条件
        product_type: 产品类型过滤条件
    
    Returns:
        检索结果列表
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="模型未初始化")
        
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 保存上传的文件
        file_extension = Path(file.filename).suffix
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # 编码图片
            query_emb = model.encode_image(temp_file_path)
            
            # 执行检索
            results = search_image_embedding(
                client=client,
                query_img_emb=query_emb,
                top_k=top_k,
                collection_name=COLLECTION_NAME,
                gender=gender,
                product_type=product_type
            )
            
            # 格式化结果
            formatted_results = []
            for result in results:
                image_id = result.get("pk")
                image_url = f"/images/{image_id}.jpg"
                formatted_results.append({
                    "id": image_id,
                    "caption": result.get("caption"),
                    "product_type": result.get("product_type"),
                    "gender": result.get("gender"),
                    "score": result.get("distance", 0.0),
                    "image_url": image_url
                })
            
            return {
                "success": True,
                "filename": file.filename,
                "total_results": len(formatted_results),
                "results": formatted_results
            }
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片检索失败: {str(e)}")

@app.post("/api/search/hybrid")
async def hybrid_search(
    text_query: str = Form(..., description="文本查询"),
    file: UploadFile = File(..., description="上传的图片文件"),
    alpha: float = Form(0.5, description="文本权重 (0-1)"),
    top_k: int = Form(5, description="返回结果数量"),
    gender: Optional[str] = Form(None, description="性别过滤"),
    product_type: Optional[str] = Form(None, description="产品类型过滤")
):
    """
    混合检索API - 结合文本和图片进行检索
    
    Args:
        text_query: 文本查询
        file: 上传的图片文件
        alpha: 文本权重 (0-1)，图片权重为 (1-alpha)
        top_k: 返回结果数量
        gender: 性别过滤条件
        product_type: 产品类型过滤条件
    
    Returns:
        检索结果列表
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="模型未初始化")
        
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 验证权重参数
        if not 0 <= alpha <= 1:
            raise HTTPException(status_code=400, detail="权重必须在0-1之间")
        
        # 保存上传的文件
        file_extension = Path(file.filename).suffix
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # 编码文本和图片
            text_emb = model.encode_text(text_query)
            img_emb = model.encode_image(temp_file_path)
            
            # 执行混合检索
            results = hybrid_search_weighted(
                client=client,
                query_img_emb=img_emb,
                query_text_emb=text_emb,
                alpha=alpha,
                top_k=top_k,
                collection_name=COLLECTION_NAME,
                gender=gender,
                product_type=product_type
            )
            
            # 格式化结果
            formatted_results = []
            for result in results:
                image_id = result.get("pk")
                image_url = f"/images/{image_id}.jpg"
                formatted_results.append({
                    "id": image_id,
                    "caption": result.get("caption"),
                    "product_type": result.get("product_type"),
                    "gender": result.get("gender"),
                    "score": result.get("distance", 0.0),
                    "image_url": image_url
                })
            
            return {
                "success": True,
                "text_query": text_query,
                "filename": file.filename,
                "alpha": alpha,
                "total_results": len(formatted_results),
                "results": formatted_results
            }
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"混合检索失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 