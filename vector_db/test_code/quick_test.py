#!/usr/bin/env python3
"""
快速测试脚本 - 验证向量检索功能
使用方法: python quick_test.py
"""

import sys
import os
import numpy as np
from pymilvus import MilvusClient

# 添加项目路径
sys.path.append('/root/autodl-tmp/multimodel_fasion_retrieval')
from data_embedding.model_vectorization import FashionEmbeddingModel
from vector_db.retrieval_util import (
    search_text_embedding, 
    search_image_embedding, 
    hybrid_search_weighted
)

def quick_test():
    """快速测试函数"""
    print("🔍 快速测试向量检索功能")
    
    # 1. 连接Milvus
    try:
        uri = "https://in03-338d6950a393b2c.serverless.gcp-us-west1.cloud.zilliz.com"
        token = "xxxxxxx"
        
        client = MilvusClient(
            uri=uri,
            token=token
        )
        print("✅ Milvus连接成功")
    except Exception as e:
        print(f"❌ Milvus连接失败: {e}")
        return
    
    # 3. 加载模型
    try:
        model_path = "../../../training_results_20250705_165112/model_weights/best_model.pth"
        processor_path = "/root/autodl-fs/blip-image-captioning-base/"
        model = FashionEmbeddingModel(model_path, processor_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 4. 简单测试
    try:
        print("\n📝 测试文本检索...")
        query_text = "red dress"
        query_emb = model.encode_text(query_text)
        
        # 使用retrieval_util中的函数进行文本检索
        results = search_text_embedding(client, query_emb, top_k=2)
        print(f"查询: '{query_text}' -> 找到 {len(results)} 个结果")
        print(results)
        
        print("\n🖼️  测试图片检索...")
        # 使用一个示例图片路径
        test_img_path = "../../../dataset/DeepFashion-MultiModal/selected_images/MEN-Denim-id_00000089-28_1_front.jpg"
        if os.path.exists(test_img_path):
            query_emb = model.encode_image(test_img_path)
            
            # 使用retrieval_util中的函数进行图片检索
            results = search_image_embedding(client, query_emb, top_k=2)
            print(f"查询图片: {test_img_path} -> 找到 {len(results)} 个结果")
            print(results)
        else:
            print(f"⚠️  测试图片不存在: {test_img_path}")
        
        print("\n🔄 测试混合检索...")
        if os.path.exists(test_img_path):
            img_emb = model.encode_image(test_img_path)
            text_emb = model.encode_text("blue jeans")
            
            # 使用retrieval_util中的函数进行混合检索
            results = hybrid_search_weighted(client, img_emb, text_emb, alpha=0.5, top_k=2)
            print(results[0])
        
        print("\n✅ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    finally:
        try:
            print("✅ 测试完成")
        except:
            pass

if __name__ == "__main__":
    quick_test() 