#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证模型加载和基本功能
"""

import sys
import os
import numpy as np
from PIL import Image
import torch

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'finetune'))

def test_model_loading():
    """测试模型加载"""
    print("=== 测试模型加载 ===")
    
    try:
        from model_vectorization import FashionEmbeddingModel
        
        # 配置路径
        model_path = "../../training_results_20250705_165112/model_weights/best_model.pth"
        processor_path = "/root/autodl-fs/blip-image-captioning-base/"
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        if not os.path.exists(processor_path):
            print(f"❌ 处理器路径不存在: {processor_path}")
            return False
        
        # 尝试加载模型
        print("正在加载模型...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = FashionEmbeddingModel(model_path, processor_path, device=device)
        print(f"✅ 模型加载成功，使用设备: {device}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_text_encoding(model):
    """测试文本编码"""
    print("\n=== 测试文本编码 ===")
    
    try:
        test_text = "red dress for women"
        print(f"测试文本: {test_text}")
        
        text_embedding = model.encode_text(test_text)
        print(f"✅ 文本编码成功")
        print(f"   向量维度: {text_embedding.shape}")
        print(f"   向量范数: {np.linalg.norm(text_embedding):.4f}")
        print(f"   前5个值: {text_embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文本编码失败: {e}")
        return False

def test_image_encoding(model):
    """测试图片编码"""
    print("\n=== 测试图片编码 ===")
    
    try:
        # 创建一个测试图片
        test_image = Image.new('RGB', (224, 224), color='red')
        print("使用红色测试图片")
        
        image_embedding = model.encode_image(test_image)
        print(f"✅ 图片编码成功")
        print(f"   向量维度: {image_embedding.shape}")
        print(f"   向量范数: {np.linalg.norm(image_embedding):.4f}")
        print(f"   前5个值: {image_embedding[:5]}")
        
        return image_embedding
        
    except Exception as e:
        print(f"❌ 图片编码失败: {e}")
        return False

def test_similarity_computation(model):
    """测试相似度计算"""
    print("\n=== 测试相似度计算 ===")
    
    try:
        # 测试相关的图片和文本
        test_image = Image.new('RGB', (224, 224), color='red')
        test_text = "red color image"
        
        image_embedding = model.encode_image(test_image)
        text_embedding = model.encode_text(test_text)
        
        similarity = model.compute_similarity(image_embedding, text_embedding)
        print(f"✅ 相似度计算成功")
        print(f"   红色图片 vs '红色图片' 文本: {similarity:.4f}")
        
        # 测试不相关的文本
        unrelated_text = "blue ocean waves"
        unrelated_embedding = model.encode_text(unrelated_text)
        unrelated_similarity = model.compute_similarity(image_embedding, unrelated_embedding)
        print(f"   红色图片 vs '蓝色海浪' 文本: {unrelated_similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 相似度计算失败: {e}")
        return False

def test_batch_processing(model):
    """测试批量处理"""
    print("\n=== 测试批量处理 ===")
    
    try:
        # 批量文本测试
        test_texts = [
            "red dress for women",
            "blue jeans for men",
            "black leather jacket"
        ]
        
        text_embeddings = model.encode_batch_texts(test_texts, batch_size=2)
        print(f"✅ 批量文本编码成功")
        print(f"   输入文本数: {len(test_texts)}")
        print(f"   输出矩阵形状: {text_embeddings.shape}")
        
        # 批量图片测试
        test_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue'),
            Image.new('RGB', (224, 224), color='green')
        ]
        
        # 保存测试图片
        test_paths = []
        for i, img in enumerate(test_images):
            path = f"test_image_{i}.jpg"
            img.save(path)
            test_paths.append(path)
        
        image_embeddings = model.encode_batch_images(test_paths, batch_size=2)
        print(f"✅ 批量图片编码成功")
        print(f"   输入图片数: {len(test_paths)}")
        print(f"   输出矩阵形状: {image_embeddings.shape}")
        
        # 清理测试文件
        for path in test_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return True
        
    except Exception as e:
        print(f"❌ 批量处理失败: {e}")
        return False

def test_search_functionality(model):
    """测试搜索功能"""
    print("\n=== 测试搜索功能 ===")
    
    try:
        # 创建测试数据
        test_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue'),
            Image.new('RGB', (224, 224), color='green')
        ]
        
        test_texts = [
            "red color dress",
            "blue color jeans", 
            "green color shirt"
        ]
        
        # 保存测试图片
        test_paths = []
        for i, img in enumerate(test_images):
            path = f"search_test_{i}.jpg"
            img.save(path)
            test_paths.append(path)
        
        # 编码
        image_embeddings = model.encode_batch_images(test_paths, batch_size=4)
        text_embeddings = model.encode_batch_texts(test_texts, batch_size=4)
        
        # 文本到图片搜索
        query_text = "red color"
        similar_images = model.find_similar_images(
            query_text, image_embeddings, test_paths, top_k=2
        )
        
        print(f"✅ 文本到图片搜索成功")
        print(f"   查询: '{query_text}'")
        for i, (path, score) in enumerate(similar_images):
            print(f"   {i+1}. {path}: {score:.4f}")
        
        # 图片到文本搜索
        query_image = test_paths[0]  # 红色图片
        similar_texts = model.find_similar_texts(
            query_image, text_embeddings, test_texts, top_k=2
        )
        
        print(f"✅ 图片到文本搜索成功")
        print(f"   查询图片: {query_image}")
        for i, (text, score) in enumerate(similar_texts):
            print(f"   {i+1}. {text}: {score:.4f}")
        
        # 清理测试文件
        for path in test_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return True
        
    except Exception as e:
        print(f"❌ 搜索功能失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始快速测试...\n")
    
    # 测试模型加载
    model = test_model_loading()
    if not model:
        print("\n❌ 模型加载失败，停止测试")
        return
    
    # 测试各项功能
    tests = [
        ("文本编码", lambda: test_text_encoding(model)),
        ("图片编码", lambda: test_image_encoding(model)),
        ("相似度计算", lambda: test_similarity_computation(model)),
        ("批量处理", lambda: test_batch_processing(model)),
        ("搜索功能", lambda: test_search_functionality(model))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✅ 所有测试通过！模型工作正常。")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 