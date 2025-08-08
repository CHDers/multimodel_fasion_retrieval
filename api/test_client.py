#!/usr/bin/env python3
"""
API测试客户端
用于测试时尚多模态检索API的各个功能
"""

import requests
import json
import os
from pathlib import Path

# API基础URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查"""
    print("🔍 测试健康检查...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_text_search():
    """测试文本检索"""
    print("\n🔍 测试文本检索...")
    try:
        data = {
            "query": "red dress",
            "top_k": 3,
            "gender": "women",
            "product_type": "dress"
        }
        
        response = requests.post(f"{BASE_URL}/api/search/text", data=data)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"查询: {result['query']}")
            print(f"结果数量: {result['total_results']}")
            for i, item in enumerate(result['results']):
                print(f"  结果 {i+1}: {item}")
        else:
            print(f"❌ 文本检索失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 文本检索测试失败: {e}")

def test_image_search():
    """测试图片检索"""
    print("\n🔍 测试图片检索...")
    try:
        # 使用测试图片路径
        test_img_path = "../../../dataset/DeepFashion-MultiModal/selected_images/MEN-Denim-id_00000089-28_1_front.jpg"
        
        if not os.path.exists(test_img_path):
            print(f"⚠️  测试图片不存在: {test_img_path}")
            return
        
        with open(test_img_path, "rb") as f:
            files = {"file": f}
            data = {
                "top_k": 3,
                "gender": "men",
                "product_type": "jeans"
            }
            
            response = requests.post(f"{BASE_URL}/api/search/image", files=files, data=data)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"文件名: {result['filename']}")
                print(f"结果数量: {result['total_results']}")
                for i, item in enumerate(result['results']):
                    print(f"  结果 {i+1}: {item}")
            else:
                print(f"❌ 图片检索失败: {response.text}")
                
    except Exception as e:
        print(f"❌ 图片检索测试失败: {e}")

def test_hybrid_search():
    """测试混合检索"""
    print("\n🔍 测试混合检索...")
    try:
        # 使用测试图片路径
        test_img_path = "../../../dataset/DeepFashion-MultiModal/selected_images/MEN-Denim-id_00000089-28_1_front.jpg"
        
        if not os.path.exists(test_img_path):
            print(f"⚠️  测试图片不存在: {test_img_path}")
            return
        
        with open(test_img_path, "rb") as f:
            files = {"file": f}
            data = {
                "text_query": "blue jeans",
                "alpha": 0.6,
                "top_k": 3,
                "gender": "men",
                "product_type": "jeans"
            }
            
            response = requests.post(f"{BASE_URL}/api/search/hybrid", files=files, data=data)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"文本查询: {result['text_query']}")
                print(f"文件名: {result['filename']}")
                print(f"权重: {result['alpha']}")
                print(f"结果数量: {result['total_results']}")
                for i, item in enumerate(result['results']):
                    print(f"  结果 {i+1}: {item}")
            else:
                print(f"❌ 混合检索失败: {response.text}")
                
    except Exception as e:
        print(f"❌ 混合检索测试失败: {e}")

def main():
    """主测试函数"""
    print("🧪 开始API测试...")
    
    # 测试健康检查
    if not test_health_check():
        print("❌ 健康检查失败，停止测试")
        return
    
    # 测试各个API端点
    test_text_search()
    test_image_search()
    test_hybrid_search()
    
    print("\n✅ API测试完成!")

if __name__ == "__main__":
    main() 