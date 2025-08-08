#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时尚多模态检索模型使用示例
"""

import numpy as np
import pandas as pd
from PIL import Image
import pickle
import os
from data_embedding.model_vectorization import FashionEmbeddingModel

def main():
    from pymilvus import MilvusClient, DataType

    uri = "https://in03-338d6950a393b2c.serverless.gcp-us-west1.cloud.zilliz.com"
    token = "xxxxxxxx"


    client = MilvusClient(
        uri=uri,
        token=token
    )


    # 配置路径
    model_path = "../../training_results_20250705_165112/model_weights/best_model.pth"
    processor_path = "/root/autodl-fs/blip-image-captioning-base/"
    
    # 初始化模型
    print("正在加载模型...")
    model = FashionEmbeddingModel(model_path, processor_path)
    print("模型加载完成！")

    # 读取CSV文件
    df = pd.read_csv('../../dataset/DeepFashion-MultiModal/labels_front.csv')
    df["path"] = df["path"].apply(lambda x: "../../dataset/DeepFashion-MultiModal/selected_images/"+x)

    # 对df的数据进行遍历
    for index, row in df.iterrows():
        # 图片路径
        image_path = row['path']
        # 图片描述
        caption = row['caption']
        # 读取图片
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image_embedding = model.encode_image(image)
        else:
            print(f"图片不存在: {image_path}")
            continue
        text_embedding = model.encode_text(caption)

        # 插入单条记录
        res = client.insert(
            collection_name="fasion_multimodel_embedding",
            data={
                'pk': row['image_id'],
                'image_embedding': image_embedding,
                'text_embedding': text_embedding,
                'gender': row['gender'],
                'product_type': row['product_type'],
                'caption': caption
            }
        )
        print(f"插入成功: {row['image_id']}")

if __name__ == "__main__":
    main() 