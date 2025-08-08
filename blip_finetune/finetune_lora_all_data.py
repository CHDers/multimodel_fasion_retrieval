import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipForImageTextRetrieval, AutoProcessor
from torch.optim import AdamW
import torch.nn.functional as F
from lora_layer import add_lora_to_linear_layer, get_lora_params, save_lora_weights, load_lora_weights
import json
from datetime import datetime
import os

# 这个 processor 是BLIP模型的多模态预处理器，它的作用是将原始的图像和文本数据转换为模型可以理解的数字格式。
class FashionDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        text = row['caption']
        # 多模态处理器的调用，同时处理图像和文本数据，将图片和文本编码转换为模型可以理解的格式
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 移除批次维度
        for k,v in inputs.items():
            inputs[k] = v.squeeze(0)
            
        return inputs


"""
1. input_ids
作用: 文本的数字化表示
内容: 将文本描述转换为词汇表中对应的整数ID序列
处理过程:
原始文本 → 分词 → 词汇表查找 → 整数序列
例如: "red dress" → [101, 2417, 4377, 102] (具体数字取决于词汇表)
在代码中: 用于文本编码器(text_encoder)的输入，生成文本特征表示

2. attention_mask
作用: 指示哪些位置是真实的文本token，哪些是填充(padding)
内容: 二进制掩码，1表示真实token，0表示padding token
为什么需要: 因为批次中不同文本长度不同，需要padding到统一长度

3. pixel_values
作用: 将图像转换为数值表示
内容: 将图像像素值转换为数值矩阵
处理过程:
原始图像 → 像素值矩阵 (通常是RGB三通道)
"""
def collate_fn(batch):
    # 获取批次中的最大长度
    max_length = max([b['input_ids'].size(0) for b in batch])
    
    processed_batch = {
        'input_ids': [],
        'attention_mask': [],
        'pixel_values': []
    }
    
    for item in batch:
        # 处理input_ids
        padding_length = max_length - item['input_ids'].size(0)
        padded_input_ids = F.pad(item['input_ids'], (0, padding_length), value=processor.tokenizer.pad_token_id)
        processed_batch['input_ids'].append(padded_input_ids)
        
        # 处理attention_mask
        padded_attention_mask = F.pad(item['attention_mask'], (0, padding_length), value=0)
        processed_batch['attention_mask'].append(padded_attention_mask)
        
        # 处理pixel_values
        processed_batch['pixel_values'].append(item['pixel_values'])
    
    # 将列表转换为张量
    processed_batch['input_ids'] = torch.stack(processed_batch['input_ids'])
    processed_batch['attention_mask'] = torch.stack(processed_batch['attention_mask'])
    processed_batch['pixel_values'] = torch.stack(processed_batch['pixel_values'])
    
    return processed_batch

def compute_loss(vision_embeds, text_embeds, temperature=0.07):
    """计算对比损失"""
    # 归一化特征
    vision_embeds = F.normalize(vision_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # 计算相似度矩阵
    logits = torch.matmul(vision_embeds, text_embeds.transpose(0, 1)) / temperature
    
    # 创建标签（对角线为正例）
    labels = torch.arange(len(logits), device=logits.device)
    
    # 计算图像到文本和文本到图像的损失
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.transpose(0, 1), labels)
    
    # 总损失为两个方向损失的平均
    total_loss = (loss_i2t + loss_t2i) / 2
    return total_loss

def train_one_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    # 记录每个batch的指标
    batch_metrics = []
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.cuda() for k, v in batch.items()}
        
        # 获取视觉特征
        vision_outputs = model.vision_model(batch["pixel_values"])
        # 获取视觉特征的最后一个隐藏状态，即图像特征，768维
        vision_embeds = vision_outputs.last_hidden_state[:, 0, :]
        
        # 获取文本特征
        text_outputs = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        # 获取文本特征的最后一个隐藏状态，即文本特征，768维
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        
        # 使用投影层，将图片和文本向量都转换为256维的向量
        vision_embeds = model.vision_proj(vision_embeds) 
        text_embeds = model.text_proj(text_embeds)
        
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(vision_embeds, text_embeds, dim=1)
        avg_cosine_similarity = cosine_similarity.mean().item()
        
        # 计算损失
        loss = compute_loss(vision_embeds, text_embeds)
        # 梯度清零，防止梯度累积
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        
        batch_loss = loss.item() # 损失值
        total_loss += batch_loss # 累加损失值
        
        # 记录当前batch的指标
        batch_metrics.append({
            'epoch': epoch + 1,
            'batch': batch_idx + 1,
            'loss': batch_loss,
            'cosine_similarity': avg_cosine_similarity
        })
        
        progress_bar.set_postfix({
            'loss': batch_loss,
            'cosine_sim': f'{avg_cosine_similarity:.4f}'
        })
    
    return total_loss / len(train_loader), batch_metrics # 返回平均损失和每个batch的指标

if __name__ == "__main__":
    # 设置随机种子，多次运行代码会产生完全相同的结果 
    # 便于调试和验证：排除随机性干扰，专注于算法本身的问题
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_seed(42)
    
    # 创建保存结果的目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'training_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建模型和LoRA权重保存子文件夹
    model_weights_dir = os.path.join(results_dir, "model_weights")
    lora_weights_dir = os.path.join(results_dir, "lora_weights")
    os.makedirs(model_weights_dir, exist_ok=True)
    os.makedirs(lora_weights_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv('../../dataset/DeepFashion-MultiModal/labels_front.csv')
    df["path"] = df["path"].apply(lambda x: "../../dataset/DeepFashion-MultiModal/selected_images/"+x)
    
    # 所有数据都用于训练
    train_df = df
    
    # 保存训练配置
    config = {
        'num_epochs': 50,
        'learning_rate': 5e-4,
        'batch_size': 8,
        'lora_rank': 16,
        'lora_alpha': 32,
        'train_size': len(train_df),
        'seed': 42
    }
    
    with open(f'{results_dir}/training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # 模型和处理器路径
    blip_path = "/root/autodl-fs/blip-image-captioning-base/"
    
    # 初始化模型和处理器
    model = BlipForImageTextRetrieval.from_pretrained(blip_path)
    processor = AutoProcessor.from_pretrained(blip_path, use_fast=True)
    
    # 添加LoRA层
    target_modules = [
        "query",
        "value",
        "vision_proj",
        "text_proj"
    ]
    
    model = add_lora_to_linear_layer(
        model,
        target_modules,
        rank=16,
        alpha=32
    )
    
    # 将整个模型移动到CUDA
    model = model.cuda()
    
    # 冻结原始参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 只训练LoRA参数
    lora_params = get_lora_params(model)
    for param in lora_params:
        param.requires_grad = True
    
    # 创建数据集和数据加载器
    train_dataset = FashionDataset(train_df, processor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # 训练配置
    num_epochs = 50
    learning_rate = 5e-4  # 对于LoRA参数可以使用更大的学习率
    
    # 优化器（只优化LoRA参数）
    optimizer = AdamW(lora_params, lr=learning_rate)
    
    # 训练循环
    history = {
        'epoch': [],
        'train_loss': [],
        'batch_metrics': [],
    }

    # 只保留最优模型
    best_loss = float('inf')
    best_model_path = os.path.join(model_weights_dir, "best_model.pth")
    best_lora_path = os.path.join(lora_weights_dir, "best_lora.pth")

    for epoch in range(config['num_epochs']):
        # 训练一个epoch
        train_loss, batch_metrics = train_one_epoch(model, train_loader, optimizer, epoch)
        
        # 记录历史
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['batch_metrics'].extend(batch_metrics)
        
        # 保存本epoch的batch指标（含余弦相似度）
        batch_df = pd.DataFrame(batch_metrics)
        batch_df.to_csv(f'{results_dir}/batch_metrics_epoch_{epoch+1}.csv', index=False)
        
        # 只保存最优模型权重
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_model_path)
            save_lora_weights(model, best_lora_path)
            print(f"New best model saved at epoch {epoch+1} with loss {train_loss:.4f}")
        
        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'Training Loss: {train_loss:.4f}')
        print('-' * 50)
    
    # 保存每个epoch的汇总指标
    epoch_df = pd.DataFrame({
        'epoch': history['epoch'],
        'train_loss': history['train_loss'],
    })
    epoch_df.to_csv(f'{results_dir}/epoch_metrics.csv', index=False)
    
    print("Training completed!")
    print(f"All training results have been saved to: {results_dir}/")