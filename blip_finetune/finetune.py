# 读取csv文件
import sys
sys.path.append('/root/autodl-tmp/multimodel_fasion_retrieval')
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

# 读取CSV文件
df = pd.read_csv('../../dataset/DeepFashion-MultiModal/labels_front.csv')
df["path"] = df["path"].apply(lambda x: "../../dataset/DeepFashion-MultiModal/selected_images/"+x)

# 划分训练集和验证集
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 模型和处理器路径
blip_path = "/root/autodl-fs/blip-image-captioning-base/"

# 初始化模型和处理器
model = BlipForImageTextRetrieval.from_pretrained(blip_path).cuda()
processor = AutoProcessor.from_pretrained(blip_path, use_fast=True)

# 创建数据集和数据加载器
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

train_dataset = FashionDataset(train_df, processor)
val_dataset = FashionDataset(val_df, processor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

model.eval()  # 将模型设置为评估模式

from PIL import Image
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(blip_path, use_fast=True)
image_path = df["path"][0]
text = df["caption"][0]
image = Image.open(image_path).convert('RGB')
inputs = processor(
    images=image,
    text=text,
    return_tensors="pt"
).to("cuda")

# 4. 获取原始的768维特征(importent)
with torch.no_grad():
    # 获取视觉特征(768维)
    vision_outputs = model.vision_model(inputs.pixel_values)
    vision_embeds = vision_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
    
    # 获取文本特征(768维)
    text_outputs = model.text_encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask
    )
    text_embeds = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
    
    # 使用投影层将特征从768维映射到256维
    vision_projected = model.vision_proj(vision_embeds)  # [batch_size, 256]
    text_projected = model.text_proj(text_embeds)      # [batch_size, 256]
    # text_projected和text_embeds都是tensor类型，请计算余弦相似度
    
    # 计算余弦相似度
    cosine_similarity = torch.nn.functional.cosine_similarity(vision_projected, text_projected, dim=1)
    print(cosine_similarity)

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
    
    for batch in progress_bar:
        # 将数据移到GPU
        batch = {k: v.cuda() for k, v in batch.items()}
        
        # 获取视觉特征
        vision_outputs = model.vision_model(batch["pixel_values"])
        vision_embeds = vision_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # 获取文本特征
        text_outputs = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        text_embeds = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # 使用投影层
        vision_embeds = model.vision_proj(vision_embeds)  # [batch_size, 256]
        text_embeds = model.text_proj(text_embeds)  # [batch_size, 256]
        
        # 计算余弦相似度
        cosine_similarity = torch.nn.functional.cosine_similarity(vision_embeds, text_embeds, dim=1)
        avg_cosine_similarity = cosine_similarity.mean().item()
        
        # 计算损失
        loss = compute_loss(vision_embeds, text_embeds)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': loss.item(),
            'cosine_sim': f'{avg_cosine_similarity:.4f}'
        })
    
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    
    for batch in tqdm(val_loader, desc='Evaluating'):
        batch = {k: v.cuda() for k, v in batch.items()}
        
        # 获取视觉特征
        vision_outputs = model.vision_model(batch["pixel_values"])
        vision_embeds = vision_outputs.last_hidden_state[:, 0, :]
        
        # 获取文本特征
        text_outputs = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        
        # 使用投影层
        vision_embeds = model.vision_proj(vision_embeds)
        text_embeds = model.text_proj(text_embeds)
        
        # 计算损失
        loss = compute_loss(vision_embeds, text_embeds)
        total_loss += loss.item()
    
    return total_loss / len(val_loader)

# 训练配置
num_epochs = 10
learning_rate = 5e-5

# 优化器
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 训练循环
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # 训练一个epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
    
    # 评估
    val_loss = evaluate(model, val_loader)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Saved new best model with validation loss: {val_loss:.4f}')
    
    print('-' * 50)

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
print("Training completed!")






