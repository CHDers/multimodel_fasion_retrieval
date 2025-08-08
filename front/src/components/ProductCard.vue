<template>
  <div class="product-card">
    <div class="card-image">
      <img 
        :src="product.image" 
        :alt="product.id"
        @error="handleImageError"
        @click="handleImageClick"
        class="clickable-image"
      />
    </div>
    <div class="card-content">
      <div class="product-id">ID: {{ product.id }}</div>
      <div class="similarity">相似度: {{ Number(product.similarity).toFixed(2) }}%</div>
      <div class="gender">性别: {{ product.gender }}</div>
      <div class="product-type">商品类型: {{ product.productType }}</div>
      <div class="description">描述: {{ product.description }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  product: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['preview-image'])

const handleImageError = (event) => {
  // 图片加载失败时显示占位符
  event.target.src = '/placeholder.svg'
}

const handleImageClick = () => {
  // 点击图片时触发预览事件
  emit('preview-image', props.product.image)
}
</script>

<style scoped>
.product-card {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  overflow: hidden;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.product-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
  border-color: rgba(255, 255, 255, 0.3);
}

.card-image {
  width: 100%;
  height: 200px;
  overflow: hidden;
  position: relative;
}

.card-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.clickable-image {
  cursor: pointer;
}

.product-card:hover .card-image img {
  transform: scale(1.05);
}

.card-content {
  padding: 16px;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.product-id {
  font-size: 12px;
  color: #ffffff;
  font-weight: 600;
  word-break: break-all;
  line-height: 1.3;
}

.similarity {
  font-size: 14px;
  color: #4CAF50;
  font-weight: 600;
}

.gender {
  font-size: 12px;
  color: #ffffff;
  opacity: 0.9;
}

.product-type {
  font-size: 12px;
  color: #ffffff;
  opacity: 0.9;
}

.description {
  font-size: 11px;
  color: #ffffff;
  opacity: 0.8;
  line-height: 1.4;
  flex: 1;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}

@media (max-width: 768px) {
  .card-content {
    padding: 12px;
  }
  
  .product-id {
    font-size: 11px;
  }
  
  .similarity {
    font-size: 13px;
  }
  
  .gender,
  .product-type {
    font-size: 11px;
  }
  
  .description {
    font-size: 10px;
  }
}
</style>