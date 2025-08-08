<script setup>
import { ref, reactive, nextTick } from 'vue'
import ProductCard from './components/ProductCard.vue'
import ControlPanel from './components/ControlPanel.vue'
import { ElImageViewer } from 'element-plus'
import axios from 'axios'

// 配置API基础URL
const API_BASE_URL = 'http://localhost:8000'

// 查询历史状态 - 改为数组存储多次查询结果
const searchHistory = ref([])
const loading = ref(false)
const searchType = ref('hybrid') // 'text', 'image', 'hybrid'
const uploadedImage = ref(null)
const uploadedImageUrl = ref('')

// 图片预览相关
const imageViewerVisible = ref(false)
const imageViewerUrl = ref('')

// 历史记录滚动容器引用
const historyContent = ref(null)

// 筛选条件
const filters = reactive({
  gender: '',
  productType: '不限商品类型',
  searchText: '',
  textWeight: 0.3,
  imageWeight: 0.7,
  resultCount: 5,
  timeline: {
    current: '0:00:00',
    total: '0:11:32'
  }
})

// 处理图片选择
const handleImageSelect = (file) => {
  if (file) {
    uploadedImage.value = file
    uploadedImageUrl.value = URL.createObjectURL(file)
    // 更新状态栏
    updateStatusBar('图片查询: 已上传图片')
    console.log('图片已上传:', file.name)
  }
}

// 处理图片预览
const handleImagePreview = (imageUrl) => {
  imageViewerUrl.value = imageUrl
  imageViewerVisible.value = true
}

// 关闭图片预览
const closeImageViewer = () => {
  imageViewerVisible.value = false
  imageViewerUrl.value = ''
}

// 更新状态栏
const updateStatusBar = (message) => {
  // 这里可以添加状态栏更新逻辑
  console.log(message)
}

// 自动滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (historyContent.value) {
      historyContent.value.scrollTop = historyContent.value.scrollHeight
    }
  })
}

// 处理搜索
const handleSearch = async (searchData) => {
  // 使用从ControlPanel传递的搜索模式和过滤器
  const mode = searchData?.mode || searchType.value
  const currentFilters = searchData?.filters || filters
  
  if (!currentFilters.searchText && !uploadedImage.value) {
    alert('请输入搜索文本或选择图片')
    return
  }

  loading.value = true
  console.log('执行搜索，模式:', mode)

  try {
    let response

    if (mode === 'text') {
      // 文本搜索
      const formData = new FormData()
      formData.append('query', currentFilters.searchText)
      formData.append('top_k', currentFilters.resultCount)
      if (currentFilters.gender !== '不限') formData.append('gender', currentFilters.gender)
      if (currentFilters.productType !== '不限商品类型') formData.append('product_type', currentFilters.productType)

      response = await axios.post(`${API_BASE_URL}/api/search/text`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
    } else if (mode === 'image') {
      // 图片搜索
      const formData = new FormData()
      formData.append('file', uploadedImage.value)
      formData.append('top_k', currentFilters.resultCount)
      if (currentFilters.gender !== '不限') formData.append('gender', currentFilters.gender)
      if (currentFilters.productType !== '不限商品类型') formData.append('product_type', currentFilters.productType)

      response = await axios.post(`${API_BASE_URL}/api/search/image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
    } else {
      // 混合搜索
      const formData = new FormData()
      formData.append('text_query', currentFilters.searchText)
      formData.append('file', uploadedImage.value)
      formData.append('alpha', currentFilters.textWeight)
      formData.append('top_k', currentFilters.resultCount)
      if (currentFilters.gender !== '不限') formData.append('gender', currentFilters.gender)
      if (currentFilters.productType !== '不限商品类型') formData.append('product_type', currentFilters.productType)

      response = await axios.post(`${API_BASE_URL}/api/search/hybrid`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
    }

    if (response.data.success) {
      // 创建新的查询结果对象
      const newSearchResult = {
        id: Date.now(), // 使用时间戳作为唯一ID
        timestamp: new Date().toLocaleString(),
        mode: mode,
        query: currentFilters.searchText || '图片查询',
        results: response.data.results.map(item => ({
          id: item.id,
          image: `${API_BASE_URL}${item.image_url}`,
          similarity: item.score * 100, // 转换距离为相似度
          gender: item.gender,
          productType: item.product_type,
          description: item.caption
        })),
        totalResults: response.data.total_results
      }
      
      // 将新结果添加到历史记录的末尾（最新的在下面）
      searchHistory.value.push(newSearchResult)
      
      // 自动滚动到底部
      scrollToBottom()
      
      // 更新结果数量
      filters.resultCount = response.data.total_results
    }
  } catch (error) {
    console.error('搜索失败:', error)
    alert('搜索失败，请检查网络连接或稍后重试')
  } finally {
    loading.value = false
  }
}

// 切换搜索类型
const switchSearchType = (type) => {
  searchType.value = type
  uploadedImage.value = null
  uploadedImageUrl.value = ''
}

// 清空搜索历史
const clearSearchHistory = () => {
  searchHistory.value = []
  uploadedImage.value = null
  uploadedImageUrl.value = ''
  filters.searchText = ''
}

// 删除单个查询结果
const deleteSearchResult = (resultId) => {
  const index = searchHistory.value.findIndex(result => result.id === resultId)
  if (index !== -1) {
    searchHistory.value.splice(index, 1)
  }
}
</script>

<template>
  <div class="app-container">
    <!-- 查询历史展示 -->
    <div v-if="searchHistory.length > 0" class="history-section">
      <div class="history-header">
        <h3>搜索历史 ({{ searchHistory.length }})</h3>
        <button @click="clearSearchHistory" class="clear-btn">清空历史</button>
      </div>
      
      <div class="history-content" ref="historyContent">
        <!-- 遍历每个查询历史 -->
        <div v-for="searchResult in searchHistory" :key="searchResult.id" class="search-result-item">
          <div class="result-header">
            <div class="result-info">
              <span class="query-type">{{ searchResult.mode === 'text' ? '文本搜索' : searchResult.mode === 'image' ? '图片搜索' : '混合搜索' }}</span>
              <span class="query-text">{{ searchResult.query }}</span>
              <span class="query-time">{{ searchResult.timestamp }}</span>
            </div>
            <button @click="deleteSearchResult(searchResult.id)" class="delete-btn">删除</button>
          </div>
          
          <div class="product-row">
            <ProductCard 
              v-for="product in searchResult.results" 
              :key="`${searchResult.id}-${product.id}`"
              :product="product"
              @preview-image="handleImagePreview"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- 默认展示区域 -->
    <div v-else class="default-section">
      <div class="product-section">
        <!-- <div class="placeholder-card">
          <div class="placeholder-content">
            <i class="el-icon-search"></i>
            <p>开始搜索，结果将显示在这里</p>
          </div>
        </div> -->
      </div>
    </div>

    <!-- 底部控制面板 -->
    <ControlPanel 
      :filters="filters"
      :search-type="searchType"
      :loading="loading"
      @search="handleSearch"
      @select-image="handleImageSelect"
      @preview-image="handleImagePreview"
    />

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner">
        <i class="el-icon-loading"></i>
        <p>搜索中...</p>
      </div>
    </div>

    <!-- 全屏图片预览器 -->
    <ElImageViewer 
      v-if="imageViewerVisible" 
      :url-list="[imageViewerUrl]" 
      @close="closeImageViewer"
      :hide-on-click-modal="true"
      :initial-index="0"
    />
  </div>
</template>

<style scoped>
.app-container {
  min-height: 100vh;
  height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  position: relative;
  overflow: hidden;
}

.search-type-tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.tab-btn {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: #ffffff;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tab-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.tab-btn.active {
  background: #4CAF50;
  border-color: #4CAF50;
}

.history-section {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 10px;
  margin-right: -10px;
  scrollbar-width: thin;
  scrollbar-color: rgba(255, 255, 255, 0.3) transparent;
  display: flex;
  flex-direction: column;
}

.history-section::-webkit-scrollbar {
  width: 8px;
}

.history-section::-webkit-scrollbar-track {
  background: transparent;
}

.history-section::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 4px;
}

.history-section::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.5);
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  color: #ffffff;
  position: sticky;
  top: 0;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  padding: 10px 0;
  z-index: 10;
  backdrop-filter: blur(10px);
  flex-shrink: 0;
}

.history-content {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 10px;
  margin-right: -10px;
  scrollbar-width: thin;
  scrollbar-color: rgba(255, 255, 255, 0.3) transparent;
}

.history-content::-webkit-scrollbar {
  width: 8px;
}

.history-content::-webkit-scrollbar-track {
  background: transparent;
}

.history-content::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 4px;
}

.history-content::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.5);
}

.history-header h3 {
  margin: 0;
  font-size: 18px;
}

.clear-btn {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: #ffffff;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.clear-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.search-result-item {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s ease;
}

.search-result-item:hover {
  border-color: rgba(255, 255, 255, 0.2);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.result-info {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #ffffff;
  flex-wrap: wrap;
}

.query-type {
  background: linear-gradient(135deg, #4CAF50, #45a049);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.query-text {
  font-size: 14px;
  font-weight: 500;
  color: #ffffff;
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.query-time {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.6);
  white-space: nowrap;
}

.delete-btn {
  background: rgba(255, 59, 48, 0.2);
  border: 1px solid rgba(255, 59, 48, 0.3);
  color: #ff3b30;
  padding: 4px 8px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 12px;
  white-space: nowrap;
}

.delete-btn:hover {
  background: rgba(255, 59, 48, 0.3);
  border-color: rgba(255, 59, 48, 0.5);
}

.default-section {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.product-section {
  flex: 1;
}

.product-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 20px;
  margin-bottom: 20px;
}

.placeholder-card {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  color: #ffffff;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  max-width: 400px;
}

.placeholder-content i {
  font-size: 48px;
  margin-bottom: 16px;
  opacity: 0.6;
}

.placeholder-content p {
  margin: 0;
  font-size: 16px;
  opacity: 0.8;
}

.status-bar {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 12px 20px;
  text-align: center;
  color: #ffffff;
  font-size: 14px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.loading-spinner {
  background: rgba(255, 255, 255, 0.1);
  padding: 30px;
  border-radius: 12px;
  text-align: center;
  color: #ffffff;
  backdrop-filter: blur(10px);
}

.loading-spinner i {
  font-size: 32px;
  margin-bottom: 16px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@media (max-width: 1200px) {
  .product-row {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 768px) {
  .product-row {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .search-type-tabs {
    flex-direction: column;
  }
  
  .result-info {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  
  .result-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .query-text {
    max-width: 200px;
  }
}

@media (max-width: 480px) {
  .product-row {
    grid-template-columns: 1fr;
  }
  
  .app-container {
    padding: 10px;
  }
  
  .history-section {
    padding-right: 5px;
    margin-right: -5px;
  }
}
</style>
