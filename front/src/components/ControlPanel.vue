<template>
    <div class="control-panel">
      <!-- 主要搜索区域 -->
      <div class="main-section">
        <!-- 筛选条件 -->
        <div class="filter-section">
          <div class="filter-item">
            <label>性别筛选:</label>
            <el-select v-model="localFilters.gender" placeholder="选择性别" class="filter-select">
              <el-option label="男装" value="MEN" />
              <el-option label="女装" value="WOMEN" />
              <el-option label="不限" value="" />
            </el-select>
          </div>
          <div class="filter-item">
            <label>商品类型:</label>
            <el-select v-model="localFilters.productType" placeholder="选择类型" class="filter-select">
              <el-option label="不限" value="不限商品类型" />
              <el-option label="Denim" value="Denim" />
              <el-option label="Jackets_Vests" value="Jackets_Vests" />
              <el-option label="Pants" value="Pants" />
              <el-option label="Shirts_Polos" value="Shirts_Polos" />
              <el-option label="Shorts" value="Shorts" />
              <el-option label="Suiting" value="Suiting" />
              <el-option label="Sweaters" value="Sweaters" />
              <el-option label="Sweatshirts_Hoodies" value="Sweatshirts_Hoodies" />
              <el-option label="Tees_Tanks" value="Tees_Tanks" />
              <el-option label="Blouses_Shirts" value="Blouses_Shirts" />
              <el-option label="Cardigans" value="Cardigans" />
              <el-option label="Dresses" value="Dresses" />
              <el-option label="Graphic_Tees" value="Graphic_Tees" />
              <el-option label="Jackets_Coats" value="Jackets_Coats" />
              <el-option label="Leggings" value="Leggings" />
              <el-option label="Rompers_Jumpsuits" value="Rompers_Jumpsuits" />
              <el-option label="Skirts" value="Skirts" />
            </el-select>
          </div>
        </div>
        <!-- 搜索区域 -->
        <div class="search-section">
          <div class="search-input-container">
            <el-input
              v-model="localFilters.searchText"
              type="textarea"
              :autosize="{ minRows: 2, maxRows: 4 }"
              placeholder="请输入检索内容..."
              class="search-textarea"
              clearable
            />
          </div>
          
          <div class="action-buttons">
            <!-- 隐藏的图片选择input -->
            <input type="file" ref="fileInput" accept="image/*" style="display:none" @change="onFileChange" />
            <el-button 
              type="primary" 
              @click="triggerFileInput"
              :disabled="loading"
              class="action-btn image-btn"
            >
              <el-icon><Picture /></el-icon>
              选择图片
            </el-button>
            <el-button 
              type="primary" 
              @click="handleSearch"
              :loading="loading"
              class="action-btn search-btn main-search-btn"
            >
              <el-icon v-if="!loading"><Search /></el-icon>
              {{ loading ? '搜索中...' : '开始搜索' }}
            </el-button>
          </div>
        </div>

        <!-- 图片预览区域 - 独立显示 -->
        <div v-if="imageUrl" class="image-preview-section">
          <div class="image-preview-container">
            <img :src="imageUrl" class="image-thumb" @click="handleImagePreview" alt="预览" />
            <el-button @click="clearImage" size="small" type="danger" class="clear-image-btn" circle>
              <el-icon><Close /></el-icon>
            </el-button>
          </div>
          <span class="image-tip">点击图片预览</span>
        </div>
      </div>

      <!-- 可折叠的高级设置 -->
      <div class="advanced-section">
        <div class="advanced-toggle" @click="showAdvanced = !showAdvanced">
          <span>高级设置</span>
          <el-icon class="toggle-icon" :class="{ 'rotated': showAdvanced }">
            <ArrowDown />
          </el-icon>
        </div>
        
        <div v-show="showAdvanced" class="advanced-content">
          <div class="settings-grid">
            <div class="setting-group">
              <label class="setting-label">文本/图片权重比例</label>
              <div class="weight-control">
                <el-slider
                  v-model="localFilters.textWeight"
                  :min="0"
                  :max="1"
                  :step="0.01"
                  class="weight-slider"
                  @input="val => localFilters.imageWeight = (1 - val).toFixed(2) * 1"
                />
                <div class="weight-display">
                  <span class="weight-text">文本: {{ localFilters.textWeight.toFixed(2) }}</span>
                  <span class="weight-text">图片: {{ (1-localFilters.textWeight).toFixed(2) }}</span>
                </div>
              </div>
            </div>
            
            <div class="setting-group">
              <label class="setting-label">返回结果数量</label>
              <div class="count-control">
                <el-slider
                  v-model="localFilters.resultCount"
                  :min="1"
                  :max="10"
                  :step="1"
                  show-input
                  class="count-slider"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref, reactive, watch } from 'vue'
  import { ArrowDown, Close, Picture, Search } from '@element-plus/icons-vue'

  const props = defineProps({
  filters: {
    type: Object,
    required: true
  },
  searchType: {
    type: String,
    default: 'hybrid'
  },
  loading: {
    type: Boolean,
    default: false
  }
})
  
  const emit = defineEmits(['search', 'select-image', 'preview-image'])
  
  const localFilters = reactive({
    gender: props.filters.gender,
    productType: props.filters.productType,
    searchText: props.filters.searchText,
    textWeight: props.filters.textWeight,
    imageWeight: 1 - props.filters.textWeight,
    resultCount: props.filters.resultCount,
    timeline: { ...props.filters.timeline },
    searchMode: 'auto' // 新增自动搜索模式
  })
  
  // 保证权重和为1
  watch(() => localFilters.textWeight, (val) => {
    localFilters.imageWeight = (1 - val).toFixed(2) * 1
  })
  
  const timelineProgress = ref(0)
  const showAdvanced = ref(false)
  
  const fileInput = ref(null)
  const imageUrl = ref('')

  function triggerFileInput() {
    fileInput.value && fileInput.value.click()
  }
  function onFileChange(e) {
    const file = e.target.files[0]
    if (file) {
      imageUrl.value = URL.createObjectURL(file)
      emit('select-image', file) // 将图片传递给父组件
    }
  }

  // 清除图片
  function clearImage() {
    imageUrl.value = ''
    if (fileInput.value) {
      fileInput.value.value = ''
    }
  }

  // 处理图片预览
  function handleImagePreview() {
    if (imageUrl.value) {
      emit('preview-image', imageUrl.value)
    }
  }

  // 根据输入内容自动判断搜索模式
  function determineSearchMode() {
    // 更严格地检查文本是否为空
    const hasText = localFilters.searchText && localFilters.searchText.trim().length > 0
    const hasImage = imageUrl.value !== ''
    
    console.log('检测搜索模式:', { hasText, hasImage, text: localFilters.searchText })
    
    if (hasText && hasImage) {
      console.log('选择混合检索模式')
      return 'hybrid' // 文本和图片都有，混合搜索
    } else if (hasText) {
      console.log('选择文本检索模式')
      return 'text' // 只有文本，文本搜索（即使选择了标签也是文本搜索）
    } else if (hasImage) {
      console.log('选择图片检索模式')
      return 'image' // 只有图片，图片检索（即使选择了标签也是图片搜索）
    } else {
      console.log('默认混合检索模式')
      return 'hybrid' // 默认混合搜索
    }
  }
  
  // 处理搜索按钮点击事件
  function handleSearch() {
    // 自动判断搜索模式
    const searchMode = determineSearchMode()
    
    // 根据不同的搜索模式调整权重
    if (searchMode === 'text') {
      localFilters.textWeight = 1
      localFilters.imageWeight = 0
    } else if (searchMode === 'image') {
      localFilters.textWeight = 0
      localFilters.imageWeight = 1
    }
    
    // 发送搜索事件，并传递搜索模式
    emit('search', { 
      mode: searchMode,
      filters: localFilters
    })
  }
  
  // 监听props变化，同步到本地状态
  watch(() => props.filters, (newFilters) => {
    Object.assign(localFilters, newFilters)
  }, { deep: true })
  
  // 监听本地状态变化，同步到父组件
  watch(localFilters, (newFilters) => {
    Object.assign(props.filters, newFilters)
  }, { deep: true })
  </script>
  
  <style scoped>
  .control-panel {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px 16px 0 0;
    padding: 24px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-bottom: none;
    max-width: 100%;
    margin: 0;
    box-shadow: 
      0 -8px 32px rgba(0, 0, 0, 0.3),
      0 0 0 1px rgba(255, 255, 255, 0.1) inset;
    position: relative;
    z-index: 100;
  }

  .control-panel::before {
    content: '';
    position: absolute;
    top: -1px;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 204, 255, 0.5), transparent);
  }

  /* 主要区域布局 */
  .main-section {
    margin-bottom: 16px;
  }

  .filter-section {
    display: flex;
    gap: 20px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }

  .filter-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .filter-item label {
    color: #ffffff;
    font-size: 14px;
    white-space: nowrap;
    min-width: 80px;
    font-weight: 500;
  }

  .filter-select {
    width: 150px;
  }

  /* 搜索区域 */
  .search-section {
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }

  .search-input-container {
    flex: 1;
  }

  .image-preview-section {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 12px;
    padding: 8px 0;
  }

  .search-textarea {
    width: 100%;
    min-height: 60px;
    font-size: 15px;
  }

  .image-preview-area {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
  }

  .image-preview-container {
    position: relative;
    display: inline-block;
  }

  .image-thumb {
    width: 50px;
    height: 50px;
    object-fit: cover;
    border-radius: 8px;
    border: 2px solid #00ccff;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 204, 255, 0.3);
  }

  .image-thumb:hover {
    box-shadow: 0 4px 16px rgba(0, 204, 255, 0.5);
    transform: scale(1.05);
  }

  .clear-image-btn {
    position: absolute;
    top: -8px;
    right: -8px;
    width: 20px !important;
    height: 20px !important;
    min-height: 20px !important;
    padding: 0 !important;
    background: rgba(255, 107, 107, 0.9) !important;
    border: 1px solid #ff6b6b !important;
  }

  .clear-image-btn:hover {
    background: rgba(255, 82, 82, 1) !important;
    transform: scale(1.1);
  }

  .image-tip {
    color: #00ccff;
    font-size: 13px;
    opacity: 0.8;
  }

  .action-buttons {
    display: flex;
    flex-direction: row;
    gap: 12px;
    flex-shrink: 0;
    align-items: flex-start;
  }

  .action-btn {
    padding: 10px 20px !important;
    font-size: 14px !important;
    height: 40px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    font-weight: 500;
    letter-spacing: 0.5px;
    border-radius: 8px;
    transition: all 0.3s ease;
  }

  .image-btn {
    background: linear-gradient(135deg, rgba(0, 153, 255, 0.7), rgba(0, 204, 255, 0.5)) !important;
    min-width: 100px;
  }

  .main-search-btn {
    background: linear-gradient(135deg, rgba(0, 102, 255, 0.9), rgba(0, 204, 255, 0.8)) !important;
    font-weight: 600;
    min-width: 120px;
    box-shadow: 0 4px 16px rgba(0, 153, 255, 0.4) !important;
  }

  .main-search-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 153, 255, 0.6) !important;
  }

  /* 图片预览内联样式 */
  .image-preview-inline {
    display: flex;
    align-items: center;
    gap: 4px;
    position: relative;
  }

  .image-thumb-small {
    width: 32px;
    height: 32px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #00ccff;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .image-thumb-small:hover {
    box-shadow: 0 2px 8px rgba(0, 204, 255, 0.4);
  }

  .clear-image-btn {
    color: #ff6b6b !important;
    font-size: 16px !important;
    padding: 0 !important;
    width: 20px !important;
    height: 20px !important;
    min-height: 20px !important;
    border: none !important;
    background: none !important;
  }

  .clear-image-btn:hover {
    color: #ff5252 !important;
    background: rgba(255, 107, 107, 0.1) !important;
  }

  /* 高级设置区域 */
  .advanced-section {
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding-top: 16px;
    margin-top: 16px;
  }

  .advanced-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    padding: 10px 0;
    color: #00ccff;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s ease;
  }

  .advanced-toggle:hover {
    color: #33d6ff;
  }

  .toggle-icon {
    transition: transform 0.3s ease;
    font-size: 16px;
  }

  .toggle-icon.rotated {
    transform: rotate(180deg);
  }

  .advanced-content {
    padding-top: 16px;
    animation: slideDown 0.3s ease;
  }

  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }

  .setting-group {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .setting-label {
    color: #ffffff;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 8px;
  }

  .weight-control,
  .count-control {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .weight-slider,
  .count-slider {
    width: 100%;
  }

  .weight-display {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
  }

  .weight-text {
    color: #00ccff;
    font-size: 13px;
    font-weight: 500;
  }

  /* 响应式设计 */
  @media (max-width: 768px) {
    .control-panel {
      padding: 16px;
    }

    .filter-section {
      flex-direction: column;
      gap: 12px;
    }

    .search-section {
      flex-direction: column;
      gap: 16px;
    }

    .action-buttons {
      flex-direction: row;
      min-width: auto;
      justify-content: center;
      align-items: flex-start;
    }

    .image-preview-section {
      justify-content: center;
    }

    .settings-grid {
      grid-template-columns: 1fr;
      gap: 20px;
    }
  }

  @media (max-width: 480px) {
    .filter-item {
      flex-direction: column;
      align-items: flex-start;
      gap: 8px;
    }

    .filter-select {
      width: 100%;
    }

    .action-buttons {
      flex-direction: column;
    }
  }
  
  .main-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    gap: 20px;
  }
  
  .filter-section {
    display: flex;
    gap: 24px;
    margin-bottom: 10px;
  }
  
  .filter-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .filter-item label {
    color: #ffffff;
    font-size: 14px;
    white-space: nowrap;
  }
  
  .filter-select {
    width: 140px;
  }
  
  .search-section {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  
  .search-input {
    width: 200px;
  }
  
  .select-image-btn,
  .search-btn {
    background: linear-gradient(135deg, rgba(0, 102, 255, 0.15), rgba(0, 204, 255, 0.05));
    color: #00ccff;
    border: 1px solid rgba(0, 153, 255, 0.4);
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 500;
    font-size: 14px;
    letter-spacing: 0.5px;
    backdrop-filter: blur(8px);
    box-shadow: 
      0 4px 16px rgba(0, 153, 255, 0.2),
      inset 0 1px 0 rgba(255, 255, 255, 0.1),
      inset 0 -1px 0 rgba(0, 153, 255, 0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    text-shadow: 0 0 8px rgba(0, 204, 255, 0.5);
  }
  
  .select-image-btn::before,
  .search-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 204, 255, 0.2), transparent);
    transition: left 0.6s;
  }
  
  .select-image-btn::after,
  .search-btn::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ccff, transparent);
    opacity: 0;
    transition: opacity 0.3s;
  }
  
  .select-image-btn:hover,
  .search-btn:hover {
    background: linear-gradient(135deg, rgba(0, 102, 255, 0.25), rgba(0, 204, 255, 0.15));
    color: #ffffff;
    border: 1px solid rgba(0, 153, 255, 0.7);
    box-shadow: 
      0 8px 24px rgba(0, 153, 255, 0.3),
      inset 0 1px 0 rgba(255, 255, 255, 0.2),
      0 0 15px rgba(0, 204, 255, 0.4);
    transform: translateY(-2px);
  }
  
  .select-image-btn:hover::before,
  .search-btn:hover::before {
    left: 100%;
  }
  
  .select-image-btn:hover::after,
  .search-btn:hover::after {
    opacity: 1;
  }
  
  .select-image-btn:active,
  .search-btn:active {
    background: linear-gradient(135deg, rgba(0, 102, 255, 0.35), rgba(0, 204, 255, 0.2));
    color: #ffffff;
    border: 1px solid rgba(0, 153, 255, 0.9);
    transform: translateY(0);
    box-shadow: 
      0 2px 10px rgba(0, 153, 255, 0.4),
      inset 0 2px 4px rgba(0, 0, 0, 0.2);
  }
  
  .select-image-btn:disabled,
  .search-btn:disabled {
    background: rgba(0, 51, 102, 0.1);
    color: rgba(0, 153, 255, 0.3);
    border: 1px solid rgba(0, 102, 153, 0.2);
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
    text-shadow: none;
  }
  
  .timeline-controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .timeline-info {
    display: flex;
    gap: 20px;
    color: #ffffff;
    font-size: 12px;
  }
  
  .result-count {
    color: #00ccff;
    font-weight: 600;
    text-shadow: 0 0 8px rgba(0, 204, 255, 0.4);
  }
  
  .text-weight,
  .image-weight {
    opacity: 0.8;
  }
  
  .timeline-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    flex: 1;
    margin: 0 20px;
  }
  
  .time-display {
    color: #ffffff;
    font-size: 12px;
    font-family: 'Courier New', monospace;
    min-width: 60px;
  }
  
  .timeline-slider {
    flex: 1;
    max-width: 300px;
  }
  
  .custom-slider {
    --el-slider-main-bg-color: #00ccff;
    --el-slider-runway-bg-color: rgba(0, 153, 255, 0.15);
    --el-slider-stop-bg-color: #00ccff;
  }
  
  .playback-controls {
    display: flex;
    gap: 8px;
  }
  
  .control-btn {
    color: #ffffff;
    font-size: 16px;
    padding: 8px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
  }
  
  .control-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    color: #00ccff;
    text-shadow: 0 0 8px rgba(0, 204, 255, 0.5);
  }
  
  .action-buttons {
    display: flex;
    gap: 8px;
  }
  
  .action-btn {
    color: #ffffff;
    font-size: 14px;
    padding: 6px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
  }
  
  .action-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    color: #00ccff;
    text-shadow: 0 0 8px rgba(0, 204, 255, 0.5);
  }
  
  .search-text-row {
    width: 100%;
    margin: 16px 0 8px 0;
  }
  .search-textarea {
    flex: 1;
    min-height: 48px;
    font-size: 15px;
    resize: vertical;
  }
  
  .weight-row {
    display: flex;
    align-items: center;
    margin: 18px 0 8px 0;
    gap: 12px;
  }
  .search-btn-row {
    justify-content: center;
    margin: 18px 0 8px 0;
    gap: 24px;
  }

  .search-row {
    display: flex;
    align-items: flex-end;
    gap: 24px;
    margin-bottom: 18px;
  }
  .search-textarea {
    flex: 1;
    min-height: 48px;
    font-size: 15px;
    resize: vertical;
  }
  .search-btn-group {
    display: flex;
    flex-direction: row;
    gap: 12px;
    align-items: center;
    justify-content: center;
  }
  @media (max-width: 900px) {
    .search-row {
      flex-direction: column;
      gap: 12px;
    }
    .search-btn-group {
      flex-direction: row;
      gap: 12px;
    }
  }
  @media (max-width: 1200px) {
    .main-controls {
      flex-direction: column;
      align-items: stretch;
      gap: 16px;
    }
    .filter-section {
      justify-content: center;
    }
    .search-section {
      justify-content: center;
    }
  }
  @media (max-width: 768px) {
    .filter-section {
      flex-direction: column;
      gap: 12px;
    }
    .search-section {
      flex-direction: column;
      gap: 12px;
    }
    .timeline-controls {
      flex-direction: column;
      gap: 16px;
    }
    .timeline-info {
      justify-content: center;
    }
    .timeline-bar {
      margin: 0;
    }
    .playback-controls,
    .action-buttons {
      justify-content: center;
    }
  }
  .image-preview-row {
    margin: 10px 0 18px 0;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .image-thumb {
    width: 64px;
    height: 64px;
    object-fit: cover;
    border-radius: 8px;
    border: 2px solid #00ccff;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px 0 rgba(0, 204, 255, 0.3);
  }
  .image-thumb:hover {
    box-shadow: 0 4px 16px 0 #00eaff99;
  }
  .image-tip {
    color: #00eaff;
    font-size: 13px;
    opacity: 0.8;
  }
  .image-large {
    border-radius: 12px;
    box-shadow: 0 4px 32px 0 #00eaff33;
  }
  
  /* 覆盖Element Plus默认样式 */
  :deep(.el-button--primary) {
    --el-button-bg-color: #00ccff !important;
    --el-button-border-color: #00ccff !important;
    --el-button-hover-bg-color: #33d6ff !important;
    --el-button-hover-border-color: #33d6ff !important;
    --el-button-active-bg-color: #0099cc !important;
    --el-button-active-border-color: #0099cc !important;
    --el-button-text-color: #ffffff !important;
    --el-button-hover-text-color: #ffffff !important;
    background: linear-gradient(135deg, rgba(0, 102, 255, 0.9), rgba(0, 204, 255, 0.8)) !important;
    border: none !important;
    box-shadow: 
      0 4px 16px rgba(0, 153, 255, 0.3),
      0 0 0 1px rgba(0, 204, 255, 0.3) inset,
      0 0 20px rgba(0, 204, 255, 0.1) !important;
    color: #ffffff !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.5) !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    position: relative;
    overflow: hidden;
    z-index: 1;
  }
  
  :deep(.el-button--primary)::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, #0066ff, #00ccff, #0066ff);
    z-index: -1;
    background-size: 400%;
    animation: borderGlow 8s linear infinite;
    filter: blur(8px);
    opacity: 0.5;
    border-radius: 10px;
  }
  
  @keyframes borderGlow {
    0% {
      background-position: 0% 50%;
    }
    50% {
      background-position: 100% 50%;
    }
    100% {
      background-position: 0% 50%;
    }
  }
  
  /* 修改输入框和选择框样式 */
  :deep(.el-input__wrapper),
  :deep(.el-textarea__inner),
  :deep(.el-select .el-input__wrapper) {
    background: rgba(0, 30, 60, 0.2) !important;
    border: 1px solid rgba(0, 153, 255, 0.3) !important;
    box-shadow: 0 0 0 1px rgba(0, 153, 255, 0.1) inset !important;
    border-radius: 8px !important;
    backdrop-filter: blur(5px) !important;
  }
  
  :deep(.el-input__wrapper:hover),
  :deep(.el-textarea__inner:hover),
  :deep(.el-select .el-input__wrapper:hover) {
    border-color: rgba(0, 204, 255, 0.5) !important;
    box-shadow: 0 0 0 1px rgba(0, 204, 255, 0.2) inset !important;
  }
  
  :deep(.el-input__wrapper.is-focus),
  :deep(.el-textarea__inner:focus),
  :deep(.el-select .el-input__wrapper.is-focus) {
    border-color: rgba(0, 204, 255, 0.8) !important;
    box-shadow: 0 0 0 1px rgba(0, 204, 255, 0.3) inset, 0 0 10px rgba(0, 204, 255, 0.2) !important;
  }
  
  :deep(.el-input__inner),
  :deep(.el-textarea__inner) {
    color: #ffffff !important;
    background: transparent !important;
  }
  
  :deep(.el-input__inner::placeholder),
  :deep(.el-textarea__inner::placeholder) {
    color: rgba(255, 255, 255, 0.5) !important;
  }
  
  :deep(.el-button--primary:hover) {
    background: linear-gradient(135deg, rgba(0, 102, 255, 1), rgba(0, 204, 255, 0.9)) !important;
    box-shadow: 0 6px 20px rgba(0, 153, 255, 0.5) !important;
    transform: translateY(-2px);
  }
  
  :deep(.el-button--primary:active) {
    background: linear-gradient(135deg, rgba(0, 82, 204, 1), rgba(0, 184, 235, 0.9)) !important;
    box-shadow: 0 2px 10px rgba(0, 153, 255, 0.4) !important;
    transform: translateY(0);
  }
  
  :deep(.el-slider__runway) {
    background-color: rgba(0, 153, 255, 0.15) !important;
  }
  
  :deep(.el-slider__bar) {
    background-color: #00ccff !important;
    background: linear-gradient(90deg, #0066ff, #00ccff) !important;
  }
  
  :deep(.el-slider__button) {
    border-color: #00ccff !important;
    background-color: #ffffff !important;
    box-shadow: 0 0 10px rgba(0, 204, 255, 0.5) !important;
  }
  
  :deep(.el-slider__button:hover) {
    transform: scale(1.2) !important;
    box-shadow: 0 0 15px rgba(0, 204, 255, 0.7) !important;
  }
  </style>
  