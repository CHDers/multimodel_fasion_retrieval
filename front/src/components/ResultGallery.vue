<template>
  <div style="margin-top: 32px;">
    <el-empty v-if="!loading && (!results || results.length === 0)" description="暂无检索结果" />
    <el-row v-else :gutter="24">
      <el-col v-for="item in pagedResults" :key="item.id" :span="8" style="margin-bottom: 24px;">
        <el-card shadow="hover" style="border-radius: 12px; overflow: hidden;">
          <div style="position: relative;">
            <img :src="item.image_url" alt="result" style="width: 100%; height: 260px; object-fit: cover; border-radius: 8px;" />
            <div class="caption-overlay">
              <div class="caption-main">{{ item.caption || '无描述' }}</div>
              <div class="caption-sub">
                <span>类型: {{ item.product_type || '-' }}</span>
                <span style="margin-left: 12px;">性别: {{ item.gender || '-' }}</span>
                <span style="margin-left: 12px;">得分: {{ item.similarity}}</span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
    <div v-if="results && results.length > pageSize" style="text-align: center; margin-top: 16px;">
      <el-pagination
        background
        layout="prev, pager, next"
        :total="results.length"
        :page-size="pageSize"
        v-model:current-page="currentPage"
      />
    </div>
    <el-skeleton v-if="loading" :rows="2" animated style="margin-top: 32px;" />
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
const props = defineProps({
  results: Array,
  loading: Boolean
})
const pageSize = 6
const currentPage = ref(1)

const pagedResults = computed(() => {
  if (!props.results) return []
  const start = (currentPage.value - 1) * pageSize
  return props.results.slice(start, start + pageSize)
})

watch(() => props.results, () => {
  currentPage.value = 1
})
</script>

<style scoped>
.caption-overlay {
  position: absolute;
  left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.45);
  color: #fff;
  padding: 12px 16px 8px 16px;
  border-radius: 0 0 8px 8px;
  font-size: 15px;
}
.caption-main {
  font-weight: bold;
  font-size: 16px;
  margin-bottom: 2px;
}
.caption-sub {
  font-size: 13px;
  opacity: 0.92;
}
</style> 