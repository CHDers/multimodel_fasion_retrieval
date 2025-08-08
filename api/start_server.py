#!/usr/bin/env python3
"""
时尚多模态检索API服务器启动脚本
"""

import uvicorn
import sys
import os

# 添加项目路径
sys.path.append('/root/autodl-tmp/multimodel_fasion_retrieval')

if __name__ == "__main__":
    print("🚀 启动时尚多模态检索API服务器...")
    print("📍 API文档地址: http://localhost:8000/docs")
    print("📍 健康检查: http://localhost:8000/api/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 