"""pytest 路径配置：让测试能 import 各阶段的服务模块"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "stage2-milvus"))
sys.path.insert(0, os.path.join(ROOT, "stage3-rag-service"))
