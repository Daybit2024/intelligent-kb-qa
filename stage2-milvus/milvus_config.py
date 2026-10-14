"""
全局配置：Milvus 向量库 + MySQL 内容库
新架构：正文存 MySQL，Milvus 只负责"id → 向量"的相似度检索
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Milvus 配置 ----------
# 注意：统一用 127.0.0.1 而非 localhost——Windows 下 localhost 会先试 IPv6(::1)，
# 对只监听 IPv4 的本地服务会先卡约 2 秒超时再回退（实测踩坑，服务间调用务必 127.0.0.1）
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
COLLECTION_NAME = "ai_knowledge_base"
EMBEDDING_DIM = 1024  # text-embedding-v4 的向量维度

# 索引配置（选型依据见 exp_index_results.md：
# 向量已归一化，L2/IP/COSINE 等价 → 保持 L2
# IVF_SQ8：压力集 98% 召回（与 IVF_FLAT 持平），内存约 1/4、建索引更快；PQ 压缩损失过大弃用）
INDEX_TYPE = "IVF_SQ8"
METRIC_TYPE = "L2"
NPROBE = 16   # 检索时扫描的簇数（压力集实验：nprobe=16 是 98% 召回拐点）
NLIST = 128   # 建索引时的聚类数（当前数据量下无害，为库增长留空间）

SEARCH_PARAMS = {
    "metric_type": METRIC_TYPE,
    "params": {"nprobe": NPROBE},
}

# ---------- 阿里云百炼（生成向量用） ----------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ---------- MySQL 配置（内容库） ----------
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
MYSQL_DATABASE = "rag"
MYSQL_TABLE = "doc_chunks"

# ---------- Elasticsearch 配置（关键词检索） ----------
ES_HOST = "127.0.0.1"
ES_PORT = 9200
ES_INDEX = "doc_chunks"   # 索引只存 id + text_seg + source（正文回 MySQL 取）

# ---------- 重排配置（百炼重排模型） ----------
RERANK_MODEL = "gte-rerank-v2"
RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

# ---------- 路径与登记表 ----------
DOC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
MYSQL_DOC_TABLE = "documents"   # 文件登记表（sha256 查重 / 增量入库用）

# ---------- 窗口切分参数（无标题文本：pdf / txt） ----------
WINDOW_TARGET_CHARS = 500     # 目标段长（字符）
WINDOW_OVERLAP_RATIO = 0.15   # 相邻段重叠比例

# ---------- 检索默认参数 ----------
DEFAULT_TOP_K = 10
