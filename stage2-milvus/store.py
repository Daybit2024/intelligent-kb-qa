"""
共享数据层：MySQL（正本）/ Milvus（向量索引）/ Elasticsearch（关键词索引）
- MySQL 是唯一正本；Milvus 和 ES 都是派生索引，可随时从 MySQL 重建
- 供 ingest.py / rebuild_indexes.py / query_engine.py 共同使用
"""
import hashlib
import os
import time
import warnings

import jieba
import pymysql
from elasticsearch import Elasticsearch
from openai import OpenAI
from pymilvus import (
    connections, utility, Collection,
    FieldSchema, CollectionSchema, DataType,
)
import milvus_config as config

warnings.filterwarnings("ignore", message=r".*ORM-style.*")


# ==================== MySQL（正本） ====================

def mysql_conn(with_db=True):
    return pymysql.connect(
        host=config.MYSQL_HOST, port=config.MYSQL_PORT,
        user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DATABASE if with_db else None,
        charset="utf8mb4",
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def ensure_schema():
    """建 documents 登记表；首次运行自动登记存量来源（迁移）"""
    conn = mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {config.MYSQL_DOC_TABLE} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(500) NOT NULL UNIQUE COMMENT '文件名（唯一标识）',
                    file_hash CHAR(64) NOT NULL COMMENT '文件内容 sha256',
                    format VARCHAR(20) COMMENT 'md / pdf / txt',
                    chunk_count INT DEFAULT 0,
                    status VARCHAR(20) DEFAULT 'done' COMMENT 'processing / done / failed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            """)
            cur.execute(f"SELECT COUNT(*) FROM {config.MYSQL_DOC_TABLE}")
            if cur.fetchone()[0] == 0:
                # 存量迁移：doc_chunks 里已有来源但还没登记过 → 自动补登记
                cur.execute(f"SELECT source, COUNT(*) FROM {config.MYSQL_TABLE} GROUP BY source")
                for filename, count in cur.fetchall():
                    path = os.path.join(config.DOC_DIR, filename)
                    fhash = sha256_file(path) if os.path.exists(path) else "unknown"
                    cur.execute(
                        f"INSERT INTO {config.MYSQL_DOC_TABLE} (filename, file_hash, format, chunk_count, status) "
                        f"VALUES (%s, %s, %s, %s, 'done')",
                        (filename, fhash, os.path.splitext(filename)[1].lstrip("."), count),
                    )
                    print(f"  [迁移] 已登记存量文件: {filename}（{count} 段）")
        conn.commit()
    finally:
        conn.close()


def get_document(cur, filename):
    """查登记表：返回 (file_hash, status, chunk_count) 或 None"""
    cur.execute(
        f"SELECT file_hash, status, chunk_count FROM {config.MYSQL_DOC_TABLE} WHERE filename=%s",
        (filename,),
    )
    return cur.fetchone()


def upsert_document(cur, filename, file_hash, fmt, chunk_count, status="done"):
    cur.execute(
        f"INSERT INTO {config.MYSQL_DOC_TABLE} (filename, file_hash, format, chunk_count, status) "
        f"VALUES (%s, %s, %s, %s, %s) "
        f"ON DUPLICATE KEY UPDATE file_hash=VALUES(file_hash), format=VALUES(format), "
        f"chunk_count=VALUES(chunk_count), status=VALUES(status)",
        (filename, file_hash, fmt, chunk_count, status),
    )


def purge_source(filename):
    """按来源清理某文件的所有痕迹：MySQL 删行 → Milvus 删向量 → ES 删文档"""
    conn = mysql_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {config.MYSQL_TABLE} WHERE source=%s", (filename,))
        ids = [r[0] for r in cur.fetchall()]
        cur.execute(f"DELETE FROM {config.MYSQL_TABLE} WHERE source=%s", (filename,))
    conn.commit()
    conn.close()

    if ids:
        col = milvus_collection()
        col.delete(expr=f"id in {ids}")
        connections.disconnect("default")

    es = es_client()
    try:
        es.delete_by_query(
            index=config.ES_INDEX,
            query={"term": {"source": filename}},
            refresh=True,
        )
    except Exception:
        pass  # 索引还不存在时忽略
    return len(ids)


# ==================== Milvus（向量索引） ====================

def milvus_collection(load=True):
    connections.connect(alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    col = Collection(config.COLLECTION_NAME)
    if load:
        col.load()
    return col


def ensure_milvus_collection(recreate=False):
    """建集合 + 索引；recreate=True 先删后建（全量重建用）"""
    connections.connect(alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    exists = utility.has_collection(config.COLLECTION_NAME)
    if recreate and exists:
        utility.drop_collection(config.COLLECTION_NAME)
        exists = False
        print(f"已删除旧集合 {config.COLLECTION_NAME}")

    if not exists:
        schema = CollectionSchema(
            [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM),
            ],
            description="RAG 知识库向量（正文存 MySQL，按 id 回表）",
        )
        col = Collection(config.COLLECTION_NAME, schema)
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": config.INDEX_TYPE,
                "metric_type": config.METRIC_TYPE,
                "params": {"nlist": config.NLIST},
            },
        )
        print(f"✓ 集合 {config.COLLECTION_NAME} 已创建（{config.INDEX_TYPE} 索引）")
    else:
        col = Collection(config.COLLECTION_NAME)
    col.load()
    return col


# ==================== Elasticsearch（关键词索引） ====================

def es_client():
    return Elasticsearch(f"http://{config.ES_HOST}:{config.ES_PORT}")


def ensure_es_index(es):
    if es.indices.exists(index=config.ES_INDEX):
        return
    es.indices.create(
        index=config.ES_INDEX,
        mappings={
            "properties": {
                "id": {"type": "integer"},
                "text_seg": {"type": "text", "analyzer": "whitespace"},
                "source": {"type": "keyword"},
            }
        },
    )
    print(f"✓ ES 索引 {config.ES_INDEX} 已创建（whitespace 分析器）")


def segment(text):
    """jieba 分词 → 空格连接（作为 ES whitespace 分析器的输入）"""
    return " ".join(jieba.cut(text))


# ==================== 向量化（阿里云百炼） ====================

_embedding_client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.EMBEDDING_BASE_URL,
)


def embed_texts(texts, batch=10):
    """批量生成向量，单批失败自动重试"""
    vectors = []
    for start in range(0, len(texts), batch):
        chunk_batch = texts[start:start + batch]
        for attempt in range(3):
            try:
                resp = _embedding_client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=chunk_batch,
                    dimensions=config.EMBEDDING_DIM,
                    encoding_format="float",
                )
                vectors.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                print(f"  向量化批次 {start} 第 {attempt + 1} 次失败: {e}")
                time.sleep(2)
        else:
            raise RuntimeError(f"批次 {start} 向量化失败")
    return vectors
