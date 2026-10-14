"""
stage2 第二课：建库建表
1. MySQL：建数据库 rag + 建表 doc_chunks（存正文和元数据）
2. Milvus：建集合 ai_knowledge_base（只存 id + 向量）
脚本可重复执行：已存在就跳过，不会清数据
"""
import sys
import warnings

sys.stdout.reconfigure(errors="replace")
warnings.filterwarnings("ignore", message=r".*ORM-style.*")

import pymysql
from pymilvus import (
    connections, utility, Collection,
    FieldSchema, CollectionSchema, DataType,
)
import milvus_config as config


def setup_mysql():
    print("=" * 50)
    print("[1/2] MySQL：建库建表（存正文）")
    print("=" * 50)

    # 连接时不指定库（库可能还不存在）
    conn = pymysql.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            # 建库：utf8mb4 才存得下中文
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS {config.MYSQL_DATABASE} "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            print(f"✓ 数据库 {config.MYSQL_DATABASE} 就绪")

            # 建表：正文 + 元数据，id 自增主键（要与 Milvus 保持一致）
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {config.MYSQL_DATABASE}.{config.MYSQL_TABLE} (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键，与 Milvus 集合中的 id 一致',
                    text TEXT NOT NULL COMMENT '文段正文',
                    source VARCHAR(500) COMMENT '来源文件',
                    section VARCHAR(500) COMMENT '所属章节',
                    keywords VARCHAR(1000) COMMENT 'jieba 提取的关键词（供关键词检索）',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            """)
            print(f"✓ 表 {config.MYSQL_TABLE} 就绪")

            cur.execute(f"SELECT COUNT(*) FROM {config.MYSQL_DATABASE}.{config.MYSQL_TABLE}")
            print(f"  当前行数: {cur.fetchone()[0]}")
        conn.commit()
    finally:
        conn.close()
        print("  MySQL 连接已关闭")


def setup_milvus():
    print()
    print("=" * 50)
    print("[2/2] Milvus：建集合（只存 id + 向量）")
    print("=" * 50)

    connections.connect(alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    print(f"✓ 已连接 Milvus ({config.MILVUS_HOST}:{config.MILVUS_PORT})")

    if utility.has_collection(config.COLLECTION_NAME):
        print(f"  集合 {config.COLLECTION_NAME} 已存在，跳过创建")
        col = Collection(config.COLLECTION_NAME)
    else:
        # 字段1：id —— 关键设计！auto_id=False，id 由 MySQL 分配后原样传进来
        field_id = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
        )
        # 字段2：embedding —— 1024 维浮点向量
        field_embedding = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM
        )
        schema = CollectionSchema(
            fields=[field_id, field_embedding],
            description="RAG 知识库向量（正文存 MySQL，按 id 回表）",
        )
        col = Collection(name=config.COLLECTION_NAME, schema=schema)
        print(f"✓ 集合 {config.COLLECTION_NAME} 已创建（id + embedding 两个字段）")

        # 建索引：IVF_FLAT + L2
        index_params = {
            "index_type": config.INDEX_TYPE,
            "metric_type": config.METRIC_TYPE,
            "params": {"nlist": config.NLIST},
        }
        col.create_index(field_name="embedding", index_params=index_params)
        print(f"✓ 向量索引已创建：{config.INDEX_TYPE} / {config.METRIC_TYPE} / nlist={config.NLIST}")

    # 加载到内存才能被检索
    col.load()
    print("✓ 集合已 load（可检索）")
    print(f"  实体数量: {col.num_entities}")

    connections.disconnect("default")
    print("  Milvus 连接已断开")


if __name__ == "__main__":
    setup_mysql()
    setup_milvus()
    print()
    print("=" * 50)
    print("全部就绪：正文进 MySQL(doc_chunks)，向量进 Milvus(ai_knowledge_base)")
    print("=" * 50)
