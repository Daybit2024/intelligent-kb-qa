"""
派生索引重建工具：从 MySQL（正本）全量重建 ES / Milvus 两个派生索引
用法：
  python rebuild_indexes.py --es      重建 ES 关键词索引（秒级，无 API 费用）
  python rebuild_indexes.py --milvus  重建 Milvus 向量索引（重新调 embedding，会产生费用）
首次运行会顺带创建 documents 登记表并自动登记存量文件
"""
import argparse
import sys

sys.stdout.reconfigure(errors="replace")

from elasticsearch.helpers import bulk
from pymilvus import connections

import store
import milvus_config as config


def rebuild_es():
    print("=" * 50)
    print("重建 ES 关键词索引（数据源：MySQL）")
    print("=" * 50)

    conn = store.mysql_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, text, source FROM {config.MYSQL_TABLE} ORDER BY id")
        rows = cur.fetchall()
    conn.close()
    print(f"从 MySQL 读到 {len(rows)} 条文段")

    es = store.es_client()
    if es.indices.exists(index=config.ES_INDEX):
        es.indices.delete(index=config.ES_INDEX)
        print(f"已删除旧索引 {config.ES_INDEX}")
    store.ensure_es_index(es)

    actions = (
        {
            "_index": config.ES_INDEX,
            "_id": r[0],
            "id": r[0],
            "text_seg": store.segment(r[1]),
            "source": r[2],
        }
        for r in rows
    )
    ok, errors = bulk(es, actions, chunk_size=100, refresh=True, raise_on_error=False)
    count = es.count(index=config.ES_INDEX)["count"]
    print(f"✓ ES 索引完成：写入 {ok} 条，当前文档数 = {count}")
    if errors:
        print(f"⚠ {len(errors)} 条报错（前 2 条）：{errors[:2]}")


def rebuild_milvus():
    print("=" * 50)
    print("重建 Milvus 向量索引（数据源：MySQL，会重新调用 embedding 产生费用）")
    print("=" * 50)

    conn = store.mysql_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, text FROM {config.MYSQL_TABLE} ORDER BY id")
        rows = cur.fetchall()
    conn.close()
    print(f"从 MySQL 读到 {len(rows)} 条文段，开始向量化...")

    ids = [r[0] for r in rows]
    vectors = store.embed_texts([r[1] for r in rows])

    col = store.ensure_milvus_collection(recreate=True)
    col.insert([ids, vectors])
    col.flush()
    col.load()
    print(f"✓ Milvus 重建完成，实体数 = {col.num_entities}")
    connections.disconnect("default")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 MySQL 重建派生索引")
    parser.add_argument("--es", action="store_true", help="重建 ES 关键词索引")
    parser.add_argument("--milvus", action="store_true", help="重建 Milvus 向量索引")
    args = parser.parse_args()

    if not args.es and not args.milvus:
        parser.print_help()
        sys.exit(0)

    store.ensure_schema()  # 建 documents 登记表 + 存量自动登记
    if args.es:
        rebuild_es()
    if args.milvus:
        rebuild_milvus()
    print("\n全部完成")
