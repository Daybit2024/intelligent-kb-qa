"""
领域表迁移（一次性，幂等可重跑）：
1. 建 domains 表并写入现有三个领域（finance / law / ai）
2. documents / doc_chunks 加 domain 列（已存在则跳过）
3. 按来源回填存量数据的 domain

运行:  python 03_migrate_domains.py
"""
import sys

sys.stdout.reconfigure(errors="replace")

import pymysql

import milvus_config as config
import store

DEFAULT_DOMAINS = [
    ("finance", "全球宏观经济、金融市场与行业分析"),
    ("law", "法律服务行业发展趋势与实务"),
    ("ai", "人工智能产业发展与应用趋势"),
]

BACKFILL = {
    "ai": ["ai_dev.md", "中国人工智能应用发展报告(2025).pdf"],
    "finance": ["fin.md", "金融分析报告.pdf"],
    "law": ["law.md", "法律服务行业.pdf"],
}


def main():
    conn = pymysql.connect(
        host=config.MYSQL_HOST, port=config.MYSQL_PORT,
        user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DATABASE, charset="utf8mb4",
    )
    cur = conn.cursor()

    # 1. 领域表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS domains (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(50) NOT NULL UNIQUE COMMENT '领域标识（英文小写）',
            description VARCHAR(500) COMMENT '领域说明（供 LLM 归类与意图识别使用）',
            file_count INT DEFAULT 0 COMMENT '该领域的文件数（由 refresh_domain_counts 重算维护）',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB
    """)
    for name, desc in DEFAULT_DOMAINS:
        cur.execute(
            "INSERT INTO domains (name, description) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE description=VALUES(description)",
            (name, desc),
        )
    print("✓ domains 表就绪（含 3 个存量领域）")

    # 2. 加列（幂等）
    for table in ["documents", "doc_chunks"]:
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN domain VARCHAR(50) DEFAULT 'general'")
            print(f"✓ {table}: 已添加 domain 列")
        except pymysql.err.OperationalError as e:
            if "Duplicate column" in str(e):
                print(f"  {table}: domain 列已存在，跳过")
            else:
                raise
    # domains 表补充 file_count 列（早于该字段创建的 domains 表需要）
    try:
        cur.execute("ALTER TABLE domains ADD COLUMN file_count INT DEFAULT 0 COMMENT '该领域的文件数（重算维护）'")
        print("✓ domains: 已添加 file_count 列")
    except pymysql.err.OperationalError as e:
        if "Duplicate column" in str(e):
            print("  domains: file_count 列已存在，跳过")
        else:
            raise

    # 3. 回填存量数据
    for domain, files in BACKFILL.items():
        ph = ",".join(["%s"] * len(files))
        cur.execute(f"UPDATE doc_chunks SET domain=%s WHERE source IN ({ph})", [domain] + files)
        n1 = cur.rowcount
        cur.execute(f"UPDATE documents SET domain=%s WHERE filename IN ({ph})", [domain] + files)
        n2 = cur.rowcount
        print(f"✓ 回填 {domain}: doc_chunks {n1} 行 / documents {n2} 行")
    conn.commit()

    # 4. 重算领域文件数（归零领域自动清除）
    store.refresh_domain_counts()

    # 5. 验证
    cur.execute("SELECT domain, COUNT(*) FROM doc_chunks GROUP BY domain")
    print("\ndoc_chunks 领域分布:", cur.fetchall())
    cur.execute("SELECT name, description, file_count FROM domains ORDER BY id")
    print("领域表（含文件数）:")
    for r in cur.fetchall():
        print(f"  {r[0]}（{r[2]} 个文件）: {r[1]}")
    conn.close()


if __name__ == "__main__":
    main()
