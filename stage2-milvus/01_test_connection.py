"""
测试 Milvus 连接（stage2 第一课）
Milvus 冷启动要几十秒到一两分钟，连不上会自动重试
"""
import sys
import time
import warnings

# GBK 控制台下打印 ✓/✗ 会直接让程序崩掉，降级为 ? 即可
sys.stdout.reconfigure(errors="replace")
# pymilvus 3.x 对 ORM 风格 API 有弃用提示，学习阶段先静音
warnings.filterwarnings("ignore", message=r".*ORM-style.*")

from pymilvus import connections, utility

def test_connection(max_retries=10, retry_delay=3):
    """测试 Milvus 连接"""
    print("正在测试 Milvus 连接...")

    for attempt in range(1, max_retries + 1):
        try:
            connections.connect(alias="default", host="localhost", port=19530)
            print(f"✓ 成功连接到 Milvus (尝试 {attempt}/{max_retries})")

            server_version = utility.get_server_version()
            print(f"  服务器版本: {server_version}")

            collections = utility.list_collections()
            print(f"  现有集合数: {len(collections)}")
            if collections:
                print(f"  集合列表: {collections}")

            connections.disconnect("default")
            print("\n✅ Milvus 服务正常运行！")
            return True

        except Exception as e:
            print(f"  尝试 {attempt}/{max_retries} 失败: {e}")
            if attempt < max_retries:
                print(f"  等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("\n❌ 无法连接到 Milvus 服务")
                print("排查顺序：")
                print("1. docker ps 看 4 个容器是否都在运行")
                print("2. 刚启动的话再等 30~60 秒（Milvus 就绪慢）")
                print("3. 检查 19530 端口是否被占用")
                return False

if __name__ == "__main__":
    print("=" * 50)
    print("Milvus 连接测试")
    print("=" * 50)
    test_connection()
