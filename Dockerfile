# RAG 应用镜像：检索服务(8000) / RAG 服务(8001) / Web UI(7860) 三合一
# 同一镜像，启动命令由 docker-compose 各自指定
FROM python:3.12-slim

WORKDIR /app

# 依赖单独一层：只改代码不触发重装（用阿里云源加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

COPY . .

CMD ["python", "stage2-milvus/api_server.py"]
