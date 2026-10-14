# 索引选型实验结果

## 先导：向量模长

- 语料向量模长：min=1.0000 max=1.0000 mean=1.0000 std=0.0000
- 是否归一化：是（L2/IP/COSINE 排序理论等价）

## 真实数据（135 条）

| 配置 | recall@10 | 平均延迟(ms) | P95(ms) | 建索引(ms) |
|---|---|---|---|---|
| FLAT L2（精确基准） | 100.0% | 7.21 | 8.49 | 536.4 |
| IVF_FLAT L2 nlist=8 nprobe=1 | 100.0% | 6.35 | 7.47 | 545.2 |
| IVF_FLAT L2 nlist=8 nprobe=4 | 100.0% | 6.99 | 10.08 | 527.8 |
| IVF_FLAT L2 nlist=8 nprobe=8 | 100.0% | 6.24 | 7.21 | 528.9 |
| IVF_FLAT L2 nlist=32 nprobe=1 | 100.0% | 6.75 | 8.50 | 530.4 |
| IVF_FLAT L2 nlist=32 nprobe=4 | 100.0% | 6.69 | 8.14 | 529.1 |
| IVF_FLAT L2 nlist=32 nprobe=8 | 100.0% | 6.17 | 7.13 | 538.4 |
| IVF_FLAT L2 nlist=32 nprobe=16 | 100.0% | 6.35 | 7.85 | 543.0 |
| IVF_FLAT L2 nlist=128 nprobe=1 | 100.0% | 6.20 | 7.33 | 532.0 |
| IVF_FLAT L2 nlist=128 nprobe=4 | 100.0% | 6.35 | 7.85 | 537.2 |
| IVF_FLAT L2 nlist=128 nprobe=8 | 100.0% | 6.66 | 8.33 | 532.6 |
| IVF_FLAT L2 nlist=128 nprobe=16 | 100.0% | 6.91 | 8.49 | 529.6 |
| IVF_FLAT L2 nlist=32 nprobe=8 | 100.0% | 6.81 | 8.73 | 530.4 |
| IVF_FLAT IP nlist=32 nprobe=8 | 100.0% | 6.19 | 6.89 | 528.6 |
| IVF_FLAT COSINE nlist=32 nprobe=8 | 100.0% | 6.50 | 8.06 | 525.4 |
| HNSW M=16 efC=200 ef=16 | 100.0% | 6.95 | 7.98 | 535.4 |
| HNSW M=16 efC=200 ef=64 | 100.0% | 6.24 | 6.97 | 531.4 |
| HNSW M=16 efC=200 ef=128 | 100.0% | 6.99 | 8.23 | 530.7 |

## 合成压力集（5000 条）

| 配置 | recall@10 | 平均延迟(ms) | P95(ms) | 建索引(ms) |
|---|---|---|---|---|
| FLAT L2（精确基准） | 100.0% | 10.05 | 11.32 | 532.1 |
| IVF_FLAT L2 nlist=32 nprobe=1 | 57.5% | 6.73 | 7.58 | 4600.2 |
| IVF_FLAT L2 nlist=32 nprobe=4 | 83.5% | 9.00 | 13.02 | 3567.3 |
| IVF_FLAT L2 nlist=32 nprobe=8 | 97.5% | 11.35 | 13.43 | 3582.5 |
| IVF_FLAT L2 nlist=32 nprobe=16 | 100.0% | 10.69 | 16.04 | 4585.4 |
| IVF_FLAT L2 nlist=32 nprobe=32 | 100.0% | 12.75 | 17.21 | 4080.6 |
| IVF_FLAT L2 nlist=128 nprobe=1 | 37.5% | 9.10 | 11.32 | 4602.5 |
| IVF_FLAT L2 nlist=128 nprobe=4 | 80.5% | 9.69 | 13.52 | 4591.5 |
| IVF_FLAT L2 nlist=128 nprobe=8 | 90.5% | 7.61 | 8.71 | 4610.4 |
| IVF_FLAT L2 nlist=128 nprobe=16 | 98.0% | 8.96 | 10.72 | 4084.2 |
| IVF_FLAT L2 nlist=128 nprobe=32 | 100.0% | 9.93 | 11.28 | 4594.4 |
| IVF_SQ8 nlist=128 nprobe=16 | 98.0% | 9.19 | 10.13 | 3567.6 |
| IVF_PQ nlist=128 m=16 nbits=8 | 29.0% | 7.53 | 8.70 | 9262.4 |
| IVF_FLAT L2 nlist=128 nprobe=16 | 98.0% | 9.88 | 11.15 | 4628.6 |
| IVF_FLAT IP nlist=128 nprobe=16 | 92.5% | 10.03 | 14.38 | 4595.2 |
| IVF_FLAT COSINE nlist=128 nprobe=16 | 100.0% | 9.21 | 13.16 | 4598.4 |
| HNSW M=16 efC=200 ef=16 | 94.0% | 8.45 | 10.44 | 4591.9 |
| HNSW M=16 efC=200 ef=64 | 100.0% | 7.88 | 11.13 | 4595.7 |
| HNSW M=16 efC=200 ef=128 | 100.0% | 8.98 | 11.58 | 4582.6 |
| HNSW M=16 efC=200 ef=256 | 100.0% | 8.65 | 9.95 | 4583.2 |

