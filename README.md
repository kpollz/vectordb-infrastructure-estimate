# RAG Infrastructure Benchmark — Qdrant hybrid (chat-with-documents)

Bộ công cụ đo đạc để **estimate hạ tầng** cho hệ thống RAG dùng vector database,
được tinh chỉnh để **bám sát 100% pipeline production** của dự án
[`chat-with-documents`](../chat-with-documents). Mục tiêu: chạy trên workstation
(CPU và/hoặc GPU) để trả lời các câu hỏi hạ tầng:

- Một truy vấn tốn bao nhiêu ms ở mỗi khâu (embed / sparse / search / rerank)?
- GPU nhanh hơn CPU bao nhiêu lần cho embed và rerank?
- Một node phục vụ được bao nhiêu **QPS** trước khi bão hòa? Nghẽn ở đâu?
- Qdrant tốn bao nhiêu **RAM** ở 100K / 1M / 10M chunk? Quantization giúp gì?
- Ingest (index tài liệu) chạy nhanh cỡ nào?

> **Vì sao viết lại từ `rag_benchmark_qdrant_v2.py`?** Script cũ đo *một hệ RAG
> chung chung* chứ không phải hệ thật: nó dùng thư viện `rank_bm25` (không phải
> sparse-vector native của Qdrant), Qdrant `:memory:` (không phải server), model
> khác production, search dense-only (không hybrid RRF), pre-compute query
> embedding (bỏ sót chi phí online), tính "QPS" kiểu tuần tự (`1/latency`), và
> ước lượng RAM bằng `RSS / n_chunks` (sai hàng chục lần). Xem
> [§8 — Khác biệt so với script cũ](#8-khác-biệt-so-với-script-cũ).

---

## Mục lục

1. [Triết lý thiết kế: tách bạch SETUP và ĐO LƯỜNG](#1-triết-lý-thiết-kế-tách-bạch-setup-và-đo-lường)
2. [Pipeline production được mô phỏng](#2-pipeline-production-được-mô-phỏng)
3. [Cài đặt & chạy](#3-cài-đặt--chạy)
4. [Cơ chế đo lường — chi tiết](#4-cơ-chế-đo-lường--chi-tiết)
5. [Ước lượng RAM Qdrant](#5-ước-lượng-ram-qdrant)
6. [Đọc kết quả & sanity-check](#6-đọc-kết-quả--sanity-check)
7. [Từ số đo → estimate hạ tầng](#7-từ-số-đo--estimate-hạ-tầng)
8. [Khác biệt so với script cũ](#8-khác-biệt-so-với-script-cũ)
9. [Landscape công nghệ RAG retrieval](#9-landscape-công-nghệ-rag-retrieval)

---

## 1. Triết lý thiết kế: tách bạch SETUP và ĐO LƯỜNG

Yêu cầu cốt lõi: **"tách bạch phần tính toán đo lường và phần set up thí nghiệm"**.
Vì vậy code chia thành 3 module, mỗi module một trách nhiệm:

```
┌─────────────────────────┐   ┌──────────────────────────┐   ┌────────────────────┐
│  experiment_setup.py    │   │   measurement.py         │   │  run_benchmark.py  │
│  ── PHẦN 1: SET UP ──    │   │   ── PHẦN 2: ĐO ──        │   │  ── CLI GLUE ──     │
│                         │   │                          │   │                    │
│  class ExperimentHarness│   │  Probe THUẦN (không biết │   │  parse args        │
│   • generate_corpus()   │──▶│  gì về RAG):             │◀──│  điều phối vòng đời │
│   • load_models()       │   │   • measure_latency()    │   │  in báo cáo        │
│   • provision_qdrant()  │   │   • measure_concurrent() │   │  dump JSON         │
│   • ingest()            │   │   • measure_torch_peak() │   │                    │
│   • encode_query_*()    │   │                          │   │                    │
│   • hybrid_search()     │   │  Runner (nối harness):   │   │                    │
│   • rerank()            │   │   • bench_embed_query()  │   │                    │
│                         │   │   • bench_hybrid_search()│   │                    │
│  KHÔNG đo thời gian ở    │   │   • bench_rerank()       │   │                    │
│  đây (trừ ingest stats) │   │   • bench_e2e()          │   │                    │
│                         │   │   • estimate_qdrant_ram()│   │                    │
└─────────────────────────┘   └──────────────────────────┘   └────────────────────┘
```

Lợi ích cụ thể của ranh giới này:

- **`measurement.py` không import `experiment_setup`.** Ba probe thuần
  (`measure_latency`, `measure_concurrent`, `measure_torch_peak`) chỉ nhận một
  `callable` và đo — có thể tái dùng cho bất kỳ hàm nào, kể cả ngoài dự án này.
- **Đổi thí nghiệm không đụng code đo, và ngược lại.** Muốn thêm model, đổi schema
  Qdrant, đổi cách sinh corpus → sửa `experiment_setup.py`. Muốn đổi cách tính
  percentile, thêm kiểu tải (ví dụ ramp-up thay vì đồng thời) → sửa
  `measurement.py`. Hai trục độc lập.
- **Kiểm thử/đọc dễ.** Mỗi file đọc được độc lập; runner trong `measurement.py` là
  chỗ *duy nhất* biết cách ghép một bước pipeline với một probe.

---

## 2. Pipeline production được mô phỏng

Mọi tham số dưới đây **lấy verbatim từ source `chat-with-documents`** (không phỏng
đoán). Đây là "ground truth" mà benchmark tái tạo:

| Khâu | Giá trị production | File nguồn |
|---|---|---|
| **Embedding** | `sentence-transformers/all-MiniLM-L12-v2`, dim=384, `normalize_embeddings=True` | `app/config.py`, `app/store/embeddings.py` |
| **Reranker** | `BAAI/bge-reranker-v2-m3`, `CrossEncoder.predict([(q,d)…])` | `app/config.py`, `app/store/rerank.py` |
| **Sparse ("bm25")** | `HashingVectorizer(n_features=2¹⁶, alternate_sign=False, norm="l2", lowercase=True, ngram_range=(1,1), dtype=float32)` | `app/store/sparse.py` |
| **Vector DB** | Qdrant **server v1.12.4**, collection có dense `"dense"`(384, COSINE) + sparse `"bm25"`, HNSW mặc định | `app/store/vector.py`, `deploy/docker-compose.yml` |
| **Search** | `query_points(prefetch=[dense 20, sparse 20], query=FusionQuery(Fusion.RRF), limit)` | `app/store/vector.py::search_hybrid` |
| **Chunk** | `CHUNK_MAX_TOKENS=512` (token = whitespace split), có prefix "heading breadcrumb" | `app/config.py`, `app/ingestion/chunk.py` |
| **Luồng query** | encode dense+sparse **online** → hybrid RRF → rerank `candidates` → lấy `top_k=5` | `app/inference/tools.py` |

### ⚠️ Điều quan trọng nhất về hạ tầng: `--device` chỉ áp dụng cho embed + rerank

`deploy/docker-compose.gpu.yml` cho thấy: **Qdrant chạy trong container riêng dùng
CPU**; chỉ `api` và `worker` (nơi chạy embedding + rerank) mới được cấp GPU. Do đó
trong benchmark:

| Component | Chạy ở đâu | Cột `placement` trong báo cáo |
|---|---|---|
| `embed_query` | GPU nếu `--device cuda`, ngược lại CPU | `compute` |
| `rerank` | GPU nếu `--device cuda`, ngược lại CPU | `compute` |
| `sparse_query` | **Luôn CPU** (HashingVectorizer trong tiến trình client) | `cpu` |
| `hybrid_search` | **Luôn Qdrant server** (CPU container) | `server` |
| `e2e` | Trộn cả ba | `mixed` |

Nếu ai đó đọc số "Qdrant chạy trên cuda" thì đó là **vô nghĩa** — Qdrant không dùng
GPU trong kiến trúc này. Benchmark ghi rõ điều đó để không ai estimate nhầm.

---

## 3. Cài đặt & chạy

### Bước 1 — Bật Qdrant server (đúng version production)

```bash
docker compose -f docker/docker-compose.bench.yml up -d qdrant
# kiểm tra:  curl http://localhost:6333/healthz
```

`docker/docker-compose.bench.yml` dùng `qdrant/qdrant:v1.12.4` — **cùng image với
production**. Đo trên server thật (không phải `:memory:`) để tính cả chi phí
gRPC/HTTP serialization, disk/mmap, và concurrency handling của server — những thứ
tạo ra cost thật mà chế độ in-process giấu đi.

### Bước 2 — Cài phụ thuộc (khớp `requirements.txt` của chat-with-documents)

```bash
pip install qdrant-client==1.12.1 sentence-transformers==3.3.1 \
            scikit-learn torch psutil numpy
```

> GPU: cài `torch` bản CUDA phù hợp máy (xem https://pytorch.org). Lần chạy đầu sẽ
> tải weights model từ HuggingFace (~2GB cho bge-reranker-v2-m3) — cần mạng.

### Bước 3 — Chạy benchmark

```bash
# (a) Smoke test nhanh trên CPU — kiểm tra pipeline thông suốt
python run_benchmark.py --docs 1000 --queries 50 --concurrency 1 4

# (b) Đo đầy đủ trên GPU (embed+rerank GPU; Qdrant vẫn CPU/server)
python run_benchmark.py --device cuda --docs 50000 --queries 200 \
       --concurrency 1 4 8 16

# (c) So sánh CPU vs GPU: chạy 2 lần, đổi --device, cùng --docs
python run_benchmark.py --device cpu  --docs 50000 --output cpu.json
python run_benchmark.py --device cuda --docs 50000 --output cuda.json

# (d) Phân tích xu hướng scale (nhiều mốc docs)
python run_benchmark.py --device cuda --scale-test \
       --scale-points 1000 10000 100000

# (e) Đo tác động quantization lên RAM/latency
python run_benchmark.py --docs 50000 --quantization scalar
python run_benchmark.py --docs 50000 --quantization binary

# (f) So sánh model khác (đa ngữ) mà không đổi code
python run_benchmark.py --embed-model BAAI/bge-m3 \
       --rerank-model BAAI/bge-reranker-v2-m3 --docs 10000
```

### Các flag chính

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--qdrant-url` | `http://localhost:6333` | Địa chỉ Qdrant server |
| `--device` | `cpu` | `cpu`/`cuda` — **chỉ** cho embed+rerank |
| `--docs` | `10000` | Số chunk (khi không scale-test) |
| `--chunk-tokens` | `512` | Số token/chunk (= `CHUNK_MAX_TOKENS`) |
| `--queries` | `200` | Số query đo tuần tự / kích thước pool |
| `--concurrency` | `1 4 8 16` | Các mức song song để đo QPS |
| `--scale-test` | off | Chạy qua nhiều `--scale-points` |
| `--quantization` | `none` | `none`/`scalar`/`binary` |
| `--embed-model` / `--rerank-model` | production | Override để so sánh |
| `--candidates` / `--top-k` | `20` / `5` | Y hệt `HYBRID_CANDIDATES` / `RETRIEVAL_TOP_K` |
| `--output` | `benchmark_results.json` | File JSON kết quả |

### Cấu hình bằng `.env` (thay cho gõ cờ mỗi lần)

Mọi cờ đều có biến môi trường `BENCH_*` tương ứng. Chép mẫu rồi sửa:

```bash
cp .env.example .env      # sửa .env theo máy bạn
python run_benchmark.py   # tự nạp .env, không cần --flag
```

Thứ tự ưu tiên: **cờ CLI > biến shell > `.env` > default trong code**. Ví dụ
`.env` đặt `BENCH_DEVICE=cuda` nhưng chạy `--device cpu` thì CPU thắng. `.env`
thật bị `.gitignore` bỏ qua (chỉ commit `.env.example`). Đổi file:
`BENCH_ENV_FILE=prod.env python run_benchmark.py`. Xem `.env.example` để biết đủ
biến (gồm cả `HF_HOME`, `HF_HUB_OFFLINE`, `CUDA_VISIBLE_DEVICES`).

---

## 4. Cơ chế đo lường — chi tiết

### 4.1 Hai loại số đo, đừng lẫn lộn

| | **Latency tuần tự** | **Throughput song song (QPS)** |
|---|---|---|
| Hàm | `measure_latency` | `measure_concurrent` |
| Câu hỏi trả lời | "1 request khi *không tải* mất bao lâu?" | "Node chịu được bao nhiêu req/s?" |
| Cách đo | Gọi `fn()` `n` lần lần lượt, đo từng lần | `concurrency` worker bắn `total` request đồng thời |
| QPS tính sao | *(không tính QPS ở đây)* | `total / wall_clock_time` |
| Dùng để | Đặt SLA độ trễ (p50/p95/p99) | Sizing số node/replica |

> **Vì sao không lấy `1/latency` làm QPS?** Vì đó là throughput của **một luồng
> duy nhất**. Server thật phục vụ nhiều request chồng lấn; QPS thật chỉ đo được khi
> bắn tải đồng thời và chia cho *wall-clock time*. Script cũ mắc đúng lỗi này.

### 4.2 `measure_latency` — chi tiết cơ chế

```
warmup (5 lần, KHÔNG tính giờ)   ← nạp cache, JIT, cấp phát VRAM, kết nối
   ↓
gc.collect() + cuda.empty_cache() + reset_peak_memory_stats()
   ↓
lặp n lần:  t0 = perf_counter(); fn(); lat.append(perf_counter()-t0)
   ↓
sort → p50/p95/p99 bằng numpy.percentile (nội suy tuyến tính)
```

- **Warmup bắt buộc.** Lần gọi đầu của model gánh chi phí một lần (cudaMalloc, tải
  kernel, autotune cuDNN). Không warmup → p99 nhiễu nặng.
- **`perf_counter`** (đồng hồ đơn điệu độ phân giải cao), không phải `time.time()`.
- **Percentile bằng `numpy.percentile`**, không phải `lat[int(n*0.99)]`. Với n=100
  thì `int(100*0.99)=99` = **max**, không phải p99 thực — script cũ sai chỗ này.
- **Peak VRAM** đọc từ `torch.cuda.max_memory_allocated()` sau reset.

### 4.3 `measure_concurrent` — chi tiết cơ chế

```
warmup
   ↓
đổ 0..total-1 vào queue.Queue
   ↓
ThreadPoolExecutor(concurrency): mỗi worker
      while queue còn: i = get_nowait(); đo make_call(i); (đếm lỗi nếu có)
   ↓
wall_s = thời gian tường (bao trùm mọi worker)
qps = total / wall_s
```

- Mô hình tải: **closed-loop, N worker luôn bận** — kéo request kế tiếp ngay khi
  xong. Đây là mô hình "N client đồng thời, không nghỉ".
- **Lỗi được đếm** (`errors`), không nuốt im lặng — nếu Qdrant timeout hay OOM,
  bạn thấy ngay trong cột `errors`.

### 4.4 ⚠️ Đọc đúng QPS song song: GIL, CUDA, và điểm bão hòa

Đây là điểm dễ hiểu sai nhất. `measure_concurrent` dùng **thread** (không phải
process), nên hành vi khác nhau theo component — và **sự khác nhau đó chính là phát
hiện hạ tầng**:

- **`hybrid_search` (Qdrant server):** thread client chỉ gửi request rồi *chờ IO*
  → nhả GIL. Công việc thật chạy song song **ở server**. Vì vậy QPS **tăng theo
  concurrency** cho tới khi CPU của container Qdrant bão hòa. Điểm QPS ngừng tăng
  = giới hạn của một node Qdrant → cần thêm core / shard / replica.

- **`embed_query`, `rerank` (model trong tiến trình):** nhiều thread gọi cùng một
  model. Trên GPU, CUDA **serialize** các kernel; phần Python cũng vướng GIL. Nên
  QPS **gần như không tăng** khi thêm thread — nó chạm trần `≈ 1/serial_latency`.
  **Đây không phải lỗi đo** — nó phản ánh đúng rằng: muốn tăng throughput
  embed/rerank thì phải **thêm replica tiến trình** hoặc **batch phía server**
  (ví dụ Triton, TEI, vLLM), chứ không phải thêm thread trong một tiến trình.

  > Muốn đo *trần throughput batch* của GPU (khác với "nhiều client đơn lẻ"), tăng
  > batch khi encode/predict. Benchmark này cố ý đo **đơn-query online** vì đó là
  > cái pipeline chat thực sự làm cho mỗi câu hỏi người dùng.

#### 4.4.1 Chế độ "served" — sửa lỗi OOM giả tạo khi đo GPU

Raw multi-thread bắn thẳng vào model khiến nhiều inference tranh cấp phát VRAM
cùng lúc → `OutOfMemoryError` ở conc cao (đặc biệt card nhỏ 4–8 GB). Nhưng đó
**không phải cách production phục vụ**: production chạy model trong **1 serving
instance có queue + dynamic batching** (TEI / Triton / vLLM).

Vì vậy, khi `--device cuda`, các component chạy model (`embed_query`, `rerank`,
`e2e`) tự переключ sang chế độ **served**: `concurrency` client enqueue request
vào 1 hàng đợi, **1 inference worker** gom dynamic batch (`--max-batch`, trong
`--batch-wait-ms`) rồi gọi model **một lượt** → không bao giờ tranh VRAM.

- Bật/tắt: `--served auto` (mặc định: on khi cuda, off khi cpu) / `on` / `off`.
- `hybrid_search` và `sparse_query` luôn raw: Qdrant đã có queue riêng ở server,
  CPU sparse thật sự song song theo core.
- Kết quả: cột `errors` của embed/rerank/e2e trên GPU giờ **gần 0** thay vì nổ ở
  conc 8/16; QPS phản ánh **throughput thật của 1 serving instance**.

> Nếu bạn serve model bằng **TEI/Triton thật**, hãy benchmark bắn thẳng vào HTTP
> của server đó (không qua chế độ served in-proc này) — đó là con số chính xác
> nhất cho estimate hạ tầng. Chế độ served in-proc là mô phỏng gần đúng khi chưa
  có serving server riêng.

### 4.5 Các runner nối pipeline (`bench_*`)

Mỗi runner dựng một closure gọi **đúng một bước** pipeline production rồi giao cho
probe. Điểm tinh tế ở việc *đặt cái gì trong/ngoài vòng đo*:

| Runner | Đo cái gì | Chuẩn bị TRƯỚC vòng đo (không tính giờ) |
|---|---|---|
| `bench_embed_query` | `encode([q])` online | — (encode **trong** vòng đo, đúng như prod) |
| `bench_sparse_query` | `HashingVectorizer.transform([q])` | — |
| `bench_hybrid_search` | `query_points(...)` RRF thuần | Encode sẵn dense+sparse của mọi query (để **cô lập** chi phí server, không lẫn embed) |
| `bench_rerank` | `predict([(q,doc)×candidates])` | Lấy sẵn candidate **thật** từ hybrid search (đúng phân phối đầu vào reranker) |
| `bench_e2e` | encode→sparse→search→rerank→top_k | — (đo trọn như người dùng cảm nhận) |

> `bench_embed_query` cố tình encode **trong** vòng đo (script cũ pre-compute nên
> bỏ sót toàn bộ chi phí này). `bench_hybrid_search` thì cố tình encode **ngoài**
> vòng đo — vì ở đây ta muốn tách riêng chi phí *server*, không muốn nó lẫn chi phí
> embedding. `bench_e2e` mới là con số phản ánh độ trễ end-to-end thật.

---

## 5. Ước lượng RAM Qdrant

`estimate_qdrant_ram()` tính RAM theo **cấu trúc dữ liệu**, không phải đo RSS process
rồi chia số chunk (cách cũ trộn lẫn RAM của Python/model/numpy và over-count
overhead cố định khi ngoại suy).

```
dense_in_ram = n × dim × 4 bytes × factor      # fp32; factor: none=1, scalar=0.25, binary=1/32
hnsw_graph  ≈ n × m × 2 × 4 bytes              # ~2m liên kết id 4B mỗi node (layer 0)
sparse      ≈ n × avg_nnz × 8 bytes            # (4B index + 4B value) mỗi phần tử khác 0
total = dense_in_ram + hnsw_graph + sparse
```

- `avg_nnz` (số token phân biệt/chunk) được **đo thật** từ corpus mẫu, không đoán.
- Kết quả có `points_count` do **Qdrant server báo về** để cross-check công thức.
- Ngoại suy 100K / 1M / 10M chunk dùng **cùng công thức** (tuyến tính theo cấu
  trúc, không ngoại suy overhead cố định).

**Ví dụ** (dim=384, 1M chunk, không quantization): dense ≈ 1.46 GB + hnsw ≈ 0.12 GB
+ sparse (nnz≈400) ≈ 3 GB ≈ **~4.6 GB**. Với `scalar` quantization: dense giảm còn
~0.37 GB. Đây là con số dùng được để chọn RAM cho node — khác hẳn kiểu `RSS/n ×
1e6` ra hàng nghìn GB của script cũ.

> Lưu ý: đây là RAM cho *vector + graph + sparse index*. Payload (text chunk) trong
> setup này để ở RAM theo mặc định; production thật có thể đẩy payload xuống disk
> (`on_disk_payload`) — điều chỉnh theo cấu hình thật của bạn.

---

## 6. Đọc kết quả & sanity-check

Báo cáo in ra 4 khối cho mỗi mốc quy mô: **latency tuần tự**, **throughput song
song**, **ingest**, và **RAM**. File JSON (`--output`) chứa toàn bộ để vẽ đồ thị.

Sau khi chạy, **đối chiếu với các kỳ vọng** dưới đây — lệch nhiều nghĩa là có gì đó
sai (sai máy, thiếu warmup, Qdrant nghẽn, v.v.):

- [ ] **RAM Qdrant @ 50K × 384d** ≈ vài trăm MB, **KHÔNG** phải hàng GB/TB.
- [ ] **GPU nhanh hơn CPU** cho `embed_query` và `rerank` (thường 5–30×). Nếu
      không, kiểm tra `torch.cuda.is_available()` và model có thật sự lên GPU.
- [ ] **`rerank` là khâu chậm nhất** trong `e2e` (bge-reranker-v2-m3 là cross-encoder
      nặng, chấm 20 cặp/query).
- [ ] **`hybrid_search` p50** tăng **rất chậm** (dưới-tuyến-tính, ~log n) khi tăng
      số chunk — đặc trưng HNSW.
- [ ] **QPS `hybrid_search`** tăng theo concurrency tới điểm bão hòa CPU server;
      **QPS `embed`/`rerank`** phẳng dần (xem §4.4).
- [ ] **`errors` = 0** ở mọi mức concurrency. Nếu > 0: tăng `QDRANT_TIMEOUT` hoặc
      giảm concurrency.
- [ ] **Ingest**: `embed` chiếm phần lớn thời gian; GPU >> CPU ở khâu này.

---

## 7. Từ số đo → estimate hạ tầng

Cách quy đổi số benchmark thành đề xuất hạ tầng cụ thể:

1. **RAM node Qdrant** = `total_ram_gb` ở quy mô chunk mục tiêu (§5) + ~30–50% dự
   phòng cho tăng trưởng/optimizer/segment. Chọn instance có RAM ≥ mức đó.
2. **Số node/replica embed+rerank** = `QPS_mục_tiêu / QPS_bão_hòa_1_tiến_trình`
   (lấy từ khối throughput, §4.4). Vì embed/rerank không scale theo thread, đây là
   phép chia theo *tiến trình/replica*, không phải theo core.
3. **GPU hay CPU cho embed/rerank?** So `cpu.json` vs `cuda.json`. Nếu QPS mục tiêu
   thấp và latency CPU vẫn đạt SLA → CPU rẻ hơn. Nếu cần latency thấp hoặc QPS cao →
   GPU (một GPU thay cho nhiều node CPU).
4. **SLA độ trễ** đặt theo **p95/p99 của `e2e`** ở mức concurrency dự kiến, không
   phải p50 tuần tự (quá lạc quan).
5. **Quantization**: nếu RAM là ràng buộc, chạy `--quantization scalar` và so
   latency/RAM. `scalar` thường gần như miễn phí về chất lượng, giảm 4× RAM vector.

---

## 8. Khác biệt so với script cũ

`rag_benchmark_qdrant_v2.py` được **giữ lại làm tham chiếu** nhưng đã bị thay thế.
Bảng dưới là mọi điểm đã sửa để bám thực tế:

| # | Script cũ | Bộ mới |
|---|---|---|
| 1 | BM25 qua thư viện `rank_bm25` (O(N)/query, không phải inverted index) | **HashingVectorizer → Qdrant native sparse** (đúng prod) |
| 2 | Qdrant `:memory:` (in-process, không server/network/disk) | **Qdrant server v1.12.4** qua `--qdrant-url` |
| 3 | Model `all-MiniLM-L6` / `bge-*-en` | **`all-MiniLM-L12-v2` + `bge-reranker-v2-m3`** (đúng prod) |
| 4 | Search **dense-only** | **Hybrid dense+sparse fuse RRF** (`query_points`) |
| 5 | Query embedding **pre-compute** (bỏ sót chi phí online) | **Encode trong vòng đo** cho `embed_query`/`e2e` |
| 6 | "QPS" = `n / Σlatency` (tuần tự) | **QPS = total/wall** dưới tải song song thật |
| 7 | RAM = `RSS / n_chunks` (sai hàng chục lần) | **Công thức cấu trúc** `dim·4·n + hnsw + sparse` |
| 8 | Nhãn "Qdrant(cuda)" (vô nghĩa) | **`placement` rõ ràng**; `--device` chỉ embed+rerank |
| 9 | Chunk 256 từ ngẫu nhiên | **512 whitespace-token + heading breadcrumb** (đúng prod) |
| 10 | p99 = `lat[int(n·0.99)]` (= max khi n=100) | **`numpy.percentile`** (nội suy) |
| 11 | Rerank trên `random.sample(docs)` | **Candidate thật từ hybrid search** |
| 12 | `try/except` nuốt lỗi trong vòng đo | Warmup sạch; **đếm lỗi** ở đo song song |

---

## 9. Landscape công nghệ RAG retrieval

Bối cảnh các công nghệ trong tầng retrieval của RAG (trả lời câu hỏi #1 ban đầu),
kèm ghi chú GPU/CPU và lựa chọn cho tiếng Việt:

| Tầng | Đại diện | GPU/CPU | Độ phức tạp query | Ghi chú |
|---|---|---|---|---|
| **Sparse / lexical** | BM25, BM25+/BM25L, TF-IDF, **SPLADE** (learned sparse), HashingVectorizer (dự án này) | CPU (RAM-bound) | inverted index + WAND/MaxScore ≈ O(postings) | Engine thật: Elasticsearch/Lucene, OpenSearch, Tantivy, Vespa, **Qdrant sparse**. SPLADE train bằng GPU |
| **Dense (bi-encoder)** | sBERT, **BGE**, E5, GTE, MiniLM; late-interaction **ColBERTv2** | Cả hai (GPU tăng tốc encode) | O(1) theo n_docs (chi phí ở ANN) | Model đa ngữ (VN): **bge-m3**, multilingual-e5-large, paraphrase-multilingual-MiniLM |
| **Re-rank** | cross-encoder ms-marco, **bge-reranker-(base/large/v2-m3)**, mxbai, jina, Cohere rerank; **RankGPT/RankT5** (LLM) | Cả hai (GPU thắng lớn) | O(top_k) — độc lập n_docs | **v2-m3 đa ngữ** — thường là bottleneck pipeline |
| **Hybrid fusion** | **RRF** (dự án này dùng), DBSF, weighted, learned fusion | — | — | Kết hợp semantic + keyword; đây là cái production chạy |
| **Vector DB / ANN** | **Qdrant**, Milvus, Weaviate, pgvector, Vespa, OpenSearch kNN | CPU (thường) | HNSW ≈ O(ef·log n); IVF-PQ / DiskANN | Quantization (scalar/PQ/binary) là đòn bẩy giảm RAM chính |

Dự án `chat-with-documents` hiện dùng: **MiniLM-L12-v2** (dense) + **HashingVectorizer**
(sparse) + **RRF** (fusion) + **bge-reranker-v2-m3** (rerank) trên **Qdrant HNSW**.
Đây chính xác là stack mà benchmark này đo.

---

## Cấu trúc thư mục

```
vectordb-infrastructure-estimate/
├── README.md                      # tài liệu này
├── experiment_setup.py            # PHẦN 1 — set up thí nghiệm (ExperimentHarness)
├── measurement.py                 # PHẦN 2 — đo lường thuần + runner + RAM estimate
├── run_benchmark.py               # CLI điều phối + báo cáo + JSON
├── docker/
│   └── docker-compose.bench.yml   # Qdrant v1.12.4 (đúng version prod)
└── rag_benchmark_qdrant_v2.py     # script cũ — GIỮ LÀM THAM CHIẾU (đã bị thay thế)
```
