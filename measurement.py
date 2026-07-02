#!/usr/bin/env python3
"""
PHẦN 2 — ĐO LƯỜNG (measurement.py)
================================================================================
Các "probe" đo lường THUẦN + các "runner" nối harness (phần 1) thành pipeline
production để đo từng component và end-to-end.

Ranh giới với experiment_setup.py:
  - Các probe (`measure_latency`, `measure_concurrent`, `measure_torch_peak`)
    KHÔNG biết gì về RAG/Qdrant — chúng chỉ nhận một `callable` và đo. Có thể
    tái dùng cho bất cứ thứ gì.
  - Các runner (`bench_*`) biết harness, dựng đúng closure gọi 1 bước pipeline,
    rồi giao cho probe đo.

CÁC LỖI CỦA SCRIPT CŨ ĐƯỢC SỬA Ở ĐÂY:
  * QPS = 1/latency (sequential)      → measure_concurrent: qps = total/wall_time
  * p99 = latencies[int(n*0.99)]      → numpy.percentile
  * query embedding pre-computed      → bench_embed_query encode TRONG vòng đo
  * "Qdrant(cuda)" (vô nghĩa)         → search/sparse gắn nhãn CPU/server rõ ràng
  * RAM Qdrant = RSS / n_chunks       → estimate_qdrant_ram: công thức cấu trúc
"""
from __future__ import annotations

import gc
import queue
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

# device nào chạy trên gì — để báo cáo trung thực (xem experiment_setup.py header).
# "compute" = phụ thuộc --device (embed/rerank chạy GPU nếu chọn cuda)
# "server"  = luôn chạy trên Qdrant server (CPU), độc lập --device
# "cpu"     = luôn CPU trong tiến trình client (sparse hashing)
PLACEMENT = {
    "embed_query": "compute",
    "sparse_query": "cpu",
    "hybrid_search": "server",
    "rerank": "compute",
    "e2e": "mixed (embed/rerank=compute, search=server, sparse=cpu)",
}


# ---------------------------------------------------------------------------
# Kết quả đo
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """Thống kê latency của một chuỗi lời gọi TUẦN TỰ (single-thread)."""
    component: str
    placement: str
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    total_s: float
    peak_vram_mb: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ConcurrentSample:
    """Thống kê throughput dưới TẢI SONG SONG (đây mới là QPS thật)."""
    component: str
    placement: str
    concurrency: int
    total: int
    wall_s: float
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float
    errors: int

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Probe thuần
# ---------------------------------------------------------------------------

def _maybe_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def measure_torch_peak(device: str) -> Optional[float]:
    """Peak VRAM (MB) kể từ lần reset gần nhất, hoặc None nếu không phải cuda."""
    torch = _maybe_torch()
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None


def _reset_peak(device: str) -> None:
    torch = _maybe_torch()
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _pct(sorted_ms: List[float], q: float) -> float:
    # numpy.percentile (nội suy tuyến tính) — thay cho latencies[int(n*q)] của script cũ.
    return float(np.percentile(sorted_ms, q)) if sorted_ms else 0.0


def _fmt_dur(sec: float) -> str:
    """Định dạng giây → '5s' / '2m30s' / '1h20m' cho dễ đọc."""
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


class _Progress:
    """In tiến độ tại chỗ (ghi đè cùng 1 dòng bằng '\\r') kèm ETA.

    Chỉ in lại mỗi `min_interval` giây để không spam, cộng 1 lần cuối khi xong.
    An toàn đa luồng (dùng cho cả đo song song): gọi update() trong lock hoặc để
    tự khóa nội bộ.
    """

    def __init__(self, label: str, total: int, *, min_interval: float = 1.0,
                 enabled: bool = True):
        self.label = label
        self.total = max(1, total)
        self.min_interval = min_interval
        self.enabled = enabled   # in cả khi pipe ra file (để theo dõi log chạy nền)
        self._done = 0
        self._t0 = time.perf_counter()
        self._last = 0.0
        self._lock = threading.Lock()

    def tick(self, k: int = 1):
        if not self.enabled:
            return
        with self._lock:
            self._done += k
            now = time.perf_counter()
            if now - self._last < self.min_interval and self._done < self.total:
                return
            self._last = now
            self._render(now)

    def _render(self, now: float):
        done, total = self._done, self.total
        elapsed = now - self._t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        pct = 100.0 * done / total
        bar_w = 24
        filled = int(bar_w * done / total)
        bar = "█" * filled + "·" * (bar_w - filled)
        msg = (f"\r      {self.label:<28} [{bar}] {done}/{total} "
               f"({pct:4.0f}%) {rate:6.1f}/s  elapsed {_fmt_dur(elapsed)} "
               f"ETA {_fmt_dur(eta)}   ")
        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self):
        if not self.enabled:
            return
        with self._lock:
            self._render(time.perf_counter())
        sys.stdout.write("\n")
        sys.stdout.flush()


def measure_latency(
    fn: Callable[[], object],
    n: int,
    *,
    component: str,
    placement: str,
    device: str = "cpu",
    warmup: int = 5,
) -> Sample:
    """Gọi `fn()` `n` lần TUẦN TỰ, đo latency từng lần. Warmup không tính giờ.

    Đây là latency "một request, không tải" — trần dưới của độ trễ cảm nhận.
    """
    for _ in range(min(warmup, n)):
        fn()

    gc.collect()
    torch = _maybe_torch()
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    _reset_peak(device)

    prog = _Progress(f"{component} (tuần tự)", n)
    lat: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000.0)
        prog.tick()
    prog.close()

    lat.sort()
    total_s = sum(lat) / 1000.0
    return Sample(
        component=component,
        placement=placement,
        n=n,
        p50_ms=_pct(lat, 50),
        p95_ms=_pct(lat, 95),
        p99_ms=_pct(lat, 99),
        avg_ms=statistics.mean(lat),
        std_ms=statistics.stdev(lat) if len(lat) > 1 else 0.0,
        min_ms=lat[0],
        max_ms=lat[-1],
        total_s=total_s,
        peak_vram_mb=measure_torch_peak(device),
    )


def measure_concurrent(
    make_call: Callable[[int], object],
    total: int,
    concurrency: int,
    *,
    component: str,
    placement: str,
    warmup: int = 5,
) -> ConcurrentSample:
    """Bắn `total` request qua `concurrency` worker song song. QPS = total/wall.

    `make_call(i)` thực thi request thứ i (đóng gói việc chọn query). Mỗi worker
    kéo chỉ số kế tiếp từ một hàng đợi cho tới khi hết → mô phỏng tải đồng thời.

    LƯU Ý TRUNG THỰC: các model (embed/rerank) chia sẻ trong-tiến-trình. Trên GPU,
    CUDA serialize kernel → QPS bão hòa gần 1/serial_latency dù tăng concurrency:
    đó CHÍNH LÀ tín hiệu hạ tầng (cần thêm replica hoặc server-side batching), chứ
    không phải lỗi đo. Qdrant search song song ở SERVER nên QPS scale tới khi CPU
    server bão hòa. So sánh QPS giữa các mức concurrency để thấy điểm bão hòa.
    """
    # warmup
    for i in range(min(warmup, total)):
        make_call(i)

    idx_q: "queue.Queue[int]" = queue.Queue()
    for i in range(total):
        idx_q.put(i)

    lat: List[float] = []
    errors = [0]
    lock = threading.Lock()
    prog = _Progress(f"{component} (song song ×{concurrency})", total)

    def worker():
        local_lat = []
        local_err = 0
        while True:
            try:
                i = idx_q.get_nowait()
            except queue.Empty:
                break
            t0 = time.perf_counter()
            try:
                make_call(i)
            except Exception:  # noqa: BLE001 — đếm lỗi để không im lặng nuốt
                local_err += 1
            local_lat.append((time.perf_counter() - t0) * 1000.0)
            prog.tick()
        with lock:
            lat.extend(local_lat)
            errors[0] += local_err

    t_wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(worker) for _ in range(concurrency)]
        for f in futs:
            f.result()
    wall_s = time.perf_counter() - t_wall0
    prog.close()

    lat.sort()
    qps = total / wall_s if wall_s > 0 else 0.0
    return ConcurrentSample(
        component=component,
        placement=placement,
        concurrency=concurrency,
        total=total,
        wall_s=wall_s,
        qps=qps,
        p50_ms=_pct(lat, 50),
        p95_ms=_pct(lat, 95),
        p99_ms=_pct(lat, 99),
        avg_ms=statistics.mean(lat) if lat else 0.0,
        errors=errors[0],
    )


def measure_served(
    produce: Callable[[int], object],
    infer_batch: Callable[[List[object]], List[object]],
    total: int,
    concurrency: int,
    *,
    component: str,
    placement: str,
    max_batch: int = 32,
    batch_wait_ms: float = 5.0,
    warmup: int = 5,
) -> ConcurrentSample:
    """Mô phỏng **1 serving instance** phục vụ qua queue + dynamic batching — đúng
    cách TEI / Triton / vLLM serve model trên GPU.

    Khác hẳn `measure_concurrent` (C thread bắn thẳng vào model → tranh VRAM → OOM
    giả tạo). Ở đây:

        C client thread ──enqueue──▶  request queue  ──▶  1 inference worker
                                                         (gom dynamic batch ≤ max_batch
                                                          trong batch_wait_ms rồi gọi
                                                          `infer_batch` MỘT lượt)

    → không bao giờ để nhiều request tranh cấp phát VRAM cùng lúc (giống serving
    server thật). `infer_batch(items)` phải xử lý cả list và trả list kết quả cùng
    độ dài.

    Đo:
      - latency mỗi request = (enqueue → nhận kết quả) = queue_wait + inference.
      - QPS = total / wall = throughput của MỘT instance serving.
    """
    # warmup: chạy vài request lẻ qua infer_batch để nạp kernel/cấp phát VRAM.
    for i in range(min(warmup, total)):
        try:
            infer_batch([produce(i)])
        except Exception:  # noqa: BLE001
            pass

    req_q: "queue.Queue" = queue.Queue()      # (item, event, result_box)
    idx_q: "queue.Queue[int]" = queue.Queue()
    for i in range(total):
        idx_q.put(i)

    lat: List[float] = []
    lat_lock = threading.Lock()
    errors = [0]
    prog = _Progress(f"{component} (served ×{concurrency}, batch≤{max_batch})", total)

    def client():
        local_lat: List[float] = []
        while True:
            try:
                i = idx_q.get_nowait()
            except queue.Empty:
                break
            ev = threading.Event()
            req_q.put((produce(i), ev))
            t0 = time.perf_counter()
            ev.wait()                       # chờ worker xử lý xong request này
            local_lat.append((time.perf_counter() - t0) * 1000.0)
            prog.tick()
        with lat_lock:
            lat.extend(local_lat)

    def worker():
        while True:
            first = req_q.get()             # block chờ request (hoặc sentinel None)
            if first is None:
                break
            batch = [first]
            deadline = time.perf_counter() + batch_wait_ms / 1000.0
            # gom thêm cho tới khi đủ max_batch hoặc hết thời gian chờ batch.
            while len(batch) < max_batch:
                rem = deadline - time.perf_counter()
                if rem <= 0:
                    break
                try:
                    it = req_q.get(timeout=rem)
                except queue.Empty:
                    break
                if it is None:              # sentinel tới sớm — để lại cho blocking get
                    req_q.put(None)
                    break
                batch.append(it)
            items = [b[0] for b in batch]
            try:
                results = infer_batch(items)
                for (_item, ev), _res in zip(batch, results):
                    ev.set()
            except Exception:  # noqa: BLE001
                errors[0] += len(batch)
                for _item, ev in batch:
                    ev.set()

    t_wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        c_futs = [ex.submit(client) for _ in range(concurrency)]
        w = threading.Thread(target=worker, daemon=True)
        w.start()
        for f in c_futs:
            f.result()                      # tất cả client đã enqueue xong + chờ xong
        req_q.put(None)                     # báo worker dừng sau khi hết queue
        w.join()
    wall_s = time.perf_counter() - t_wall0
    prog.close()

    lat.sort()
    qps = total / wall_s if wall_s > 0 else 0.0
    return ConcurrentSample(
        component=component,
        placement=placement,
        concurrency=concurrency,
        total=total,
        wall_s=wall_s,
        qps=qps,
        p50_ms=_pct(lat, 50),
        p95_ms=_pct(lat, 95),
        p99_ms=_pct(lat, 99),
        avg_ms=statistics.mean(lat) if lat else 0.0,
        errors=errors[0],
    )


# ---------------------------------------------------------------------------
# Component runners — nối ExperimentHarness thành pipeline production
# ---------------------------------------------------------------------------
# Mỗi runner trả (Sample tuần tự, list[ConcurrentSample]) cho các mức concurrency.

def _run_component(
    call_i: Callable[[int], object],
    *,
    component: str,
    placement: str,
    device: str,
    n_seq: int,
    concurrency_levels: List[int],
    n_conc: int,
    warmup: int = 5,
    served: bool = False,
    produce: Optional[Callable[[int], object]] = None,
    infer_batch: Optional[Callable[[List[object]], List[object]]] = None,
    max_batch: int = 32,
    batch_wait_ms: float = 5.0,
):
    """Đo 1 component: latency tuần tự (luôn raw, 1 request/lần) + throughput.

    Phần throughput chọn 1 trong 2 cơ chế:
      - served=True  (GPU): queue + 1 inference worker + dynamic batching (measure_served)
        → đúng cách serving server phục vụ, KHÔNG tranh VRAM → không OOM giả tạo.
      - served=False (CPU / Qdrant-server / sparse): C thread song song thẳng
        (measure_concurrent) — CPU thật sự song song theo core; Qdrant search đã có
        queue riêng ở server nên raw là đúng.
    """
    seq = measure_latency(
        lambda: call_i(_rand_idx()), n_seq,
        component=component, placement=placement, device=device, warmup=warmup,
    )
    conc: List[ConcurrentSample] = []
    for c in concurrency_levels:
        if served and produce is not None and infer_batch is not None:
            conc.append(measure_served(
                produce, infer_batch, n_conc, c,
                component=component, placement=placement,
                max_batch=max_batch, batch_wait_ms=batch_wait_ms, warmup=warmup,
            ))
        else:
            conc.append(measure_concurrent(
                call_i, n_conc, c,
                component=component, placement=placement, warmup=warmup,
            ))
    return seq, conc


# state nhỏ để chọn query ngẫu nhiên ổn định giữa các lần gọi tuần tự
_rng = np.random.default_rng(0)
_n_queries_global = [1]


def _rand_idx() -> int:
    return int(_rng.integers(0, _n_queries_global[0]))


def _set_query_pool(n: int):
    _n_queries_global[0] = max(1, n)


def _e2e_one(h, q):
    """Chạy trọn pipeline production cho 1 query (dùng cho e2e sequential + served)."""
    dv = h.encode_query_dense(q)
    sv = h.encode_query_sparse(q)
    hits = h.hybrid_search(dv, sv)
    texts = [hit["content"] for hit in hits]
    if texts:
        scores = h.rerank(q, texts)
        order = np.argsort(scores)[::-1][: h.top_k]
        return [hits[j] for j in order]
    return []


def _rerank_batched(h, items):
    """items = list của (query, list_doc_texts). Gộp toàn bộ (q,d) thành 1 lượt
    predict (dynamic batching đúng nghĩa serving), rồi chia score lại từng request."""
    flat = []
    sizes = []
    for q, docs in items:
        for d in docs:
            flat.append((q, d))
        sizes.append(len(docs))
    scores = h.reranker.predict(flat) if flat else []
    out, k = [], 0
    for s in sizes:
        out.append(scores[k:k + s])
        k += s
    return out


def bench_embed_query(h, queries, *, device, n_seq, concurrency_levels, n_conc,
                      served=False, max_batch=32, batch_wait_ms=5.0):
    """Đo encode 1 query (dense) ONLINE — chi phí GPU/CPU thật mỗi truy vấn."""
    _set_query_pool(len(queries))
    call = lambda i: h.encode_query_dense(queries[i % len(queries)])
    produce = lambda i: queries[i % len(queries)]
    infer_batch = lambda items: h.embedder.encode(
        items, batch_size=max(1, len(items)), normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    return _run_component(call, component="embed_query", placement=PLACEMENT["embed_query"],
                          device=device, n_seq=n_seq, concurrency_levels=concurrency_levels,
                          n_conc=n_conc, served=served, produce=produce, infer_batch=infer_batch,
                          max_batch=max_batch, batch_wait_ms=batch_wait_ms)


def bench_sparse_query(h, queries, *, device, n_seq, concurrency_levels, n_conc):
    """Đo HashingVectorizer 1 query — luôn CPU (nhẹ, thường không phải bottleneck)."""
    _set_query_pool(len(queries))
    call = lambda i: h.encode_query_sparse(queries[i % len(queries)])
    return _run_component(call, component="sparse_query", placement=PLACEMENT["sparse_query"],
                          device="cpu", n_seq=n_seq, concurrency_levels=concurrency_levels, n_conc=n_conc)


def bench_hybrid_search(h, queries, *, device, n_seq, concurrency_levels, n_conc):
    """Đo Qdrant hybrid search (RRF) THUẦN — encode đã tính TRƯỚC, ngoài vòng đo,
    để cô lập chi phí server. Chạy trên Qdrant server (CPU)."""
    _set_query_pool(len(queries))
    dense = [h.encode_query_dense(q) for q in queries]
    sparse = [h.encode_query_sparse(q) for q in queries]
    call = lambda i: h.hybrid_search(dense[i % len(queries)], sparse[i % len(queries)])
    return _run_component(call, component="hybrid_search", placement=PLACEMENT["hybrid_search"],
                          device="cpu", n_seq=n_seq, concurrency_levels=concurrency_levels, n_conc=n_conc)


def bench_rerank(h, queries, *, device, n_seq, concurrency_levels, n_conc,
                 served=False, max_batch=32, batch_wait_ms=5.0):
    """Đo cross-encoder rerank `candidates` cặp (q, doc) THẬT lấy từ hybrid search
    (đúng phân phối đầu vào reranker gặp trong production)."""
    _set_query_pool(len(queries))
    # Lấy candidate thật cho từng query (một lần, ngoài vòng đo).
    cand_texts: List[List[str]] = []
    for q in queries:
        dv = h.encode_query_dense(q)
        sv = h.encode_query_sparse(q)
        hits = h.hybrid_search(dv, sv)
        cand_texts.append([hit["content"] for hit in hits] or [q])
    call = lambda i: h.rerank(queries[i % len(queries)], cand_texts[i % len(queries)])
    produce = lambda i: (queries[i % len(queries)], cand_texts[i % len(queries)])
    infer_batch = lambda items: _rerank_batched(h, items)
    return _run_component(call, component="rerank", placement=PLACEMENT["rerank"],
                          device=device, n_seq=n_seq, concurrency_levels=concurrency_levels,
                          n_conc=n_conc, served=served, produce=produce, infer_batch=infer_batch,
                          max_batch=max_batch, batch_wait_ms=batch_wait_ms)


def bench_e2e(h, queries, *, device, n_seq, concurrency_levels, n_conc,
              served=False, max_batch=32, batch_wait_ms=5.0):
    """Đo TOÀN pipeline 1 query như production: encode dense+sparse → hybrid RRF →
    rerank → top_k. Đây là latency người dùng thực sự cảm nhận."""
    _set_query_pool(len(queries))
    call = lambda i: _e2e_one(h, queries[i % len(queries)])
    produce = lambda i: queries[i % len(queries)]
    # e2e qua queue + 1 worker (không gộp GPU跨 request vì pipeline pha search là IO);
    # worker xử lý tuần tự → 1 inference 1 lúc → không tranh VRAM. max_batch chỉ giới
    # hạn số request worker lấy mỗi lượt (vẫn chạy lần lượt trong infer_batch).
    def infer_batch(items):
        return [_e2e_one(h, q) for q in items]
    return _run_component(call, component="e2e", placement=PLACEMENT["e2e"],
                          device=device, n_seq=n_seq, concurrency_levels=concurrency_levels,
                          n_conc=n_conc, served=served, produce=produce, infer_batch=infer_batch,
                          max_batch=max(1, min(max_batch, 4)), batch_wait_ms=batch_wait_ms)


# ---------------------------------------------------------------------------
# Ước lượng RAM Qdrant — CÔNG THỨC CẤU TRÚC (không dùng RSS)
# ---------------------------------------------------------------------------

# Hệ số nén theo quantization (so với fp32 raw vectors).
_QUANT_FACTOR = {None: 1.0, "none": 1.0, "scalar": 0.25, "binary": 1.0 / 32.0}


def estimate_qdrant_ram(
    dim: int,
    n_chunks: int,
    *,
    hnsw_m: int = 16,
    quantization: Optional[str] = None,
    sparse_avg_nnz: int = 0,
) -> Dict[str, float]:
    """Ước lượng RAM Qdrant theo CẤU TRÚC dữ liệu, không phải RSS/n.

    Thành phần (bytes):
      raw_vectors : n × dim × 4               (fp32; ×0.25 scalar, ×1/32 binary)
      hnsw_graph  : n × m × 2 × 4  (~= n·m·8)  (mỗi node ~2m liên kết id 4B; layer 0)
      sparse      : n × avg_nnz × (4 idx + 4 val)   (nếu dùng hybrid; ~= n·nnz·8)

    Lưu ý: khi bật quantization, Qdrant vẫn có thể GIỮ raw vectors trên disk và
    chỉ nạp bản nén vào RAM (always_ram=True). Ước lượng RAM ở đây tính bản NÉN
    cho phần vector; raw fp32 coi như ở disk. Cross-check với collection_info().
    """
    factor = _QUANT_FACTOR.get(quantization, 1.0)
    raw_bytes_full = n_chunks * dim * 4
    vec_ram_bytes = raw_bytes_full * factor
    hnsw_bytes = n_chunks * hnsw_m * 2 * 4
    sparse_bytes = n_chunks * sparse_avg_nnz * 8
    total = vec_ram_bytes + hnsw_bytes + sparse_bytes

    mb = 1024 * 1024
    return {
        "dim": dim,
        "n_chunks": n_chunks,
        "quantization": quantization or "none",
        "dense_raw_fp32_mb": raw_bytes_full / mb,
        "dense_in_ram_mb": vec_ram_bytes / mb,
        "hnsw_graph_mb": hnsw_bytes / mb,
        "sparse_mb": sparse_bytes / mb,
        "total_ram_mb": total / mb,
        "total_ram_gb": total / mb / 1024,
        "per_chunk_bytes": total / n_chunks if n_chunks else 0.0,
    }


def extrapolate_ram(dim: int, *, hnsw_m: int, quantization: Optional[str],
                    sparse_avg_nnz: int, points=(100_000, 1_000_000, 10_000_000)) -> List[Dict]:
    """Ngoại suy RAM cho các mốc quy mô lớn — dùng CÙNG công thức cấu trúc."""
    return [
        estimate_qdrant_ram(dim, n, hnsw_m=hnsw_m, quantization=quantization,
                            sparse_avg_nnz=sparse_avg_nnz)
        for n in points
    ]
