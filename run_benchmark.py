#!/usr/bin/env python3
"""
RUN BENCHMARK (run_benchmark.py) — CLI điều phối
================================================================================
Nối PHẦN 1 (experiment_setup.ExperimentHarness) với PHẦN 2 (measurement.*) để
đo hạ tầng RAG SÁT production `chat-with-documents`, rồi in báo cáo + dump JSON.

CÁCH DÙNG
---------
1) Bật Qdrant server (đúng version prod):
     docker compose -f docker/docker-compose.bench.yml up -d qdrant

2) Cài phụ thuộc (khớp requirements của chat-with-documents):
     pip install qdrant-client==1.12.1 sentence-transformers==3.3.1 \
                 scikit-learn torch psutil numpy

3) Chạy:
     # Smoke test nhanh trên CPU
     python run_benchmark.py --docs 1000 --queries 50 --concurrency 1 4

     # Đủ tải trên GPU (embed+rerank chạy GPU; Qdrant vẫn CPU/server)
     python run_benchmark.py --device cuda --docs 50000 --queries 200 \
            --concurrency 1 4 8 16

     # Phân tích xu hướng scale
     python run_benchmark.py --device cuda --scale-test

LƯU Ý HẠ TẦNG: --device CHỈ áp dụng cho embedding + rerank (khớp
docker-compose.gpu.yml: chỉ api/worker được cấp GPU). Qdrant hybrid search chạy
trên server (CPU); sparse HashingVectorizer chạy CPU. Báo cáo ghi rõ cột
"placement" cho từng component.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import measurement as M
from experiment_setup import ExperimentHarness


# ---------------------------------------------------------------------------
# Nạp .env (không cần thư viện ngoài) + helper đọc biến môi trường
# ---------------------------------------------------------------------------

def load_dotenv(path: str = ".env") -> None:
    """Đọc file .env đơn giản (KEY=VALUE mỗi dòng) vào os.environ nếu chưa có.

    KHÔNG ghi đè biến đã tồn tại trong môi trường → biến shell thật vẫn thắng .env.
    Bỏ qua dòng trống và dòng bắt đầu bằng '#'. Không hỗ trợ nội suy/multiline.
    """
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int_list(name: str, default: list) -> list:
    """Đọc list số nguyên, phân tách bởi dấu phẩy hoặc khoảng trắng (vd '1,4,8,16')."""
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    parts = v.replace(",", " ").split()
    try:
        return [int(x) for x in parts] or default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Chạy 1 điểm quy mô (một n_chunks) — trả dict kết quả JSON-serializable
# ---------------------------------------------------------------------------

def run_one_scale(h: ExperimentHarness, n_chunks: int, args) -> dict:
    print(f"\n{'='*72}")
    print(f"SCALE POINT: {n_chunks:,} chunks × {args.chunk_tokens} tokens | device={args.device}")
    print(f"{'='*72}")

    # --- PHẦN 1: setup ---
    print(f"  [setup] Sinh corpus {n_chunks:,} chunks ...")
    docs = h.generate_corpus(n_chunks, args.chunk_tokens)

    print(f"  [setup] Provision Qdrant collection (recreate) ...")
    h.provision_qdrant()

    print(f"  [setup] Ingest (embed + sparse + upsert) ...")
    ingest = h.ingest(docs, batch_size=args.batch_size)
    print(f"          embed={ingest.embed_ms:.0f}ms ({ingest.embed_chunks_per_s:,.0f} ch/s) "
          f"sparse={ingest.sparse_ms:.0f}ms upsert={ingest.upsert_ms:.0f}ms "
          f"| overall {ingest.overall_chunks_per_s:,.0f} ch/s")

    queries = h.generate_queries(args.queries)

    # ước lượng avg nnz của sparse để tính RAM chính xác hơn (≈ số token phân biệt/chunk)
    sample_sp = h.encode_query_sparse(docs[0])
    sparse_avg_nnz = len(sample_sp[0])

    # --- PHẦN 2: đo lường ---
    runners = [
        ("embed_query", M.bench_embed_query),
        ("sparse_query", M.bench_sparse_query),
        ("hybrid_search", M.bench_hybrid_search),
        ("rerank", M.bench_rerank),
        ("e2e", M.bench_e2e),
    ]
    seq_samples = []
    conc_samples = []
    # In trước KHỐI LƯỢNG công việc để không bị "đứng im không biết bao lâu".
    n_conc = max(args.queries, max(args.concurrency) * 10)
    calls_per_comp = args.queries + n_conc * len(args.concurrency)
    # Cơ chế serving: embed/rerank/e2e (chạy model trong tiến trình) dùng "served"
    # (queue + 1 worker + dynamic batching) khi GPU → không tranh VRAM → không OOM
    # giả tạo. sparse/hybrid_search luôn raw (CPU / đã có queue ở Qdrant server).
    served_gpu = (args.served == "on") or (args.served == "auto" and args.device == "cuda")
    served_comps = {"embed_query", "rerank", "e2e"} if served_gpu else set()
    print(f"\n  [measure] Sẽ đo {len(runners)} component. Mỗi component:")
    print(f"            • tuần tự  : {args.queries} lần")
    print(f"            • song song: {n_conc} lần × {len(args.concurrency)} mức "
          f"{args.concurrency}  = {n_conc * len(args.concurrency)} lần")
    print(f"            → {calls_per_comp} call/component. "
          f"rerank & e2e mỗi call chấm {args.candidates} cặp.")
    if served_gpu:
        print(f"            • GPU serving: {sorted(served_comps)} chạy qua queue + 1 worker "
              f"+ dynamic batch≤{args.max_batch} (đúng cơ chế serving, không tranh VRAM).")
    else:
        print(f"            • GPU serving: TẮT (raw thread). Bật bằng --served on (hoặc auto trên cuda).")
    for ci, (name, fn) in enumerate(runners, 1):
        print(f"\n  [measure {ci}/{len(runners)}] {name}")
        kw = dict(device=args.device, n_seq=args.queries,
                  concurrency_levels=args.concurrency, n_conc=n_conc)
        if name in served_comps:
            kw.update(served=True, max_batch=args.max_batch, batch_wait_ms=args.batch_wait_ms)
        seq, conc = fn(h, queries, **kw)
        seq_samples.append(seq)
        conc_samples.extend(conc)

    # --- RAM ước lượng (công thức cấu trúc) + cross-check server ---
    ram_now = M.estimate_qdrant_ram(
        h.embed_dim, n_chunks, hnsw_m=h.hnsw_m,
        quantization=h.quantization, sparse_avg_nnz=sparse_avg_nnz,
    )
    ram_extrap = M.extrapolate_ram(
        h.embed_dim, hnsw_m=h.hnsw_m, quantization=h.quantization,
        sparse_avg_nnz=sparse_avg_nnz,
    )
    try:
        info = h.collection_info()
        server_points = getattr(info, "points_count", None)
    except Exception:  # noqa: BLE001
        server_points = None

    _print_scale_report(n_chunks, seq_samples, conc_samples, ingest, ram_now,
                        ram_extrap, server_points, sparse_avg_nnz)

    return {
        "n_chunks": n_chunks,
        "ingest": ingest.to_dict(),
        "sparse_avg_nnz": sparse_avg_nnz,
        "sequential": [s.to_dict() for s in seq_samples],
        "concurrent": [c.to_dict() for c in conc_samples],
        "ram_estimate": ram_now,
        "ram_extrapolation": ram_extrap,
        "server_points_count": server_points,
    }


# ---------------------------------------------------------------------------
# In báo cáo
# ---------------------------------------------------------------------------

def _print_scale_report(n_chunks, seq_samples, conc_samples, ingest, ram_now,
                        ram_extrap, server_points, sparse_avg_nnz):
    print(f"\n  ── LATENCY TUẦN TỰ (1 request, không tải) — {n_chunks:,} chunks ──")
    print(f"  {'component':<15} {'placement':<10} {'p50(ms)':>9} {'p95(ms)':>9} "
          f"{'p99(ms)':>9} {'avg(ms)':>9} {'VRAM(MB)':>9}")
    for s in seq_samples:
        place = "compute" if s.placement == "compute" else ("server" if s.placement == "server"
                 else ("cpu" if s.placement == "cpu" else "mixed"))
        vram = f"{s.peak_vram_mb:.0f}" if s.peak_vram_mb else "-"
        print(f"  {s.component:<15} {place:<10} {s.p50_ms:>9.2f} {s.p95_ms:>9.2f} "
              f"{s.p99_ms:>9.2f} {s.avg_ms:>9.2f} {vram:>9}")

    print(f"\n  ── THROUGHPUT SONG SONG (QPS thật = total/wall) — {n_chunks:,} chunks ──")
    print(f"  {'component':<15} {'conc':>5} {'QPS':>10} {'p50(ms)':>9} "
          f"{'p99(ms)':>9} {'errors':>7}")
    # nhóm theo component để thấy QPS thay đổi khi tăng concurrency
    by_comp = {}
    for c in conc_samples:
        by_comp.setdefault(c.component, []).append(c)
    for comp, rows in by_comp.items():
        for c in sorted(rows, key=lambda x: x.concurrency):
            print(f"  {comp:<15} {c.concurrency:>5} {c.qps:>10.1f} {c.p50_ms:>9.2f} "
                  f"{c.p99_ms:>9.2f} {c.errors:>7}")

    print(f"\n  ── RAM QDRANT ƯỚC LƯỢNG (công thức cấu trúc, KHÔNG phải RSS) ──")
    print(f"     dim={ram_now['dim']} quant={ram_now['quantization']} "
          f"sparse_nnz≈{sparse_avg_nnz}")
    print(f"     @ {n_chunks:,} chunks : dense_in_ram={ram_now['dense_in_ram_mb']:.1f}MB "
          f"+ hnsw={ram_now['hnsw_graph_mb']:.1f}MB + sparse={ram_now['sparse_mb']:.1f}MB "
          f"= {ram_now['total_ram_mb']:.1f}MB ({ram_now['per_chunk_bytes']:.0f} B/chunk)")
    if server_points is not None:
        print(f"     (Qdrant server báo points_count={server_points:,})")
    print(f"     Ngoại suy:")
    for e in ram_extrap:
        print(f"        {e['n_chunks']:>12,} chunks → {e['total_ram_gb']:.2f} GB "
              f"(dense {e['dense_in_ram_mb']/1024:.2f}GB + hnsw {e['hnsw_graph_mb']/1024:.2f}GB "
              f"+ sparse {e['sparse_mb']/1024:.2f}GB)")


def _print_header(args, h):
    print("=" * 72)
    print("RAG INFRASTRUCTURE BENCHMARK — sát production chat-with-documents")
    print("=" * 72)
    print(f"Qdrant URL   : {args.qdrant_url}")
    print(f"Device       : {args.device}  (CHỈ áp dụng embed+rerank; Qdrant/sparse=CPU)")
    print(f"Embedding    : {h.embed_model} (dim={h.embed_dim})")
    print(f"Reranker     : {h.rerank_model}")
    print(f"Quantization : {h.quantization or 'none'}")
    print(f"HNSW         : m={h.hnsw_m} ef_construct={h.hnsw_ef_construct}")
    print(f"Retrieval    : candidates={h.candidates} top_k={h.top_k}")
    print(f"Chunk tokens : {args.chunk_tokens}")
    print(f"Queries      : {args.queries}")
    print(f"Concurrency  : {args.concurrency}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    from experiment_setup import (
        DEFAULT_CANDIDATES,
        DEFAULT_CHUNK_TOKENS,
        DEFAULT_EMBED_MODEL,
        DEFAULT_RERANK_MODEL,
        DEFAULT_TOP_K,
    )
    p = argparse.ArgumentParser(
        description="RAG infra benchmark (Qdrant hybrid) — production chat-with-documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mọi default đọc từ biến môi trường (xem .env.example) rồi mới tới hằng số.
    # CLI luôn thắng env; env luôn thắng hằng số mặc định. Thứ tự ưu tiên:
    #   CLI flag  >  biến môi trường (BENCH_*)  >  default trong code
    p.add_argument("--qdrant-url", default=_env_str("BENCH_QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--device", choices=["cpu", "cuda"],
                   default=_env_str("BENCH_DEVICE", "cpu"),
                   help="Chỉ ảnh hưởng embed+rerank (Qdrant/sparse luôn CPU)")
    p.add_argument("--docs", type=int, default=_env_int("BENCH_DOCS", 10000),
                   help="Số chunk khi KHÔNG scale-test")
    p.add_argument("--chunk-tokens", type=int,
                   default=_env_int("BENCH_CHUNK_TOKENS", DEFAULT_CHUNK_TOKENS))
    p.add_argument("--queries", type=int, default=_env_int("BENCH_QUERIES", 200),
                   help="Số query đo tuần tự / pool")
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=_env_int_list("BENCH_CONCURRENCY", [1, 4, 8, 16]))
    p.add_argument("--batch-size", type=int, default=_env_int("BENCH_BATCH_SIZE", 256),
                   help="Batch encode khi ingest")
    # --- Cơ chế serving cho GPU (sửa lỗi OOM giả tạo khi raw multi-thread bắn GPU) ---
    p.add_argument("--served", choices=["auto", "on", "off"],
                   default=_env_str("BENCH_SERVED", "auto"),
                   help="on=luôn queue+1 worker+dynamic batch cho embed/rerank/e2e; "
                        "off=raw thread; auto=on khi --device cuda, off khi cpu")
    p.add_argument("--max-batch", type=int, default=_env_int("BENCH_MAX_BATCH", 32),
                   help="Serving: số request tối đa gộp trong 1 lượt inference")
    p.add_argument("--batch-wait-ms", type=float, default=_env_float("BENCH_BATCH_WAIT_MS", 5.0),
                   help="Serving: thời gian chờ gom batch tối đa (ms) trước khi chạy inference")
    p.add_argument("--scale-test", action="store_true",
                   default=_env_bool("BENCH_SCALE_TEST", False))
    p.add_argument("--scale-points", type=int, nargs="+",
                   default=_env_int_list("BENCH_SCALE_POINTS", [1000, 5000, 10000, 50000, 100000]))
    p.add_argument("--quantization", choices=["none", "scalar", "binary"],
                   default=_env_str("BENCH_QUANTIZATION", "none"))
    p.add_argument("--embed-model", default=_env_str("BENCH_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    p.add_argument("--rerank-model", default=_env_str("BENCH_RERANK_MODEL", DEFAULT_RERANK_MODEL))
    p.add_argument("--candidates", type=int,
                   default=_env_int("BENCH_CANDIDATES", DEFAULT_CANDIDATES))
    p.add_argument("--top-k", type=int, default=_env_int("BENCH_TOP_K", DEFAULT_TOP_K))
    p.add_argument("--collection", default=_env_str("BENCH_COLLECTION", "bench_chunks"))
    p.add_argument("--hnsw-m", type=int, default=_env_int("BENCH_HNSW_M", 16))
    p.add_argument("--hnsw-ef-construct", type=int,
                   default=_env_int("BENCH_HNSW_EF_CONSTRUCT", 100))
    p.add_argument("--output", default=_env_str("BENCH_OUTPUT", "benchmark_results.json"))
    p.add_argument("--seed", type=int, default=_env_int("BENCH_SEED", 42))
    return p


def main():
    # Nạp .env TRƯỚC khi build parser để env trở thành default của argparse.
    load_dotenv(os.environ.get("BENCH_ENV_FILE", ".env"))
    args = build_parser().parse_args()

    # cuda fallback
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[!] CUDA không khả dụng → fallback CPU")
                args.device = "cpu"
        except ImportError:
            print("[!] Chưa cài torch → CPU")
            args.device = "cpu"

    quant = None if args.quantization == "none" else args.quantization

    h = ExperimentHarness(
        qdrant_url=args.qdrant_url,
        device=args.device,
        embed_model=args.embed_model,
        rerank_model=args.rerank_model,
        collection=args.collection,
        quantization=quant,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construct=args.hnsw_ef_construct,
        candidates=args.candidates,
        top_k=args.top_k,
        chunk_tokens=args.chunk_tokens,
        seed=args.seed,
    )

    print("Loading models (có thể tải weights lần đầu) ...")
    try:
        h.load_models()
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] Không load được model: {e}")
        sys.exit(1)

    _print_header(args, h)

    # kiểm tra kết nối Qdrant sớm để fail nhanh với thông báo rõ ràng
    try:
        h.client = None
        h.provision_qdrant()
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] Không kết nối/provision được Qdrant tại {args.qdrant_url}: {e}")
        print("        → Đã chạy `docker compose -f docker/docker-compose.bench.yml up -d qdrant` chưa?")
        sys.exit(1)

    scale_points = args.scale_points if args.scale_test else [args.docs]

    started = time.time()
    per_scale = []
    for n in scale_points:
        per_scale.append(run_one_scale(h, n, args))

    payload = {
        "config": h.config().to_dict(),
        "elapsed_s": time.time() - started,
        "scale_results": per_scale,
        "notes": {
            "device_scope": "device chỉ áp dụng embed+rerank; Qdrant search & sparse encode luôn CPU/server",
            "qps_definition": "QPS = total_requests / wall_clock_time dưới tải song song (không phải 1/latency)",
            "ram_method": "công thức cấu trúc (dim*4*n + hnsw + sparse), KHÔNG phải RSS/n_chunks",
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Đã lưu: {args.output}  (tổng {payload['elapsed_s']:.0f}s)")
    print("🎉 Hoàn thành.")


if __name__ == "__main__":
    main()
