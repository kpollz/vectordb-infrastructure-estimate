#!/usr/bin/env python3
"""
PHẦN 1 — SETUP THÍ NGHIỆM (experiment_setup.py)
================================================================================
Dựng MÔI TRƯỜNG benchmark sao cho SÁT production `chat-with-documents`.

File này CHỈ lo build môi trường — KHÔNG chứa logic đo latency/throughput.
Việc đo lường nằm riêng ở `measurement.py`. Ranh giới đó là cố ý:
  - experiment_setup.py  = "thí nghiệm được set up thế nào" (corpus, model, Qdrant)
  - measurement.py       = "đo cái gì, đo ra sao" (probe latency/throughput/memory)

GROUND TRUTH lấy verbatim từ source production (đã đọc, không phỏng đoán):
  app/config.py, app/store/{embeddings,rerank,sparse,vector,indexing}.py,
  app/ingestion/chunk.py, app/inference/tools.py, deploy/docker-compose*.yml
--------------------------------------------------------------------------------
  Embedding : sentence-transformers/all-MiniLM-L12-v2  (dim=384, normalize=True)
  Reranker  : BAAI/bge-reranker-v2-m3  (CrossEncoder.predict([(q,d), ...]))
  Sparse    : sklearn HashingVectorizer(2**16, alternate_sign=False, norm="l2",
              lowercase=True, ngram_range=(1,1), dtype="float32")  → Qdrant sparse
  Qdrant    : SERVER v1.12.4, collection dense("dense",384,COSINE)+sparse("bm25"),
              HNSW mặc định, KHÔNG quantization (mặc định prod)
  Search    : query_points(prefetch=[dense k, sparse k], FusionQuery(RRF), limit)
  Chunk     : CHUNK_MAX_TOKENS=512 (token = whitespace split), có heading breadcrumb
  Query path: encode(dense+sparse ONLINE) → hybrid RRF → rerank → top_k=5, cand=20

LƯU Ý HẠ TẦNG QUAN TRỌNG (phản ánh docker-compose.gpu.yml):
  Trong production, Qdrant chạy trong CONTAINER RIÊNG dùng CPU; chỉ api/worker
  (embedding + rerank) mới được cấp GPU. Vì vậy `device` ở đây CHỈ áp dụng cho
  embed + rerank. `hybrid_search` luôn chạy trên Qdrant server (CPU), và
  `encode_query_sparse` (HashingVectorizer) luôn chạy CPU. measurement.py sẽ
  ghi rõ điều này trong báo cáo.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Model production (mặc định). Cho phép override qua CLI để so sánh (bge-m3, e5...).
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Sparse config — copy y hệt app/store/sparse.py::_vec()
SPARSE_N_FEATURES = 2 ** 16

# Named vectors — copy y hệt app/store/vector.py (DENSE / SPARSE)
DENSE_NAME = "dense"
SPARSE_NAME = "bm25"

# Retrieval defaults — copy từ app/config.py (RETRIEVAL_TOP_K, HYBRID_CANDIDATES)
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATES = 20
DEFAULT_CHUNK_TOKENS = 512  # CHUNK_MAX_TOKENS

# ---------------------------------------------------------------------------
# Từ vựng sinh corpus. Trộn Việt (domain doc điển hình) + thuật ngữ kỹ thuật.
# Mục tiêu là số TỪ / ký tự đại diện, không phải nội dung có nghĩa — benchmark
# đo COMPUTE (encode/search/rerank) nên phân bố độ dài token mới là điều quan trọng.
# ---------------------------------------------------------------------------
_WORDS = [
    "hệ", "thống", "dữ", "liệu", "phân", "tích", "báo", "cáo", "tài", "chính",
    "khách", "hàng", "sản", "phẩm", "dịch", "vụ", "hợp", "đồng", "thanh", "toán",
    "hóa", "đơn", "kho", "vận", "chuyển", "giao", "nhận", "bảo", "hành", "khiếu",
    "nại", "hỗ", "trợ", "kỹ", "thuật", "phần", "mềm", "cứng", "máy", "chủ",
    "quy", "trình", "sửa", "lỗi", "thao", "tác", "bước", "kiểm", "tra", "cài",
    "đặt", "cấu", "hình", "vận", "hành", "bảo", "trì", "thay", "thế", "linh",
    "kiện", "cảnh", "báo", "sự", "cố", "khắc", "phục", "nguyên", "nhân", "giải",
    "pháp", "hướng", "dẫn", "tham", "số", "thiết", "bị", "cảm", "biến", "động",
    "cơ", "điện", "áp", "dòng", "nhiệt", "độ", "áp", "suất", "lưu", "lượng",
    "database", "vector", "embedding", "retrieval", "index", "search", "rerank",
    "latency", "throughput", "benchmark", "GPU", "CPU", "memory", "chunk", "token",
]


def _one_chunk(n_tokens: int, rng: random.Random) -> str:
    """Sinh 1 chunk ~n_tokens từ (whitespace-token, khớp cách app/ingestion/chunk.py
    đếm token: len(text.split())). Có prefix breadcrumb như _make() trong chunk.py."""
    # Breadcrumb heading: "máy ABC › Lỗi X" — chiếm ~4-6 token đầu (giống prod).
    head = " › ".join(rng.choices(_WORDS, k=rng.randint(2, 4)))
    body_tokens = max(1, n_tokens - len(head.split()))
    body = " ".join(rng.choices(_WORDS, k=body_tokens))
    return f"{head}\n\n{body}"


@dataclass
class IngestStats:
    """Thống kê giai đoạn INGEST (offline workload) — không phải query latency."""
    n_chunks: int
    embed_ms: float
    sparse_ms: float
    upsert_ms: float
    total_ms: float
    embed_chunks_per_s: float
    overall_chunks_per_s: float
    peak_vram_mb: Optional[float] = None

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Toàn bộ tham số định nghĩa MỘT thí nghiệm (để dump vào JSON kết quả)."""
    qdrant_url: str
    device: str
    embed_model: str
    rerank_model: str
    collection: str
    embed_dim: int
    quantization: Optional[str]
    hnsw_m: int
    hnsw_ef_construct: int
    candidates: int
    top_k: int
    chunk_tokens: int
    seed: int

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


class ExperimentHarness:
    """Dựng và giữ mọi state của thí nghiệm: model, Qdrant client/collection, corpus.

    Vòng đời điển hình (do run_benchmark.py điều phối):
        h = ExperimentHarness(qdrant_url, device)
        h.load_models()
        h.provision_qdrant()
        docs = h.generate_corpus(n_chunks)
        stats = h.ingest(docs)
        queries = h.generate_queries(n_queries)
        # → chuyển h + queries sang measurement.py để đo
    """

    def __init__(
        self,
        qdrant_url: str,
        device: str,
        *,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        collection: str = "bench_chunks",
        quantization: Optional[str] = None,   # None | "scalar" | "binary"
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        candidates: int = DEFAULT_CANDIDATES,
        top_k: int = DEFAULT_TOP_K,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        seed: int = 42,
    ):
        self.qdrant_url = qdrant_url
        self.device = device
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.collection = collection
        self.quantization = quantization
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.candidates = candidates
        self.top_k = top_k
        self.chunk_tokens = chunk_tokens
        self.seed = seed
        self.rng = random.Random(seed)

        # Điền bởi load_models() / provision_qdrant() / ingest().
        self.embedder = None
        self.reranker = None
        self.vectorizer = None
        self.client = None
        self.embed_dim: Optional[int] = None
        self._corpus: List[str] = []   # giữ để rerank/e2e lấy candidate text thật

    # ------------------------------------------------------------------ config
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            qdrant_url=self.qdrant_url,
            device=self.device,
            embed_model=self.embed_model,
            rerank_model=self.rerank_model,
            collection=self.collection,
            embed_dim=self.embed_dim or 0,
            quantization=self.quantization,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construct=self.hnsw_ef_construct,
            candidates=self.candidates,
            top_k=self.top_k,
            chunk_tokens=self.chunk_tokens,
            seed=self.seed,
        )

    # ------------------------------------------------------------- 1a. corpus
    def generate_corpus(self, n_chunks: int, chunk_tokens: Optional[int] = None) -> List[str]:
        """Sinh n_chunks đoạn văn, mỗi đoạn ~chunk_tokens whitespace-token, kèm
        breadcrumb — khớp cấu trúc chunk đã index của production (chunk.py::_make)."""
        n_tok = chunk_tokens or self.chunk_tokens
        self._corpus = [_one_chunk(n_tok, self.rng) for _ in range(n_chunks)]
        return self._corpus

    def generate_queries(self, n_queries: int, query_tokens: int = 12) -> List[str]:
        """Query người dùng ngắn hơn chunk nhiều (câu hỏi). ~12 token là điển hình."""
        return [" ".join(self.rng.choices(_WORDS, k=query_tokens)) for _ in range(n_queries)]

    @property
    def corpus(self) -> List[str]:
        return self._corpus

    # ------------------------------------------------------------- 1b. models
    def load_models(self) -> None:
        """Load embedder + reranker (lên `device`) + sparse vectorizer (CPU).

        Params khớp từng dòng với app/store/{embeddings,rerank,sparse}.py.
        """
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from sklearn.feature_extraction.text import HashingVectorizer

        self.embedder = SentenceTransformer(self.embed_model, device=self.device)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()

        # CrossEncoder: prod để CrossEncoder tự chọn device; ta ép cho khớp benchmark.
        self.reranker = CrossEncoder(self.rerank_model, device=self.device)

        # Sparse — y hệt app/store/sparse.py::_vec()
        self.vectorizer = HashingVectorizer(
            n_features=SPARSE_N_FEATURES,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 1),
            dtype="float32",
        )

    # -------------------------------------------------------- 1c. provision QD
    def provision_qdrant(self) -> None:
        """Tạo (recreate) collection SCHEMA Y HỆT production trên Qdrant SERVER.

        Khác biệt duy nhất có chủ đích: cho phép bật quantization qua flag để đo
        đánh đổi RAM/độ chính xác (prod mặc định tắt).
        """
        if self.embed_dim is None:
            raise RuntimeError("Gọi load_models() trước provision_qdrant().")

        from qdrant_client import QdrantClient
        from qdrant_client.http.models import (
            BinaryQuantization,
            BinaryQuantizationConfig,
            Distance,
            HnswConfigDiff,
            ScalarQuantization,
            ScalarQuantizationConfig,
            ScalarType,
            SparseVectorParams,
            VectorParams,
        )

        self.client = QdrantClient(url=self.qdrant_url, timeout=60)

        quant_cfg = None
        if self.quantization == "scalar":
            quant_cfg = ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
            )
        elif self.quantization == "binary":
            quant_cfg = BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))

        # recreate_collection: xóa nếu tồn tại rồi tạo mới — sạch giữa các scale point.
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_NAME: VectorParams(size=self.embed_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={SPARSE_NAME: SparseVectorParams()},
            hnsw_config=HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct),
            quantization_config=quant_cfg,
        )

    # ---------------------------------------------------------- 1d. ingest
    def _encode_sparse_batch(self, texts: List[str]) -> List[Tuple[List[int], List[float]]]:
        """Y hệt app/store/sparse.py::encode_sparse()."""
        mat = self.vectorizer.transform(texts)
        out = []
        for i in range(mat.shape[0]):
            row = mat.getrow(i)
            out.append((row.indices.tolist(), row.data.tolist()))
        return out

    def ingest(self, docs: List[str], batch_size: int = 256, upsert_batch: int = 512,
               progress: bool = True) -> IngestStats:
        """Embed (normalize) + sparse-encode + upsert vào Qdrant. Đây là OFFLINE
        workload (khi approve tài liệu), đo tách khỏi query latency.

        Trả IngestStats: phase timings + throughput + peak VRAM (nếu cuda).
        """
        if self.embedder is None or self.client is None:
            raise RuntimeError("Cần load_models() và provision_qdrant() trước khi ingest().")

        from qdrant_client.http.models import PointStruct, SparseVector

        torch = _maybe_torch()
        if self.device == "cuda" and torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()

        # --- embed (normalize_embeddings=True, y app/store/embeddings.py::encode) ---
        # progress=True → sentence-transformers in progress bar tqdm cho khâu embed
        # (chậm nhất của ingest). Có thể tắt bằng progress=False.
        print(f"        → embed {len(docs):,} chunks (batch={batch_size}) ...", flush=True)
        t_e0 = time.perf_counter()
        vectors = self.embedder.encode(
            docs, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=progress,
        )
        t_e1 = time.perf_counter()

        # --- sparse (CPU) ---
        sparse_vecs = self._encode_sparse_batch(docs)
        t_s1 = time.perf_counter()

        # --- upsert theo batch (id = uuid4, y app/store/indexing.py) ---
        print(f"        → upsert {len(docs):,} points vào Qdrant ...", flush=True)
        t_u0 = time.perf_counter()
        buf: List[PointStruct] = []
        n_sent = 0
        for i, (vec, sp) in enumerate(zip(vectors, sparse_vecs)):
            idx, vals = sp
            buf.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={DENSE_NAME: vec.tolist(), SPARSE_NAME: SparseVector(indices=idx, values=vals)},
                payload={"content": docs[i]},
            ))
            if len(buf) >= upsert_batch:
                self.client.upsert(collection_name=self.collection, points=buf, wait=True)
                n_sent += len(buf)
                buf = []
                if progress:
                    print(f"\r          upserted {n_sent:,}/{len(docs):,}", end="", flush=True)
        if buf:
            self.client.upsert(collection_name=self.collection, points=buf, wait=True)
            n_sent += len(buf)
        if progress:
            print(f"\r          upserted {n_sent:,}/{len(docs):,}", flush=True)
        t_u1 = time.perf_counter()

        peak_vram = None
        if self.device == "cuda" and torch is not None and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        n = len(docs)
        embed_ms = (t_e1 - t_e0) * 1000
        sparse_ms = (t_s1 - t_e1) * 1000
        upsert_ms = (t_u1 - t_u0) * 1000
        total_ms = (t_u1 - t0) * 1000
        return IngestStats(
            n_chunks=n,
            embed_ms=embed_ms,
            sparse_ms=sparse_ms,
            upsert_ms=upsert_ms,
            total_ms=total_ms,
            embed_chunks_per_s=(n / (embed_ms / 1000)) if embed_ms > 0 else 0.0,
            overall_chunks_per_s=(n / (total_ms / 1000)) if total_ms > 0 else 0.0,
            peak_vram_mb=peak_vram,
        )

    # ------------------------------------------------ query helpers (ONLINE)
    # Các hàm dưới đây thực hiện ĐÚNG một bước của pipeline production cho MỘT
    # query. measurement.py bọc chúng trong vòng đo. Tách riêng để đo từng
    # component (embed / sparse / search / rerank) lẫn e2e mà không lặp code.

    def encode_query_dense(self, query: str) -> List[float]:
        """1 query → dense vector (normalize=True), y app/inference/tools.py."""
        v = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.tolist()

    def encode_query_sparse(self, query: str) -> Tuple[List[int], List[float]]:
        """1 query → sparse (indices, values), y app/store/sparse.py::encode_sparse_one."""
        return self._encode_sparse_batch([query])[0]

    def hybrid_search(self, dense: List[float], sparse: Tuple[List[int], List[float]]):
        """Hybrid dense+sparse fuse RRF — copy y app/store/vector.py::search_hybrid.

        LƯU Ý: chạy trên Qdrant SERVER (CPU), độc lập với self.device.
        """
        from qdrant_client.http.models import (
            Fusion,
            FusionQuery,
            Prefetch,
            SparseVector,
        )
        idx, vals = sparse
        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=dense, using=DENSE_NAME, limit=self.candidates),
                Prefetch(query=SparseVector(indices=idx, values=vals),
                         using=SPARSE_NAME, limit=self.candidates),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.candidates,   # lấy `candidates` để feed reranker (y tools.py fetch)
            with_payload=True,
        )
        points = result.points if hasattr(result, "points") else result
        return [{"id": str(p.id), "content": (p.payload or {}).get("content", "")} for p in points]

    def rerank(self, query: str, candidate_texts: List[str]) -> List[float]:
        """Cross-encoder score từng (query, doc) — y app/store/rerank.py::rerank."""
        if not candidate_texts:
            return []
        return [float(s) for s in self.reranker.predict([(query, d) for d in candidate_texts])]

    def collection_info(self):
        """Trả CollectionInfo của Qdrant (points_count, vectors_count...) để
        measurement.py cross-check ước lượng RAM lý thuyết với thực tế server."""
        return self.client.get_collection(self.collection)


def _maybe_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None
