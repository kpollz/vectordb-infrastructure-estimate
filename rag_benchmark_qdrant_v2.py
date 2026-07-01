#!/usr/bin/env python3
"""
RAG Infrastructure Benchmark Suite — Qdrant Edition v2
========================================================
Quy chuẩn chunk, phân tích xu hướng scale-up, và estimate infrastructure.

CÁC KHÁI NIỆM QUY CHUẨN:
-------------------------
1. CHUNK: Đơn vị nhỏ nhất được index vào vector DB. 1 chunk = 1 vector.
2. CHUNK SIZE: Được đo bằng số từ (words) hoặc token (≈ 0.75 words).
3. QUY ĐỔI:
   - Tiếng Việt: 1 từ ≈ 2-3 ký tự (không dấu) hoặc 3-5 ký tự (có dấu)
   - Tiếng Anh: 1 từ ≈ 5-6 ký tự
   - 1 token ≈ 0.75 từ tiếng Anh, ≈ 0.5-0.6 từ tiếng Việt

CÁC MỐC CHUNK SIZE PHỔ BIẾN:
----------------------------
| Chunk Size | Words | Characters (VN) | Characters (EN) | Token (approx) | Use Case |
|------------|-------|-----------------|-------------------|----------------|----------|
| 128 words  | 128   | ~400-600        | ~640-768          | ~170           | Short QA |
| 256 words  | 256   | ~800-1200       | ~1280-1536        | ~340           | Standard |
| 512 words  | 512   | ~1600-2400      | ~2560-3072        | ~680           | Long doc |
| 1024 words | 1024  | ~3200-4800      | ~5120-6144        | ~1360          | Article  |

XU HƯỚNG KHI TĂNG SỐ LƯỢNG CHUNK:
----------------------------------
- BM25:      O(1) per query (inverted index), RAM tăng tuyến tính với số docs
- Embedding: O(n) per batch, VRAM tăng tuyến tính với model size (không đổi với n_docs)
- Qdrant:    O(log n) per query (HNSW), RAM tăng tuyến tính với n_docs × dim × 4 bytes
- Rerank:    O(k) per query (k = top_k candidates), không phụ thuộc n_docs

Usage:
  # Benchmark với chunk size cụ thể
  python rag_benchmark_qdrant_v2.py --device cuda --pair 2 --chunk-size 256 --docs 10000

  # Phân tích xu hướng scale (tự động chạy nhiều mốc docs)
  python rag_benchmark_qdrant_v2.py --device cuda --pair 2 --scale-test --chunk-size 256

Dependencies:
  pip install torch sentence-transformers qdrant-client rank-bm25 psutil numpy matplotlib
"""

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import psutil
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Model pairs
# ---------------------------------------------------------------------------

MODEL_PAIRS = {
    1: {
        "name": "Lightweight (MiniLM + ms-marco-MiniLM)",
        "embed": "all-MiniLM-L6-v2",
        "embed_dim": 384,
        "rerank": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "Nhẹ nhất, tốc độ cao, phù hợp prototype hoặc CPU-only"
    },
    2: {
        "name": "Balanced (bge-base + bge-reranker-base)",
        "embed": "BAAI/bge-base-en-v1.5",
        "embed_dim": 768,
        "rerank": "BAAI/bge-reranker-base",
        "description": "Cân bằng chất lượng/tốc độ, khuyến nghị cho production"
    },
    3: {
        "name": "Heavy (bge-large + bge-reranker-v2-m3)",
        "embed": "BAAI/bge-large-en-v1.5",
        "embed_dim": 1024,
        "rerank": "BAAI/bge-reranker-v2-m3",
        "description": "Chất lượng cao nhất, cần GPU mạnh, phù hợp domain-specific"
    },
}

# ---------------------------------------------------------------------------
# Quy chuẩn chunk size
# ---------------------------------------------------------------------------

CHUNK_SIZE_TABLE = {
    128: {"words": 128, "chars_vn": 500, "chars_en": 700, "tokens": 170, "label": "Short QA"},
    256: {"words": 256, "chars_vn": 1000, "chars_en": 1400, "tokens": 340, "label": "Standard"},
    512: {"words": 512, "chars_vn": 2000, "chars_en": 2800, "tokens": 680, "label": "Long doc"},
    1024: {"words": 1024, "chars_vn": 4000, "chars_en": 5600, "tokens": 1360, "label": "Article"},
}

# ---------------------------------------------------------------------------
# Synthetic corpus generator với quy chuẩn chunk
# ---------------------------------------------------------------------------

WORDS_POOL = [
    "hệ thống", "dữ liệu", "phân tích", "báo cáo", "tài chính", "khách hàng",
    "sản phẩm", "dịch vụ", "hợp đồng", "thanh toán", "hóa đơn", "đơn hàng",
    "kho hàng", "vận chuyển", "giao nhận", "bảo hành", "khiếu nại", "hỗ trợ",
    "kỹ thuật", "phần mềm", "phần cứng", "máy chủ", "cloud", "API", "gateway",
    "microservices", "database", "vector", "embedding", "retrieval", "LLM",
    "token", "chunking", "index", "search", "ranking", "rerank", "latency",
    "throughput", "benchmark", "GPU", "CPU", "CUDA", "memory", "VRAM", "disk",
    "infrastructure", "estimate", "cost", "scaling", "sharding", "replication",
    "backup", "restore", "monitoring", "logging", "alert", "dashboard",
    "authentication", "authorization", "OAuth2", "JWT", "mTLS", "SSL",
    "error", "exception", "timeout", "retry", "circuit breaker", "rate limit",
    "throttle", "queue", "worker", "scheduler", "cron", "webhook", "event",
    "stream", "batch", "ETL", "pipeline", "workflow", "orchestration",
    "Kubernetes", "Docker", "container", "image", "registry", "helm", "terraform",
    "ansible", "CI/CD", "git", "repository", "branch", "merge", "deploy",
    "production", "staging", "development", "testing", "QA", "UAT", "SIT",
    "performance", "load", "stress", "capacity", "planning", "architecture",
    "design", "pattern", "principle", "best practice", "standard", "guideline",
    "policy", "procedure", "process", "framework", "methodology", "agile",
    "scrum", "sprint", "backlog", "story", "task", "epic", "milestone",
    "roadmap", "strategy", "vision", "mission", "goal", "objective", "KPI",
    "metric", "indicator", "measurement", "evaluation", "assessment", "review",
    "audit", "compliance", "regulation", "governance", "risk", "security",
    "privacy", "confidentiality", "integrity", "availability", "resilience",
    "disaster", "recovery", "business", "continuity", "plan", "incident",
    "response", "management", "operation", "maintenance", "support", "service",
    "desk", "ticket", "SLA", "OLA", "UC", "contract", "agreement", "terms",
    "condition", "clause", "provision", "stipulation", "requirement",
    "specification", "scope", "deliverable", "milestone", "phase", "stage",
    "gate", "checkpoint", "review", "approval", "sign-off", "handover",
    "transition", "transformation", "migration", "upgrade", "update", "patch",
    "version", "release", "deployment", "rollout", "launch", "go-live",
    "sunset", "deprecation", "end-of-life", "retirement", "disposal",
    "sustainability", "efficiency", "optimization", "tuning", "profiling",
    "debugging", "troubleshooting", "root cause", "analysis", "resolution",
    "fix", "workaround", "solution", "alternative", "option", "choice",
    "decision", "trade-off", "compromise", "balance", "alignment", "integration",
    "interoperability", "compatibility", "portability", "flexibility",
    "extensibility", "scalability", "reliability", "maintainability", "usability",
    "accessibility", "localizability", "globalization", "internationalization",
    "customization", "configuration", "personalization", "adaptation", "evolution",
    "innovation", "disruption", "transformation", "digital", "automation",
    "intelligence", "artificial", "machine", "learning", "deep", "neural",
    "network", "model", "algorithm", "training", "inference", "prediction",
    "classification", "regression", "clustering", "dimensionality", "reduction",
    "feature", "extraction", "selection", "engineering", "representation",
    "encoding", "decoding", "compression", "decompression", "encryption",
    "decryption", "hashing", "signature", "checksum", "validation",
    "verification", "authentication", "identification", "authorization",
    "permission", "role", "group", "user", "account", "profile", "session",
    "cookie", "token", "claim", "scope", "audience", "issuer", "subject",
    "resource", "endpoint", "route", "path", "parameter", "query", "body",
    "header", "response", "status", "code", "message", "error", "warning",
    "info", "debug", "trace", "log", "event", "audit", "history", "record",
    "transaction", "operation", "action", "activity", "behavior", "pattern",
    "trend", "anomaly", "outlier", "correlation", "causation", "inference",
    "deduction", "induction", "abduction", "hypothesis", "theory", "law",
    "principle", "axiom", "postulate", "theorem", "proof", "evidence",
    "data", "fact", "observation", "measurement", "experiment", "test",
    "trial", "pilot", "prototype", "MVP", "POC", "pilot", "beta", "alpha",
    "stable", "mature", "legacy", "modern", "future", "vision", "roadmap",
    "plan", "schedule", "timeline", "budget", "cost", "expense", "revenue",
    "profit", "loss", "investment", "return", "ROI", "NPV", "IRR", "payback",
    "break-even", "margin", "markup", "discount", "rebate", "commission",
    "fee", "charge", "price", "value", "worth", "asset", "liability",
    "equity", "capital", "fund", "budget", "allocation", "appropriation",
    "expenditure", "spending", "saving", "reserve", "provision", "accrual",
    "deferral", "amortization", "depreciation", "impairment", "write-off",
    "gain", "loss", "income", "earnings", "dividend", "yield", "interest",
    "rate", "exchange", "currency", "forex", "hedge", "derivative", "option",
    "future", "forward", "swap", "bond", "stock", "share", "equity", "debt",
    "loan", "credit", "debit", "balance", "statement", "report", "ledger",
    "journal", "entry", "posting", "reconciliation", "adjustment", "closing",
    "opening", "period", "fiscal", "calendar", "quarter", "month", "week",
    "day", "hour", "minute", "second", "millisecond", "microsecond",
    "timestamp", "datetime", "timezone", "UTC", "GMT", "local", "relative",
    "absolute", "duration", "interval", "elapsed", "remaining", "deadline",
    "due", "overdue", "pending", "active", "inactive", "completed", "cancelled",
    "failed", "success", "partial", "full", "total", "sum", "average", "mean",
    "median", "mode", "range", "variance", "standard deviation", "confidence",
    "interval", "significance", "p-value", "hypothesis", "null", "alternative",
    "test", "chi-square", "t-test", "ANOVA", "regression", "correlation",
    "coefficient", "determination", "slope", "intercept", "residual", "error",
    "RMSE", "MAE", "MAPE", "accuracy", "precision", "recall", "F1", "AUC",
    "ROC", "confusion", "matrix", "true positive", "false positive", "true",
    "false", "negative", "positive", "label", "class", "category", "type",
    "kind", "sort", "genre", "domain", "field", "area", "sector", "industry",
    "market", "segment", "niche", "vertical", "horizontal", "ecosystem",
    "platform", "marketplace", "network", "community", "audience", "customer",
    "client", "partner", "supplier", "vendor", "distributor", "reseller",
    "retailer", "wholesaler", "manufacturer", "producer", "provider", "carrier",
    "operator", "tenant", "landlord", "owner", "stakeholder", "shareholder",
    "investor", "creditor", "debtor", "guarantor", "beneficiary", "trustee",
    "fiduciary", "agent", "principal", "representative", "delegate", "proxy",
    "ambassador", "advocate", "champion", "sponsor", "patron", "donor",
    "contributor", "volunteer", "member", "subscriber", "follower", "fan",
    "user", "consumer", "end-user", "prosumer", "creator", "maker", "builder",
    "developer", "engineer", "architect", "designer", "analyst", "scientist",
    "researcher", "scholar", "academic", "practitioner", "professional",
    "expert", "specialist", "generalist", "novice", "beginner", "intermediate",
    "advanced", "master", "guru", "ninja", "rockstar", "wizard", "magician",
    "artist", "craftsman", "artisan", "technician", "mechanic", "operator",
    "driver", "pilot", "captain", "commander", "leader", "manager",
    "director", "executive", "officer", "president", "CEO", "CFO", "CTO",
    "COO", "CMO", "CIO", "CSO", "CCO", "CHRO", "CPO", "CDO", "CAO", "CLO",
    "general counsel", "secretary", "treasurer", "controller", "auditor",
    "accountant", "bookkeeper", "clerk", "assistant", "associate", "partner",
    "principal", "managing director", "senior", "junior", "lead", "head",
    "chief", "vice", "deputy", "assistant", "associate", "staff", "contractor",
    "consultant", "advisor", "counselor", "coach", "mentor", "trainer",
    "educator", "instructor", "teacher", "professor", "lecturer", "tutor",
    "facilitator", "moderator", "coordinator", "organizer", "planner",
    "scheduler", "dispatcher", "controller", "supervisor", "foreman",
    "superintendent", "inspector", "examiner", "investigator", "detective",
    "analyst", "evaluator", "assessor", "appraiser", "reviewer", "critic",
    "editor", "publisher", "writer", "author", "journalist", "reporter",
    "correspondent", "broadcaster", "anchor", "host", "presenter", "speaker",
    "narrator", "storyteller", "poet", "novelist", "playwright", "screenwriter",
    "copywriter", "content", "creator", "curator", "librarian", "archivist",
    "historian", "genealogist", "anthropologist", "sociologist", "psychologist",
    "philosopher", "theologian", "linguist", "translator", "interpreter",
    "diplomat", "negotiator", "mediator", "arbitrator", "conciliator",
    "peacemaker", "reconciler", "healer", "therapist", "counselor", "social",
    "worker", "caregiver", "nurse", "doctor", "physician", "surgeon",
    "specialist", "general practitioner", "dentist", "pharmacist", "optometrist",
    "ophthalmologist", "dermatologist", "cardiologist", "neurologist",
    "psychiatrist", "radiologist", "pathologist", "oncologist", "pediatrician",
    "geriatrician", "obstetrician", "gynecologist", "orthopedist", "physiatrist",
    "podiatrist", "chiropractor", "osteopath", "naturopath", "homeopath",
    "acupuncturist", "herbalist", "nutritionist", "dietitian", "trainer",
    "coach", "instructor", "therapist", "masseur", "aesthetician", "stylist",
    "barber", "hairdresser", "manicurist", "pedicurist", "cosmetologist",
    "makeup", "artist", "fashion", "designer", "tailor", "seamstress",
    "cobbler", "jeweler", "watchmaker", "goldsmith", "silversmith",
    "blacksmith", "welder", "machinist", "toolmaker", "die maker", "mold",
    "maker", "patternmaker", "foundry", "worker", "caster", "forger",
    "roller", "extruder", "drawer", "presser", "stamper", "embosser",
    "engraver", "etcher", "printer", "lithographer", "photographer",
    "videographer", "cinematographer", "director", "producer", "actor",
    "actress", "performer", "musician", "singer", "dancer", "choreographer",
    "composer", "conductor", "arranger", "orchestrator", "sound", "engineer",
    "mixer", "mastering", "editor", "DJ", "VJ", "MC", "host", "announcer",
    "commentator", "critic", "reviewer", "blogger", "vlogger", "streamer",
    "influencer", "celebrity", "public figure", "personality", "icon",
    "legend", "star", "superstar", "megastar", "diva", "prodigy", "virtuoso",
    "maestro", "genius", "pioneer", "trailblazer", "innovator", "inventor",
    "discoverer", "explorer", "adventurer", "navigator", "cartographer",
    "surveyor", "geologist", "geographer", "oceanographer", "meteorologist",
    "astronomer", "astrophysicist", "cosmologist", "physicist", "chemist",
    "biologist", "botanist", "zoologist", "ecologist", "environmentalist",
    "conservationist", "activist", "advocate", "campaigner", "lobbyist",
    "politician", "statesman", "diplomat", "ambassador", "envoy", "delegate",
    "representative", "senator", "congressman", "parliamentarian", "minister",
    "secretary", "commissioner", "governor", "mayor", "councilman", "alderman",
    "supervisor", "trustee", "board member", "director", "chairman", "chairwoman",
    "chairperson", "presiding officer", "moderator", "facilitator", "mediator",
    "arbitrator", "judge", "magistrate", "justice", "referee", "umpire",
    "official", "regulator", "enforcer", "police", "officer", "detective",
    "sergeant", "lieutenant", "captain", "chief", "sheriff", "marshal",
    "ranger", "trooper", "agent", "special agent", "investigator", "inspector",
    "customs", "border", "immigration", "coast guard", "navy", "marine",
    "soldier", "sailor", "airman", "guardsman", "reservist", "veteran",
    "retiree", "pensioner", "senior citizen", "elder", "youth", "teenager",
    "adolescent", "child", "toddler", "infant", "baby", "newborn", "fetus",
    "embryo", "zygote", "offspring", "descendant", "ancestor", "predecessor",
    "successor", "heir", "beneficiary", "legatee", "devisee", "assignee",
    "transferee", "recipient", "receiver", "collector", "gatherer", "hunter",
    "gatherer", "farmer", "rancher", "herder", "breeder", "fisherman",
    "hunter", "trapper", "miner", "logger", "forester", "ranger", "warden",
    "conservation officer", "gamekeeper", "park ranger", "zookeeper", "aquarist",
    "veterinarian", "animal trainer", "handler", "groomer", "walker", "sitter",
    "pet", "owner", "companion", "friend", "ally", "confidant", "comrade",
    "colleague", "coworker", "teammate", "partner", "collaborator", "coauthor",
    "coeditor", "copublisher", "coproducer", "codirector", "coactor",
    "costar", "coplayer", "opponent", "rival", "competitor", "adversary",
    "enemy", "foe", "antagonist", "villain", "criminal", "offender",
    "perpetrator", "suspect", "defendant", "prisoner", "inmate", "convict",
    "felon", "misdemeanant", "juvenile delinquent", "probationer", "parolee",
    "escapee", "fugitive", "refugee", "asylum seeker", "immigrant", "migrant",
    "emigrant", "expatriate", "citizen", "national", "subject", "resident",
    "domiciliary", "inhabitant", "occupant", "tenant", "lodger", "boarder",
    "guest", "visitor", "traveler", "tourist", "passenger", "commuter",
    "pilgrim", "wanderer", "nomad", "vagabond", "drifter", "transient",
    "migrant worker", "seasonal worker", "day laborer", "freelancer",
    "independent contractor", "gig worker", "sole proprietor", "entrepreneur",
    "founder", "cofounder", "owner", "proprietor", "landlord", "lessor",
    "lessee", "renter", "subletter", "subtenant", "roommate", "flatmate",
    "housemate", "neighbor", "acquaintance", "stranger", "outsider",
    "foreigner", "alien", "extraterrestrial", "immigrant", "newcomer",
    "arrival", "entrant", "initiate", "novice", "beginner", "learner",
    "student", "pupil", "disciple", "follower", "adherent", "believer",
    "devotee", "fanatic", "enthusiast", "aficionado", "connoisseur",
    "cognoscente", "expert", "authority", "pundit", "commentator", "critic",
    "reviewer", "judge", "evaluator", "assessor", "appraiser", "estimator",
    "surveyor", "inspector", "examiner", "auditor", "accountant", "actuary",
    "underwriter", "broker", "dealer", "trader", "merchant", "retailer",
    "shopkeeper", "store owner", "franchisee", "licensee", "permittee",
    "certificate holder", "credential holder", "license holder", "degree holder",
    "graduate", "alumnus", "alumna", "former student", "dropout", "expellee",
    "reject", "failure", "loser", "underdog", "dark horse", "sleeper",
    "surprise", "upset", "shock", "sensation", "phenomenon", "marvel",
    "wonder", "miracle", "prodigy", "genius", "mastermind", "brain",
    "intellect", "thinker", "philosopher", "theorist", "ideologist",
    "ideologue", "doctrinaire", "dogmatist", "sectarian", "partisan",
    "factionalist", "tribalist", "nationalist", "patriot", "loyalist",
    "royalist", "republican", "democrat", "liberal", "conservative",
    "progressive", "reactionary", "radical", "extremist", "moderate",
    "centrist", "independent", "nonpartisan", "bipartisan", "multipartisan",
    "crossbench", "swing voter", "undecided", "apolitical", "uninvolved",
    "disengaged", "alienated", "disaffected", "dissident", "dissenter",
    "protester", "demonstrator", "activist", "organizer", "mobilizer",
    "agitator", "instigator", "provocateur", "troublemaker", "rabble-rouser",
    "firebrand", "revolutionary", "insurrectionist", "rebel", "insurgent",
    "guerrilla", "freedom fighter", "terrorist", "extremist", "militant",
    "paramilitary", "mercenary", "soldier of fortune", "privateer", "corsair",
    "buccaneer", "pirate", "freebooter", "raider", "marauder", "plunderer",
    "looter", "pillager", "sacker", "ravager", "destroyer", "vandal",
    "saboteur", "arsonist", "bomber", "hijacker", "kidnapper", "abductor",
    "hostage taker", "extortionist", "blackmailer", "racketeer", "mobster",
    "gangster", "thug", "hoodlum", "hooligan", "delinquent", "criminal",
    "felon", "convict", "prisoner", "inmate", "detainee", "captive",
    "slave", "servant", "serf", "peon", "villein", "bondsman", "indentured",
    "apprentice", "journeyman", "master craftsman", "guild member", "union",
    "member", "card holder", "dues payer", "striker", "picket", "boycotter",
    "sanctioner", "embargoer", "blockader", "besieger", "investor",
]


def generate_chunk(n_words: int) -> str:
    """Tạo 1 chunk có đúng n_words từ."""
    return " ".join(random.choices(WORDS_POOL, k=n_words))


def generate_corpus(n_chunks: int, chunk_words: int) -> List[str]:
    """Tạo corpus gồm n_chunks, mỗi chunk có chunk_words từ."""
    return [generate_chunk(chunk_words) for _ in range(n_chunks)]


def generate_queries(n_queries: int, query_words: int = 15) -> List[str]:
    return [generate_chunk(query_words) for _ in range(n_queries)]


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    component: str
    device: str
    model_pair: str
    n_chunks: int
    chunk_words: int
    chunk_chars_approx: int
    n_queries: int
    batch_size: int
    p50_ms: float
    p99_ms: float
    avg_ms: float
    std_ms: float
    throughput_qps: float
    peak_memory_mb: float
    peak_vram_mb: Optional[float] = None
    index_build_time_s: Optional[float] = None
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class MemoryMonitor:
    def __init__(self, device: str):
        self.device = device
        self.process = psutil.Process(os.getpid())
        self.peak_ram = 0.0
        self.peak_vram = 0.0
        self._running = False

    def start(self):
        self._running = True
        self.peak_ram = self.process.memory_info().rss / (1024 * 1024)
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.peak_vram = torch.cuda.memory_allocated() / (1024 * 1024)

    def snapshot(self):
        if not self._running:
            return
        ram = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram = max(self.peak_ram, ram)
        if self.device == "cuda" and torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 * 1024)
            self.peak_vram = max(self.peak_vram, vram)

    def stop(self) -> Tuple[float, Optional[float]]:
        self._running = False
        ram = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram = max(self.peak_ram, ram)
        vram = None
        if self.device == "cuda" and torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self.peak_vram = max(self.peak_vram, vram)
        return self.peak_ram, self.peak_vram


def run_benchmark(name, device, n_chunks, chunk_words, n_queries, batch_size, fn, warmup=3):
    print(f"\n[Benchmark] {name} | device={device} | chunks={n_chunks}×{chunk_words}w | queries={n_queries} | batch={batch_size}")

    for _ in range(warmup):
        try:
            fn(batch_size=batch_size)
        except Exception:
            fn()

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    monitor = MemoryMonitor(device)
    monitor.start()

    latencies = []
    for _ in range(n_queries):
        monitor.snapshot()
        t0 = time.perf_counter()
        try:
            fn(batch_size=batch_size)
        except Exception:
            fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        monitor.snapshot()

    peak_ram, peak_vram = monitor.stop()
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = statistics.mean(latencies)
    std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    total_time = sum(latencies) / 1000.0
    qps = n_queries / total_time if total_time > 0 else 0.0

    return BenchmarkResult(
        component=name,
        device=device,
        model_pair="",
        n_chunks=n_chunks,
        chunk_words=chunk_words,
        chunk_chars_approx=chunk_words * 4,  # Approx VN chars
        n_queries=n_queries,
        batch_size=batch_size,
        p50_ms=p50,
        p99_ms=p99,
        avg_ms=avg,
        std_ms=std,
        throughput_qps=qps,
        peak_memory_mb=peak_ram,
        peak_vram_mb=peak_vram,
    )


# ---------------------------------------------------------------------------
# 1. BM25 Benchmark
# ---------------------------------------------------------------------------

def benchmark_bm25(docs, queries, device, n_queries, batch_size):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[!] pip install rank-bm25")
        sys.exit(1)

    tokenized_docs = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    def _search(batch_size=1):
        q = random.choice(queries).lower().split()
        scores = bm25.get_scores(q)
        top_k = np.argsort(scores)[-10:][::-1]
        return top_k

    return run_benchmark("BM25", device, len(docs), len(docs[0].split()), n_queries, batch_size, _search)


# ---------------------------------------------------------------------------
# 2. Embedding + Qdrant Vector Search Benchmark
# ---------------------------------------------------------------------------

def benchmark_embedding_qdrant(docs, queries, device, n_queries, batch_size, embed_model, embed_dim, chunk_words):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    model = SentenceTransformer(embed_model, device=device)
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(":memory:")
    collection_name = "benchmark"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Encode & upsert docs
    print(f"  → Encoding {len(docs)} chunks with {embed_model} ...")
    t0 = time.time()
    doc_embeddings = model.encode(docs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    t1 = time.time()
    build_time = t1 - t0
    print(f"  → Encoded in {build_time:.2f}s, shape={doc_embeddings.shape}")

    batch_upsert = 1000
    points = []
    for i, emb in enumerate(doc_embeddings):
        points.append(PointStruct(id=i, vector=emb.tolist(), payload={"text": docs[i][:200]}))
        if len(points) >= batch_upsert:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    if points:
        client.upsert(collection_name=collection_name, points=points)
    print(f"  → Upserted {len(docs)} vectors to Qdrant")

    query_embeddings = model.encode(queries, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

    def _search(batch_size=1):
        q_idx = random.randint(0, len(queries) - 1)
        q_emb = query_embeddings[q_idx].tolist()
        result = client.search(
            collection_name=collection_name,
            query_vector=q_emb,
            limit=10,
        )
        return [r.id for r in result]

    result = run_benchmark(f"Qdrant({embed_model})", device, len(docs), chunk_words, n_queries, batch_size, _search)
    result.index_build_time_s = build_time
    return result


# ---------------------------------------------------------------------------
# 3. Re-ranking Benchmark
# ---------------------------------------------------------------------------

def benchmark_rerank(docs, queries, device, n_queries, batch_size, rerank_model, chunk_words, top_k=20):
    model = CrossEncoder(rerank_model, device=device)

    def _rerank(batch_size=1):
        q = random.choice(queries)
        candidates = random.sample(docs, min(top_k, len(docs)))
        pairs = [(q, d) for d in candidates]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:5]

    return run_benchmark(f"Rerank({rerank_model})", device, len(docs), chunk_words, n_queries, batch_size, _rerank)


# ---------------------------------------------------------------------------
# 4. End-to-end Pipeline Benchmark
# ---------------------------------------------------------------------------

def benchmark_pipeline(docs, queries, device, n_queries, batch_size, embed_model, rerank_model, chunk_words, top_k_retrieve=50, top_k_rerank=5):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    embedder = SentenceTransformer(embed_model, device=device)
    reranker = CrossEncoder(rerank_model, device=device)
    dim = embedder.get_sentence_embedding_dimension()

    client = QdrantClient(":memory:")
    collection_name = "benchmark_pipeline"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    print(f"  → Pipeline: encoding {len(docs)} chunks ...")
    t0 = time.time()
    doc_embeddings = embedder.encode(docs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    t1 = time.time()
    build_time = t1 - t0

    points = []
    for i, emb in enumerate(doc_embeddings):
        points.append(PointStruct(id=i, vector=emb.tolist(), payload={"text": docs[i][:200]}))
        if len(points) >= 1000:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    if points:
        client.upsert(collection_name=collection_name, points=points)

    query_embeddings = embedder.encode(queries, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

    def _pipeline(batch_size=1):
        q_idx = random.randint(0, len(queries) - 1)
        q_text = queries[q_idx]
        q_emb = query_embeddings[q_idx].tolist()

        result = client.search(
            collection_name=collection_name,
            query_vector=q_emb,
            limit=top_k_retrieve,
        )
        candidates = [docs[r.id] for r in result]

        pairs = [(q_text, d) for d in candidates]
        scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k_rerank]

    result = run_benchmark("E2E-Pipeline(Qdrant+Rerank)", device, len(docs), chunk_words, n_queries, batch_size, _pipeline)
    result.index_build_time_s = build_time
    return result


# ---------------------------------------------------------------------------
# Scale test: chạy nhiều mốc docs
# ---------------------------------------------------------------------------

def run_scale_test(device, pair, chunk_size, batch_size, queries_per_point=100):
    """Chạy benchmark với nhiều mốc số lượng chunk để phân tích xu hướng."""
    scale_points = [1000, 5000, 10000, 50000, 100000]
    pair_info = MODEL_PAIRS[pair]

    all_results = []

    for n_chunks in scale_points:
        if n_chunks > 100000 and device == "cpu":
            print(f"\n[!] Bỏ qua {n_chunks} chunks trên CPU (quá chậm)")
            continue

        print(f"\n{'='*60}")
        print(f"SCALE TEST: {n_chunks} chunks × {chunk_size} words")
        print(f"{'='*60}")

        docs = generate_corpus(n_chunks, chunk_size)
        queries = generate_queries(queries_per_point, query_words=15)

        # BM25
        r = benchmark_bm25(docs, queries, device, queries_per_point, batch_size)
        r.model_pair = pair_info["name"]
        all_results.append(r)

        # Qdrant
        r = benchmark_embedding_qdrant(docs, queries, device, queries_per_point, batch_size,
                                       pair_info["embed"], pair_info["embed_dim"], chunk_size)
        r.model_pair = pair_info["name"]
        all_results.append(r)

        # Rerank
        r = benchmark_rerank(docs, queries, device, queries_per_point, batch_size,
                             pair_info["rerank"], chunk_size)
        r.model_pair = pair_info["name"]
        all_results.append(r)

        # Pipeline
        r = benchmark_pipeline(docs, queries, device, queries_per_point, batch_size,
                               pair_info["embed"], pair_info["rerank"], chunk_size)
        r.model_pair = pair_info["name"]
        all_results.append(r)

        # Cleanup
        del docs, queries
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results):
    print("\n" + "=" * 140)
    print(f"{'Component':<50} {'Device':<8} {'Chunks':>10} {'Words':>8} {'Q':>6} {'Batch':>5} {'P50(ms)':>10} {'P99(ms)':>10} {'Avg(ms)':>10} {'QPS':>8} {'RAM(MB)':>10} {'VRAM(MB)':>10}")
    print("=" * 140)
    for r in results:
        vram = f"{r.peak_vram_mb:.0f}" if r.peak_vram_mb is not None else "-"
        print(f"{r.component:<50} {r.device:<8} {r.n_chunks:>10} {r.chunk_words:>8} {r.n_queries:>6} {r.batch_size:>5} {r.p50_ms:>10.2f} {r.p99_ms:>10.2f} {r.avg_ms:>10.2f} {r.throughput_qps:>8.2f} {r.peak_memory_mb:>10.0f} {vram:>10}")
    print("=" * 140)


def print_scale_analysis(results):
    """Phân tích xu hướng khi tăng số lượng chunk."""
    print("\n📈 PHÂN TÍCH XU HƯỚNG SCALE-UP")
    print("-" * 80)

    # Group by component
    by_comp = {}
    for r in results:
        by_comp.setdefault(r.component, []).append(r)

    for comp, rs in sorted(by_comp.items()):
        rs_sorted = sorted(rs, key=lambda x: x.n_chunks)
        print(f"\n  ▶ {comp}")
        print(f"    {'Chunks':>12} {'P50(ms)':>12} {'P99(ms)':>12} {'QPS':>12} {'RAM(MB)':>12} {'VRAM(MB)':>12}")
        print(f"    {'-'*72}")
        for r in rs_sorted:
            vram = f"{r.peak_vram_mb:.0f}" if r.peak_vram_mb else "-"
            print(f"    {r.n_chunks:>12} {r.p50_ms:>12.2f} {r.p99_ms:>12.2f} {r.throughput_qps:>12.2f} {r.peak_memory_mb:>12.0f} {vram:>12}")

        # Phân tích xu hướng
        if len(rs_sorted) >= 2:
            first = rs_sorted[0]
            last = rs_sorted[-1]
            chunk_ratio = last.n_chunks / first.n_chunks
            latency_ratio = last.p50_ms / first.p50_ms
            ram_ratio = last.peak_memory_mb / first.peak_memory_mb

            print(f"    {'-'*72}")
            print(f"    Scale factor: {chunk_ratio:.0f}x chunks")
            print(f"    Latency growth: {latency_ratio:.2f}x (lý thuyết: O(1) hoặc O(log n))")
            print(f"    RAM growth: {ram_ratio:.2f}x (lý thuyết: tuyến tính)")

            if "BM25" in comp:
                print(f"    → BM25: Latency gần như không đổi (O(1)), RAM tăng tuyến tính")
            elif "Qdrant" in comp:
                print(f"    → Qdrant: Latency tăng chậm (O(log n) với HNSW), RAM tăng tuyến tính")
            elif "Rerank" in comp:
                print(f"    → Rerank: Latency không đổi (không phụ thuộc n_chunks), VRAM không đổi")
            elif "Pipeline" in comp:
                print(f"    → Pipeline: Bottleneck thường ở Rerank hoặc Embedding")


def print_chunk_reference():
    print("\n📏 QUY CHUẨN CHUNK SIZE")
    print("-" * 80)
    print(f"{'Chunk Size':<12} {'Words':<8} {'Chars (VN)':<12} {'Chars (EN)':<12} {'Tokens':<10} {'Use Case'}")
    print("-" * 80)
    for size, info in CHUNK_SIZE_TABLE.items():
        print(f"{size:<12} {info['words']:<8} {info['chars_vn']:<12} {info['chars_en']:<12} {info['tokens']:<10} {info['label']}")
    print("-" * 80)
    print("Note: 1 token ≈ 0.75 word (EN), ≈ 0.5-0.6 word (VN)")
    print("      1 word VN ≈ 3-5 ký tự (có dấu), 1 word EN ≈ 5-6 ký tự")


def print_recommendations(results, pair_info, chunk_size):
    print(f"\n📊 KHUYẾN NGHỊ CHO: {pair_info['name']}")
    print(f"   Embedding : {pair_info['embed']} (dim={pair_info['embed_dim']})")
    print(f"   Re-ranker : {pair_info['rerank']}")
    print(f"   Chunk size: {chunk_size} words (~{CHUNK_SIZE_TABLE.get(chunk_size, {}).get('chars_vn', '?')} chars VN)")
    print("-" * 60)

    by_comp = {}
    for r in results:
        by_comp.setdefault(r.component, []).append(r)

    for comp, rs in by_comp.items():
        cpu = [r for r in rs if r.device == "cpu"]
        gpu = [r for r in rs if r.device == "cuda"]

        if cpu and gpu:
            c = cpu[0]
            g = gpu[0]
            speedup = c.avg_ms / g.avg_ms if g.avg_ms > 0 else 0
            vram = f"{g.peak_vram_mb:.0f} MB" if g.peak_vram_mb else "N/A"
            print(f"  • {comp}: CPU {c.avg_ms:.1f}ms → GPU {g.avg_ms:.1f}ms | Speedup {speedup:.1f}x | VRAM {vram}")
        elif cpu:
            c = cpu[0]
            print(f"  • {comp}: CPU-only | {c.avg_ms:.1f}ms | RAM {c.peak_memory_mb:.0f} MB")
        elif gpu:
            g = gpu[0]
            print(f"  • {comp}: GPU-only | {g.avg_ms:.1f}ms | VRAM {g.peak_vram_mb:.0f} MB")

    # RAM estimate for Qdrant
    qdrant_results = [r for r in results if "Qdrant" in r.component]
    if qdrant_results:
        r = qdrant_results[0]
        dim = pair_info["embed_dim"]
        ram_per_chunk = r.peak_memory_mb / r.n_chunks
        print(f"\n  💾 Qdrant RAM estimate:")
        print(f"     Per chunk: ~{ram_per_chunk:.2f} MB (dim={dim})")
        print(f"     100K chunks: ~{ram_per_chunk * 100000 / 1024:.1f} GB")
        print(f"     1M chunks:   ~{ram_per_chunk * 1000000 / 1024:.1f} GB")
        print(f"     10M chunks:  ~{ram_per_chunk * 10000000 / 1024:.1f} GB")


def save_json(results, path, pair_info, chunk_size):
    payload = {
        "model_pair": pair_info,
        "chunk_size_words": chunk_size,
        "chunk_size_reference": CHUNK_SIZE_TABLE.get(chunk_size, {}),
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Đã lưu kết quả: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG Infrastructure Benchmark — Qdrant Edition v2")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--pair", type=int, choices=[1, 2, 3], default=2, help="Model pair: 1=Lightweight, 2=Balanced, 3=Heavy")
    parser.add_argument("--docs", type=int, default=10000, help="Số lượng chunks (docs)")
    parser.add_argument("--chunk-size", type=int, default=256, choices=[128, 256, 512, 1024], help="Số từ (words) mỗi chunk")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--scale-test", action="store_true", help="Chạy nhiều mốc docs để phân tích xu hướng")
    parser.add_argument("--output", default="rag_benchmark_qdrant_v2_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pair_info = MODEL_PAIRS[args.pair]

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA không khả dụng, fallback về CPU")
        args.device = "cpu"

    print_chunk_reference()

    print("\n" + "=" * 60)
    print("RAG INFRASTRUCTURE BENCHMARK — QDRANT EDITION v2")
    print("=" * 60)
    print(f"Device       : {args.device}")
    print(f"Model Pair   : {args.pair} — {pair_info['name']}")
    print(f"  Embedding  : {pair_info['embed']} (dim={pair_info['embed_dim']})")
    print(f"  Re-ranker  : {pair_info['rerank']}")
    print(f"Chunk size   : {args.chunk_size} words (~{CHUNK_SIZE_TABLE[args.chunk_size]['chars_vn']} chars VN)")
    print(f"Docs (chunks): {args.docs}")
    print(f"Queries      : {args.queries}")
    print(f"Batch size   : {args.batch_size}")
    print("=" * 60)

    if args.scale_test:
        results = run_scale_test(args.device, args.pair, args.chunk_size, args.batch_size, args.queries)
        print_scale_analysis(results)
    else:
        print(f"\n[1/4] Generating corpus: {args.docs} chunks × {args.chunk_size} words ...")
        docs = generate_corpus(args.docs, args.chunk_size)
        queries = generate_queries(args.queries, query_words=15)
        print(f"      Corpus: {len(docs)} chunks | Queries: {len(queries)}")
        print(f"      Approx: {args.docs * args.chunk_size} total words | {args.docs * args.chunk_size * 4} chars (VN)")

        results = []

        print("\n[2/4] Benchmarking BM25 ...")
        r = benchmark_bm25(docs, queries, args.device, args.queries, args.batch_size)
        r.model_pair = pair_info["name"]
        results.append(r)

        print("\n[3/4] Benchmarking Embedding + Qdrant ...")
        r = benchmark_embedding_qdrant(docs, queries, args.device, args.queries, args.batch_size,
                                       pair_info["embed"], pair_info["embed_dim"], args.chunk_size)
        r.model_pair = pair_info["name"]
        results.append(r)

        print("\n[4/4] Benchmarking Re-ranking ...")
        r = benchmark_rerank(docs, queries, args.device, args.queries, args.batch_size,
                             pair_info["rerank"], args.chunk_size)
        r.model_pair = pair_info["name"]
        results.append(r)

        if args.run_pipeline:
            print("\n[5/5] Benchmarking End-to-End Pipeline ...")
            r = benchmark_pipeline(docs, queries, args.device, args.queries, args.batch_size,
                                   pair_info["embed"], pair_info["rerank"], args.chunk_size)
            r.model_pair = pair_info["name"]
            results.append(r)

        print_results(results)
        print_recommendations(results, pair_info, args.chunk_size)

    save_json(results, args.output, pair_info, args.chunk_size)
    print("\n🎉 Hoàn thành!")


if __name__ == "__main__":
    main()
