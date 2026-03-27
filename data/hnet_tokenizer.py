from __future__ import annotations

import base64
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


WHITESPACE = set(b" \n\r\t")
PUNCT = set(b",.;:!?()[]{}<>\"'`|/\\+-=*~@#$%^&")


def sha256_path(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_docs_bytes(path: Path, max_docs: int | None = None) -> Iterable[bytes]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            yield json.loads(line)["text"].encode("utf-8", errors="replace")


def split_microchunks(data: bytes) -> list[bytes]:
    chunks: list[bytes] = []
    i, n = 0, len(data)
    while i < n:
        b = data[i]
        if b in WHITESPACE:
            j = i + 1
            while j < n and data[j] in WHITESPACE:
                j += 1
            chunks.append(data[i:j]); i = j; continue
        if b in PUNCT:
            chunks.append(bytes((b,))); i += 1; continue
        j = i + 1
        while j < n and data[j] not in WHITESPACE and data[j] not in PUNCT:
            j += 1
        chunks.append(data[i:j]); i = j
    return [chunk for chunk in chunks if chunk]


def _candidate_pre_score(payload: bytes, count: int) -> tuple[float, int, int, bytes]:
    return (count * max(len(payload) - 1, 1), count, len(payload), payload)


def _top_payloads(counter: Counter[bytes], limit: int, min_freq: int) -> set[bytes]:
    scored = [(_candidate_pre_score(payload, count), payload) for payload, count in counter.items() if count >= min_freq]
    scored.sort(key=lambda item: item[0], reverse=True)
    return {payload for _, payload in scored[:limit]}


def _gravity_score(payload: bytes, count: int, lefts: set[bytes], rights: set[bytes], parts: int, stage: int) -> float:
    saved = max(len(payload) - 1, 1)
    ctx = 1.0 + 0.35 * math.log1p(len(lefts)) + 0.35 * math.log1p(len(rights))
    stage_boost = 1.12 if stage == 2 else 1.0
    part_boost = 1.0 + 0.12 * max(parts - 1, 0)
    return float(count) * saved * ctx * stage_boost * part_boost


def _score_stage1(docs_jsonl: Path, max_docs: int, min_bytes: int, max_bytes: int, top_k: int, min_freq: int) -> list[dict[str, Any]]:
    counts: Counter[bytes] = Counter()
    for doc in iter_docs_bytes(docs_jsonl, max_docs):
        for chunk in split_microchunks(doc):
            if min_bytes <= len(chunk) <= max_bytes:
                counts[chunk] += 1
    keep = _top_payloads(counts, top_k, min_freq)
    if not keep:
        return []
    ctx = {payload: [0, set(), set()] for payload in keep}
    for doc in iter_docs_bytes(docs_jsonl, max_docs):
        chunks = split_microchunks(doc)
        for i, chunk in enumerate(chunks):
            entry = ctx.get(chunk)
            if entry is None:
                continue
            entry[0] += 1
            entry[1].add(chunks[i - 1] if i else b"")
            entry[2].add(chunks[i + 1] if i + 1 < len(chunks) else b"")
    out = []
    for payload, (count, lefts, rights) in ctx.items():
        out.append({
            "payload": payload,
            "count": count,
            "lefts": len(lefts),
            "rights": len(rights),
            "parts": 1,
            "stage": 1,
            "score": _gravity_score(payload, count, lefts, rights, 1, 1),
        })
    out.sort(key=lambda item: (item["score"], len(item["payload"]), item["payload"]), reverse=True)
    return out


def _score_stage2(
    docs_jsonl: Path,
    max_docs: int,
    min_bytes: int,
    max_bytes: int,
    max_parts: int,
    top_k: int,
    min_freq: int,
) -> list[dict[str, Any]]:
    counts: Counter[bytes] = Counter()
    part_counts: dict[bytes, int] = {}
    for doc in iter_docs_bytes(docs_jsonl, max_docs):
        chunks = split_microchunks(doc)
        for i in range(len(chunks)):
            joined = b""
            for parts in range(1, max_parts + 1):
                j = i + parts - 1
                if j >= len(chunks):
                    break
                joined += chunks[j]
                if parts >= 2 and min_bytes <= len(joined) <= max_bytes:
                    counts[joined] += 1
                    part_counts[joined] = parts
                if len(joined) > max_bytes:
                    break
    keep = _top_payloads(counts, top_k, min_freq)
    if not keep:
        return []
    ctx = {payload: [0, set(), set(), part_counts[payload]] for payload in keep}
    for doc in iter_docs_bytes(docs_jsonl, max_docs):
        chunks = split_microchunks(doc)
        for i in range(len(chunks)):
            joined = b""
            for parts in range(1, max_parts + 1):
                j = i + parts - 1
                if j >= len(chunks):
                    break
                joined += chunks[j]
                entry = ctx.get(joined)
                if entry is not None and parts >= 2:
                    entry[0] += 1
                    entry[1].add(chunks[i - 1] if i else b"")
                    entry[2].add(chunks[j + 1] if j + 1 < len(chunks) else b"")
                if len(joined) > max_bytes:
                    break
    out = []
    for payload, (count, lefts, rights, parts) in ctx.items():
        out.append({
            "payload": payload,
            "count": count,
            "lefts": len(lefts),
            "rights": len(rights),
            "parts": parts,
            "stage": 2,
            "score": _gravity_score(payload, count, lefts, rights, parts, 2),
        })
    out.sort(key=lambda item: (item["score"], len(item["payload"]), item["payload"]), reverse=True)
    return out


def _select_learned_tokens(stage1: list[dict[str, Any]], stage2: list[dict[str, Any]], learned_slots: int) -> list[dict[str, Any]]:
    quota1 = min(len(stage1), learned_slots * 2 // 5)
    quota2 = min(len(stage2), learned_slots - quota1)
    picked: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for pool, limit in ((stage1, quota1), (stage2, quota2), (stage2 + stage1, learned_slots)):
        for item in pool:
            payload = item["payload"]
            if payload in seen or len(payload) <= 1:
                continue
            picked.append(item)
            seen.add(payload)
            if len(picked) >= limit:
                break
        if len(picked) >= learned_slots:
            break
    return picked[:learned_slots]


@dataclass
class HNetChunkTokenizer:
    name: str
    dataset_suffix: str
    vocab_size: int
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    byte_offset: int = 4
    byte_count: int = 256
    token_bytes: list[bytes] | None = None
    token_scores: list[float] | None = None
    trainer_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.vocab_size != self.byte_offset + self.byte_count + (len(self.token_bytes or []) - (self.byte_offset + self.byte_count)):
            pass
        if self.token_bytes is None or self.token_scores is None:
            raise ValueError("token_bytes and token_scores must be provided")
        self._trie: dict[int, Any] = {}
        for token_id in range(self.byte_offset, self.vocab_size):
            payload = self.token_bytes[token_id]
            if not payload:
                continue
            node = self._trie
            for b in payload:
                node = node.setdefault(int(b), {})
            node["_id"] = token_id

    def encode(self, text: str) -> np.ndarray:
        data = text.encode("utf-8", errors="replace")
        n = len(data)
        best_score = [-1e30] * (n + 1)
        best_score[n] = 0.0
        best_id = [self.byte_offset] * max(n, 1)
        best_len = [1] * max(n, 1)
        for i in range(n - 1, -1, -1):
            byte_id = self.byte_offset + data[i]
            score = self.token_scores[byte_id] + best_score[i + 1]
            tok_id, tok_len = byte_id, 1
            node = self._trie.get(int(data[i]))
            j = i + 1
            while node is not None:
                candidate_id = node.get("_id")
                if candidate_id is not None:
                    cand_score = self.token_scores[candidate_id] + best_score[j]
                    cand_len = len(self.token_bytes[candidate_id])
                    if cand_score > score or (cand_score == score and (cand_len > tok_len or (cand_len == tok_len and candidate_id < tok_id))):
                        score, tok_id, tok_len = cand_score, candidate_id, cand_len
                if j >= n:
                    break
                node = node.get(int(data[j]))
                j += 1
            best_score[i], best_id[i], best_len[i] = score, tok_id, tok_len
        out = []
        i = 0
        while i < n:
            out.append(best_id[i]); i += best_len[i]
        return np.asarray(out, dtype=np.uint16)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(text) for text in texts]

    def decode_token_ids(self, token_ids: Iterable[int]) -> bytes:
        return b"".join(self.token_bytes[int(token_id)] for token_id in token_ids if int(token_id) >= self.byte_offset)

    def build_byte_luts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = np.fromiter((len(payload) for payload in self.token_bytes), dtype=np.int16, count=self.vocab_size)
        leading = np.zeros((self.vocab_size,), dtype=np.bool_)
        boundary = np.zeros((self.vocab_size,), dtype=np.bool_)
        boundary[[self.pad_id, self.bos_id, self.eos_id, self.unk_id]] = True
        return base, leading, boundary

    def save_json(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> None:
        learned = []
        for token_id in range(self.byte_offset + self.byte_count, self.vocab_size):
            payload = self.token_bytes[token_id]
            if not payload:
                continue
            learned.append({
                "id": token_id,
                "bytes_b64": base64.b64encode(payload).decode("ascii"),
                "score": self.token_scores[token_id],
            })
        payload: dict[str, Any] = {
            "tokenizer_type": "hnet_chunk_v1",
            "version": 1,
            "name": self.name,
            "dataset_suffix": self.dataset_suffix,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "byte_offset": self.byte_offset,
            "byte_count": self.byte_count,
            "learned_tokens": learned,
            "trainer_config": self.trainer_config or {},
        }
        if extra_meta:
            payload.update(extra_meta)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "HNetChunkTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("tokenizer_type") != "hnet_chunk_v1":
            raise ValueError(f"Unsupported tokenizer_type: {payload.get('tokenizer_type')}")
        vocab_size = int(payload["vocab_size"])
        byte_offset = int(payload.get("byte_offset", 4))
        byte_count = int(payload.get("byte_count", 256))
        token_bytes = [b""] * vocab_size
        token_scores = [0.0] * vocab_size
        for token_id in range(byte_offset, min(byte_offset + byte_count, vocab_size)):
            token_bytes[token_id] = bytes((token_id - byte_offset,))
        for item in payload.get("learned_tokens", []):
            token_id = int(item["id"])
            token_bytes[token_id] = base64.b64decode(item["bytes_b64"])
            token_scores[token_id] = float(item.get("score", 1.0))
        return cls(
            name=str(payload.get("name", f"hnet_chunk_v1_{vocab_size}")),
            dataset_suffix=str(payload.get("dataset_suffix", f"hnet{vocab_size}")),
            vocab_size=vocab_size,
            pad_id=int(payload.get("pad_id", 0)),
            bos_id=int(payload.get("bos_id", 1)),
            eos_id=int(payload.get("eos_id", 2)),
            unk_id=int(payload.get("unk_id", 3)),
            byte_offset=byte_offset,
            byte_count=byte_count,
            token_bytes=token_bytes,
            token_scores=token_scores,
            trainer_config=dict(payload.get("trainer_config", {})),
        )


def build_hnet_chunk_tokenizer(*, spec: dict[str, Any], docs_jsonl: Path, tokenizers_dir: Path) -> dict[str, Any]:
    vocab_size = int(spec["vocab_size"])
    byte_offset = int(spec.get("byte_offset", 4))
    byte_count = int(spec.get("byte_count", 256))
    learned_slots = vocab_size - byte_offset - byte_count
    if learned_slots <= 0:
        raise ValueError(f"hnet_chunk_v1 requires vocab_size > {byte_offset + byte_count}")
    max_docs = int(spec.get("tokenizer_train_docs", 200_000))
    stage1 = _score_stage1(
        docs_jsonl,
        max_docs,
        int(spec.get("stage1_min_bytes", 2)),
        int(spec.get("stage1_max_bytes", 24)),
        int(spec.get("stage1_top_k", 4096)),
        int(spec.get("stage1_min_freq", 4)),
    )
    stage2 = _score_stage2(
        docs_jsonl,
        max_docs,
        int(spec.get("stage2_min_bytes", 3)),
        int(spec.get("stage2_max_bytes", 32)),
        int(spec.get("stage2_max_parts", 4)),
        int(spec.get("stage2_top_k", 6144)),
        int(spec.get("stage2_min_freq", 3)),
    )
    learned = _select_learned_tokens(stage1, stage2, learned_slots)
    token_bytes = [b""] * vocab_size
    token_scores = [0.0] * vocab_size
    for token_id in range(byte_offset, byte_offset + byte_count):
        token_bytes[token_id] = bytes((token_id - byte_offset,))
    for i, item in enumerate(learned, start=byte_offset + byte_count):
        token_bytes[i] = item["payload"]
        token_scores[i] = float(math.log1p(item["score"]))
    path = tokenizers_dir / spec.get("filename", f"fineweb_hnet_{vocab_size}.json")
    trainer_config = {
        "tokenizer_train_docs": max_docs,
        "stage1_top_k": int(spec.get("stage1_top_k", 4096)),
        "stage2_top_k": int(spec.get("stage2_top_k", 6144)),
        "stage1_min_bytes": int(spec.get("stage1_min_bytes", 2)),
        "stage1_max_bytes": int(spec.get("stage1_max_bytes", 24)),
        "stage2_min_bytes": int(spec.get("stage2_min_bytes", 3)),
        "stage2_max_bytes": int(spec.get("stage2_max_bytes", 32)),
        "stage2_max_parts": int(spec.get("stage2_max_parts", 4)),
    }
    tok = HNetChunkTokenizer(
        name=str(spec.get("name", f"hnet_chunk_v1_{vocab_size}")),
        dataset_suffix=str(spec.get("dataset_suffix", f"hnet{vocab_size}")),
        vocab_size=vocab_size,
        byte_offset=byte_offset,
        byte_count=byte_count,
        token_bytes=token_bytes,
        token_scores=token_scores,
        trainer_config=trainer_config,
    )
    tok.save_json(path, extra_meta={"docs_sha256": sha256_path(docs_jsonl)})
    return {
        "name": tok.name,
        "kind": "hnet_chunk_v1",
        "dataset_suffix": tok.dataset_suffix,
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "encode": tok.encode,
        "encode_batch": tok.encode_batch,
        "recommended_bigram_vocab_size": int(spec.get("recommended_bigram_vocab_size", 1536)),
        "manifest": {"path": str(path)},
    }
