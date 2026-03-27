from __future__ import annotations

import glob
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 131072))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 20))
    iterations = int(os.environ.get("ITERATIONS", 4096))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    model_family = os.environ.get("MODEL_FAMILY", "gdn")
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 1024))
    model_dim = int(os.environ.get("MODEL_DIM", 384))
    num_heads = int(os.environ.get("NUM_HEADS", 6))
    gdn_blocks = int(os.environ.get("GDN_BLOCKS", 8))
    gdn_attn_tail = int(os.environ.get("GDN_ATTN_TAIL", 1))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    batch_schedule = tuple(int(x) for x in os.environ.get("BATCH_SCHEDULE", "64,128,192").split(",") if x)
    batch_schedule_fracs = tuple(float(x) for x in os.environ.get("BATCH_SCHEDULE_FRACS", "0.15,0.45").split(",") if x)
    fast_loader = bool(int(os.environ.get("FAST_LOADER", "1")))
    adam_lr = float(os.environ.get("ADAM_LR", 1.8e-3))
    adam_wd = float(os.environ.get("ADAM_WD", 0.05))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))


def make_adamw(params, *, lr: float, wd: float, beta1: float, beta2: float, eps: float) -> torch.optim.Optimizer:
    kwargs = dict(lr=lr, weight_decay=wd, betas=(beta1, beta2), eps=eps)
    try:
        return torch.optim.AdamW(params, fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(params, **kwargs)


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self._advance_file()
                continue
            count = min(remaining, available)
            chunks.append(self.tokens[self.pos : self.pos + count])
            self.pos += count
            remaining -= count
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class FastLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device, pin_memory: bool):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.pin_memory = pin_memory
        self.stream = TokenStream(pattern)

    def next_batch(self, global_batch_seqs: int, seq_len: int) -> tuple[Tensor, Tensor]:
        local_batch_seqs = max(1, global_batch_seqs // self.world_size)
        local_tokens = local_batch_seqs * seq_len
        span = local_tokens + 1
        all_tokens = self.stream.take(span * self.world_size)
        start = self.rank * span
        local = all_tokens[start : start + span].to(dtype=torch.int64)
        if self.pin_memory and self.device.type == "cuda":
            local = local.pin_memory()
        local = local.to(self.device, non_blocking=True)
        x = local[:-1].reshape(local_batch_seqs, seq_len)
        y = local[1:].reshape(local_batch_seqs, seq_len)
        return x, y


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.weight.to(dtype=x.dtype) * F.rms_norm(x, (x.size(-1),), eps=self.eps)


class MLP(nn.Module):
    def __init__(self, dim: int, mult: float):
        super().__init__()
        hidden = int(dim * mult)
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.fc(x)))


class GatedDeltaMixer(nn.Module):
    def __init__(self, dim: int, num_heads: int, chunk: int = 64):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk = chunk
        self.in_proj = nn.Linear(dim, 5 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _scan(self, a: Tensor, u: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        prefix = torch.cumprod(a, dim=1)
        acc = torch.cumsum(u / prefix.clamp_min(1e-6), dim=1)
        states = prefix * (state.unsqueeze(1) + acc)
        return states, states[:, -1]

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        q, k, v, a, g = self.in_proj(x).chunk(5, dim=-1)

        def reshape(t: Tensor) -> Tensor:
            return t.view(bsz, seqlen, self.num_heads, self.head_dim)

        q, k, v, a, g = map(reshape, (q, k, v, a, g))
        q = F.rms_norm(q, (q.size(-1),))
        k = torch.tanh(k.float())
        v32 = v.float()
        a = torch.sigmoid(a.float() + 2.0).clamp(0.6, 0.9995)
        updates = k * v32
        state = torch.zeros(bsz, self.num_heads, self.head_dim, device=x.device, dtype=torch.float32)
        outputs: list[Tensor] = []
        for start in range(0, seqlen, self.chunk):
            end = min(start + self.chunk, seqlen)
            chunk_states, state = self._scan(a[:, start:end], updates[:, start:end], state)
            outputs.append(chunk_states)
        memory = torch.cat(outputs, dim=1).to(q.dtype)
        gate = torch.sigmoid(g)
        y = gate * (q * memory) + (1.0 - gate) * v
        return self.out_proj(y.reshape(bsz, seqlen, -1))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim)
        if _HAS_FLASH_ATTN and x.is_cuda:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
            ).transpose(1, 2)
        return self.c_proj(y.reshape(bsz, seqlen, dim))


class GDNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: float, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mixer = GatedDeltaMixer(dim, num_heads)
        self.mlp = MLP(dim, mlp_mult)
        self.res_scale = 1.0 / math.sqrt(2.0 * (layer_idx + 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.res_scale * self.mixer(self.norm1(x))
        return x + self.res_scale * self.mlp(self.norm2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: float, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.mlp = MLP(dim, mlp_mult)
        self.res_scale = 1.0 / math.sqrt(2.0 * (layer_idx + 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.res_scale * self.attn(self.norm1(x))
        return x + self.res_scale * self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        total_blocks = args.gdn_blocks + args.gdn_attn_tail
        if total_blocks <= 0:
            raise ValueError("GDN_BLOCKS + GDN_ATTN_TAIL must be positive")
        self.model_family = args.model_family
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        if args.model_family == "transformer":
            self.blocks = nn.ModuleList(
                [TransformerBlock(args.model_dim, args.num_heads, args.mlp_mult, i) for i in range(total_blocks)]
            )
        elif args.model_family == "gdn":
            self.blocks = nn.ModuleList(
                [GDNBlock(args.model_dim, args.num_heads, args.mlp_mult, i) for i in range(args.gdn_blocks)]
                + [
                    TransformerBlock(args.model_dim, args.num_heads, args.mlp_mult, args.gdn_blocks + i)
                    for i in range(args.gdn_attn_tail)
                ]
            )
        else:
            raise ValueError(f"Unsupported MODEL_FAMILY={args.model_family}")
        self.final_norm = RMSNorm(args.model_dim)
        self.lm_head = nn.Linear(args.model_dim, args.vocab_size, bias=True)
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights(args.model_dim)

    def _init_weights(self, model_dim: int) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.lm_head:
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / (5.0 * model_dim)))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.lm_head(self._hidden(input_ids))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        return self.lm_head(self._hidden(input_ids))


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = max(args.val_batch_size // world_size, seq_len)
    local_batch_seqs = max(1, local_batch_tokens // seq_len)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            start = batch_seq_start * seq_len
            end = batch_seq_end * seq_len + 1
            local = val_tokens[start:end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=_AUTODTYPE, enabled=True):
                batch_loss = model(x, y).detach()
            batch_tokens = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * batch_tokens
            token_count += batch_tokens
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = loss_sum / token_count
    val_bpb = (val_loss.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return float(val_loss.item()), float(val_bpb)


def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_start = (total_windows * rank) // world_size
    my_end = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_start:my_end]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[batch_start : batch_start + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws : end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=_AUTODTYPE, enabled=True):
                logits = model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                score_start = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, score_start:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - score_start)
                tgt = y_batch[i, score_start:wlen]
                prev = x_batch[i, score_start:wlen]
                token_bytes = base_bytes_lut[tgt].to(torch.float64)
                token_bytes += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += token_bytes.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return val_loss, val_bpb


def eval_val_sliding_ttt(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    log0=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    chunk_tokens = args.ttt_chunk_tokens
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        score_start = 0 if ws == 0 else max(wlen - stride, 0)
        scored_token = ws + score_start
        chunk_idx = min(scored_token // chunk_tokens, num_chunks - 1)
        chunk_windows[chunk_idx].append(ws)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params: list[Tensor] = []
    for name, param in base_model.named_parameters():
        freeze = any(f"blocks.{block_idx}." in name for block_idx in frozen_block_ids)
        param.requires_grad_(not freeze)
        if not freeze:
            ttt_params.append(param)
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()
    for chunk_idx, windows in enumerate(chunk_windows):
        if not windows:
            continue
        my_start = (len(windows) * rank) // world_size
        my_end = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_start:my_end]
        chunk_start = chunk_idx * chunk_tokens
        chunk_end = min((chunk_idx + 1) * chunk_tokens, total_tokens)
        base_model.eval()
        with torch.inference_mode():
            for batch_start in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[batch_start : batch_start + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk = val_tokens[ws : end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]
                with torch.autocast(device_type="cuda", dtype=_AUTODTYPE, enabled=True):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    score_start = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, score_start:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - score_start)
                    tgt = y_batch[i, score_start:wlen]
                    prev = x_batch[i, score_start:wlen]
                    token_bytes = base_bytes_lut[tgt].to(torch.float64)
                    token_bytes += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += token_bytes.sum()
        if chunk_idx < num_chunks - 1 and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * chunk_idx / max(num_chunks - 1, 1)))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                my_seq_start = (chunk_seqs * rank) // world_size
                my_seq_end = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_end - my_seq_start
                for _ in range(args.ttt_epochs):
                    for batch_start in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        batch_end = min(batch_start + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_start = chunk_start + (my_seq_start + batch_start) * seq_len
                        actual_end = chunk_start + (my_seq_start + batch_end) * seq_len + 1
                        if actual_end > val_tokens.numel():
                            continue
                        local = val_tokens[actual_start:actual_end].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=_AUTODTYPE, enabled=True):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for param in ttt_params:
                                if param.grad is not None:
                                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()
        if rank == 0 and (chunk_idx % 10 == 0 or chunk_idx == num_chunks - 1):
            val_loss = loss_sum.item() / max(token_count.item(), 1.0)
            val_bpb = (val_loss / math.log(2.0)) * (token_count.item() / max(byte_count.item(), 1.0))
            log0(f"ttt_chunk:{chunk_idx + 1}/{num_chunks} val_bpb:{val_bpb:.6f} elapsed:{time.perf_counter() - t0:.1f}s")
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
    for param in base_model.parameters():
        param.requires_grad_(True)
    base_model.eval()
    log0(f"ttt_done val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f} elapsed:{time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb


INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 0.999998


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict(state_dict: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, str], int]:
    result: dict[str, Tensor] = {}
    meta: dict[str, str] = {}
    payload_bytes = 0
    for name, tensor in state_dict.items():
        cpu = tensor.detach().cpu().contiguous()
        if cpu.is_floating_point() and cpu.ndim >= 2:
            q, scale = quantize_float_tensor(cpu)
            result[name + ".q"] = q
            result[name + ".scale"] = scale
            meta[name] = "int8"
            payload_bytes += q.numel() * q.element_size() + scale.numel() * scale.element_size()
        else:
            stored = cpu.to(torch.float16) if cpu.is_floating_point() else cpu
            result[name] = stored
            meta[name] = "passthrough"
            payload_bytes += stored.numel() * stored.element_size()
    return result, meta, payload_bytes


def dequantize_state_dict(blob: dict[str, Tensor], meta: dict[str, str], template: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, tensor in template.items():
        orig_dtype = tensor.dtype
        if meta[name] == "passthrough":
            restored = blob[name]
            if restored.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                restored = restored.to(orig_dtype)
            out[name] = restored
            continue
        q = blob[name + ".q"]
        scale = blob[name + ".scale"]
        if scale.ndim > 0:
            restored = q.float() * scale.float().view(q.shape[0], *([1] * (q.ndim - 1)))
        else:
            restored = q.float() * float(scale.item())
        out[name] = restored.to(orig_dtype)
    return out


def scheduled_global_batch(args: Hyperparameters, elapsed_seconds: float) -> int:
    if len(args.batch_schedule) == 1:
        return args.batch_schedule[0]
    if len(args.batch_schedule_fracs) != len(args.batch_schedule) - 1:
        raise ValueError("BATCH_SCHEDULE_FRACS must have len(BATCH_SCHEDULE)-1 entries")
    frac = elapsed_seconds / max(args.max_wallclock_seconds, 1e-9) if args.max_wallclock_seconds > 0 else 1.0
    for i, boundary in enumerate(args.batch_schedule_fracs):
        if frac < boundary:
            return args.batch_schedule[i]
    return args.batch_schedule[-1]


def metric_summary(**kwargs: float) -> str:
    return "final_metric_summary:" + json.dumps(kwargs, sort_keys=True)


def main() -> None:
    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl")
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    nvidia_smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    log0(nvidia_smi.stdout, console=False)
    git_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    git_dirty = subprocess.run(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    log0(f"git_commit:{git_commit.stdout.strip() or 'unknown'} git_dirty:{int(bool(git_dirty.stdout.strip()))}")
    log0("=" * 100, console=False)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    val_tokens = load_validation_tokens(args.val_files, max(args.train_seq_len, args.eval_seq_len))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    dataset_dir = Path(args.data_path).resolve()
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{len(list(dataset_dir.glob('fineweb_train_*.bin')))}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = GPT(args).to(device)
    model = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True) if distributed else base_model
    optimizer = make_adamw(
        model.parameters(),
        lr=args.adam_lr,
        wd=args.adam_wd,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.adam_eps,
    )
    train_loader = FastLoader(args.train_files, rank, world_size, device, pin_memory=args.fast_loader)
    total_blocks = args.gdn_blocks + args.gdn_attn_tail
    log0(
        f"model_family:{args.model_family} blocks:{total_blocks} gdn_blocks:{args.gdn_blocks} "
        f"attn_tail:{args.gdn_attn_tail} model_dim:{args.model_dim} num_heads:{args.num_heads}"
    )
    log0(
        f"batch_schedule:{','.join(map(str, args.batch_schedule))} "
        f"fractions:{','.join(map(str, args.batch_schedule_fracs))} fast_loader:{int(args.fast_loader)}"
    )
    log0(
        f"train_seq_len:{args.train_seq_len} eval_seq_len:{args.eval_seq_len} "
        f"iterations:{args.iterations} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"optimizer:adamw tf32=1 flash_attn_tail:{int(_HAS_FLASH_ATTN)} world_size:{world_size}")
    train_tokens_seen = 0
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                base_model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                eval_seq_len=args.eval_seq_len,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_seconds = training_time_ms / 1000.0 + (time.perf_counter() - t0)
        global_batch_seqs = scheduled_global_batch(args, elapsed_seconds)
        optimizer.zero_grad(set_to_none=True)
        x, y = train_loader.next_batch(global_batch_seqs, args.train_seq_len)
        with torch.autocast(device_type="cuda", dtype=_AUTODTYPE, enabled=True):
            loss = model(x, y)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer.step()
        train_tokens_seen += y.numel() * world_size
        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(
                f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} train_time:{approx_ms:.0f}ms "
                f"step_avg:{approx_ms / step:.2f}ms global_batch_seqs:{global_batch_seqs}"
            )
        reached_cap = args.max_wallclock_seconds > 0 and approx_ms >= 1000.0 * args.max_wallclock_seconds
        if distributed and args.max_wallclock_seconds > 0:
            flag = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            reached_cap = bool(flag.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"train_tokens_seen:{train_tokens_seen}")
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    base_model.eval()
    sd_cpu = {name: tensor.detach().cpu() for name, tensor in base_model.state_dict().items()}
    torch.cuda.synchronize()
    pre_t = time.perf_counter()
    pre_loss, pre_bpb = eval_val(
        args,
        base_model,
        rank,
        world_size,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        eval_seq_len=args.eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(f"final_pre_export val_loss:{pre_loss:.4f} val_bpb:{pre_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - pre_t):.0f}ms")
    log0(f"final_pre_export_exact val_loss:{pre_loss:.8f} val_bpb:{pre_bpb:.8f}")
    raw_state_buf = io.BytesIO()
    torch.save(sd_cpu, raw_state_buf)
    raw_state_bytes = raw_state_buf.getvalue()
    code_bytes = len(code.encode("utf-8"))
    quant_weights, quant_meta, payload_bytes = quantize_state_dict(sd_cpu)
    raw_quant_buf = io.BytesIO()
    torch.save({"w": quant_weights, "m": quant_meta}, raw_quant_buf)
    quant_raw = raw_quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        log0(f"Serialized model: {len(raw_state_bytes)} bytes")
        log0(
            f"Serialized model int8+zlib: {len(quant_blob)} bytes "
            f"(payload:{payload_bytes} packed:{len(quant_blob)} raw_torch:{len(quant_raw)} payload_ratio:{len(quant_blob) / max(payload_bytes, 1):.6f})"
        )
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size int8+zlib: {len(quant_blob) + code_bytes} bytes")
    if distributed:
        dist.barrier()
    restored_state = torch.load(
        io.BytesIO(zlib.decompress(Path("final_model.int8.ptz").read_bytes())),
        map_location="cpu",
        weights_only=False,
    )
    dequantized = dequantize_state_dict(restored_state["w"], restored_state["m"], sd_cpu)
    eval_model = GPT(args).to(device)
    eval_model.load_state_dict(dequantized, strict=True)
    eval_model.eval()
    torch.cuda.synchronize()
    qeval_t0 = time.perf_counter()
    post_loss, post_bpb = eval_val(
        args,
        eval_model,
        rank,
        world_size,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        eval_seq_len=args.eval_seq_len,
    )
    torch.cuda.synchronize()
    eval_ms = 1000.0 * (time.perf_counter() - qeval_t0)
    log0(f"final_int8_zlib_roundtrip val_loss:{post_loss:.4f} val_bpb:{post_bpb:.4f} eval_time:{eval_ms:.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{post_loss:.8f} val_bpb:{post_bpb:.8f}")
    pre_ttt_bpb = post_bpb
    if args.eval_stride > 0 and args.eval_stride < args.eval_seq_len:
        torch.cuda.synchronize()
        slide_t0 = time.perf_counter()
        slide_loss, slide_bpb = eval_val_sliding(
            args,
            eval_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=args.eval_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int8_zlib_sliding_window val_loss:{slide_loss:.4f} val_bpb:{slide_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - slide_t0):.0f}ms"
        )
        log0(f"final_int8_zlib_sliding_window_exact val_loss:{slide_loss:.8f} val_bpb:{slide_bpb:.8f}")
        pre_ttt_bpb = slide_bpb
    log0(f"pre_ttt_base_exact val_bpb:{pre_ttt_bpb:.8f}")
    summary = {
        "artifact_bytes_total": len(quant_blob) + code_bytes,
        "bytes_total": len(quant_blob) + code_bytes,
        "post_quant_val_bpb": round(post_bpb, 8),
        "pre_quant_val_bpb": round(pre_bpb, 8),
        "step_stop": step,
        "train_time_ms": round(training_time_ms, 3),
    }
    if args.ttt_enabled:
        torch.cuda.synchronize()
        ttt_t0 = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args,
            eval_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            log0=log0,
        )
        torch.cuda.synchronize()
        ttt_ms = 1000.0 * (time.perf_counter() - ttt_t0)
        log0(f"final_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{ttt_ms:.0f}ms")
        log0(f"final_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
        log0(f"ttt_gain:{ttt_bpb - pre_ttt_bpb:.8f}")
        summary["ttt_val_bpb"] = round(ttt_bpb, 8)
        summary["ttt_gain"] = round(ttt_bpb - pre_ttt_bpb, 8)
    log0(metric_summary(**summary))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    _AUTODTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    main()
