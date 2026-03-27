# H-Tokenizer Scrutiny Notes

This lane keeps the `#549` model path unchanged and only swaps the tokenizer backend.

## Exactness claims
- `hnet_chunk_v1` stores exact UTF-8 byte payloads per token.
- Encoding is deterministic dynamic programming over that fixed byte inventory.
- Decoding is exact byte concatenation of token payloads.
- The validation split is unchanged at the document level; only tokenization differs.
- Custom-tokenizer runs embed the tokenizer manifest bytes into the exported artifact, so tokenizer bytes are counted.

## Proof script
Run:

```bash
py -3.11 scripts/prove_hnet_tokenizer_bpb.py --docs-jsonl data/docs_selected.jsonl --val-glob data/datasets/fineweb10B_hnet1024/fineweb_val_*.bin --tokenizer-path data/tokenizers/fineweb_hnet_1024.json --stock-sp-model data/tokenizers/fineweb_1024_bpe.model --stock-sp-val-glob data/datasets/fineweb10B_sp1024/fineweb_val_*.bin
```

The script checks:
- decoded H-tokenizer validation docs match the raw validation bytes exactly
- decoded byte count matches raw UTF-8 byte count on the first `50,000` docs exactly
- byte count from token LUTs matches decoded bytes exactly
- the stock SentencePiece byte-accounting path also matches the raw validation bytes exactly

## Hash anchors
- The H-tokenizer manifest stores `docs_sha256`.
- The dataset export manifest stores `docs_sha256_local`, sidecar hash, and per-shard SHA256 values.
- Any submission using this lane should copy those hashes into its record folder.
