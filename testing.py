from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("examples/experiment.yaml").read_text(encoding="utf-8"))
cfg["datasets"][0]["path"] = "examples/eval_samples.jsonl"

cfg["pipelines"][0]["embedder"] = {
    "provider": "openrouter",
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "batch_size": 8,
    "metadata": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1/embeddings",
        "input_type": "multimodal_text",
    },
}

cfg["pipelines"][0]["reranker"] = {
    "type": "openrouter",
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
    "top_k": 3,
    "metadata": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1/rerank",
    },
}

cfg["pipelines"][0]["generator"] = {
    "provider": "openrouter",
    "api_format": "openai_compatible",
    "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "temperature": 0.0,
    "max_tokens": 256,
    "cost_usd": 0.0,
}

cfg["pipelines"][0]["sweep"] = {
    "enabled": True,
    "overrides": [{"retriever": {"top_k": 7}}],
}

Path("/tmp/phase4_smoke.yaml").write_text(
    yaml.safe_dump(cfg, sort_keys=False),
    encoding="utf-8",
)
