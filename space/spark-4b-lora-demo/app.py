"""ZeroGPU Gradio demo for ordlibrary/clawd-spark-4b-lora-pilot."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hashlib
import time
from pathlib import Path

import spaces
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

BASE = "XHToken/Spark-X2.5-4B"
BASE_REV = "5e10fcc0286756aebf7c41dc52c1e42d95c70281"
ADAPTER = "ordlibrary/clawd-spark-4b-lora-pilot"
ADAPTER_REV = "31b3053a70f4884b6107dfa0ded1a4d854889dab"
ADAPTER_SHA256 = "fa4d5d75e1fffe1a6ba1ac4f824162a6bc490a3559517c8b789bb8e2a8cb546f"
CODE_HASHES = {
    "configuration_spark.py": "02218597240490f490659b184052db940b08163ff9c3af7b7e59323f6f963722",
    "modeling_spark.py": "9cf0d1ad2b54b9f7088792779ddd8b5cb4d4fe63bfea054da7f2dd16362bcaf4",
}
CONTEXT = 4096
SYSTEM = (
    "You are Clawd Spark, a Solana-native assistant. Use exact on-chain field "
    "names. When the user asks for JSON, reply with JSON only."
)
EXAMPLES = [
    ['Which base Mint account field determines the decimal precision used for token amounts? Return JSON {"field": string}.'],
    ["A developer wrote mint.decimal_precision. What real member should replace decimal_precision? Return JSON with key field."],
    ['Does every Solana mint necessarily use eight decimals? Return JSON {"all_mints_use_eight": boolean}.'],
    ["Copy this raw amount without rounding or scientific notation: 18446744073709551615. Return JSON {\"raw_amount\": string}."],
    ["Copy the address So11111111111111111111111111111111111111112 exactly. Return JSON {\"address\": string}."],
    ["An external signing service can sign approved transactions. Does the language model need the wallet seed phrase in its prompt to explain a proposed transaction? Return JSON {\"seed_phrase_needed\": boolean}."],
]


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_model():
    root = Path(
        snapshot_download(
            BASE,
            revision=BASE_REV,
            allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.py", "*.model"],
        )
    )
    for name, digest in CODE_HASHES.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"Spark custom code hash mismatch: {name}")
    adapter_weights = hf_hub_download(
        ADAPTER, "adapter_model.safetensors", revision=ADAPTER_REV
    )
    actual_adapter = _sha256_file(adapter_weights)
    if actual_adapter != ADAPTER_SHA256:
        raise RuntimeError("Adapter SHA-256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        root,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, ADAPTER, revision=ADAPTER_REV)
    model = model.eval().to("cuda")
    print(
        f"Loaded {ADAPTER}@{ADAPTER_REV} on {BASE}@{BASE_REV} "
        f"adapter_sha256={actual_adapter}",
        flush=True,
    )
    return tokenizer, model


TOKENIZER, MODEL = load_model()


def _history_messages(history: list | None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in history or []:
        if isinstance(turn, dict) and turn.get("role") in {"user", "assistant"}:
            content = turn.get("content")
            if isinstance(content, str) and content:
                messages.append({"role": turn["role"], "content": content})
    return messages


@spaces.GPU(duration=120)
def chat(message: str, history: list | None = None, max_new_tokens: int = 256) -> tuple[str, list]:
    """Chat with the Clawd Spark 4B LoRA adapter.

    Args:
        message: User prompt. JSON-asking prompts match the pilot eval best.
        history: Prior Gradio chat turns.
        max_new_tokens: Cap on newly generated tokens (1–512).
    """
    prompt = (message or "").strip()
    history = list(history or [])
    if not prompt:
        return "", history
    max_new_tokens = max(1, min(int(max_new_tokens or 256), 512))
    messages = _history_messages(history) + [{"role": "user", "content": prompt}]
    text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = TOKENIZER(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[-1]
    if prompt_len + max_new_tokens > CONTEXT:
        reply = (
            f"Prompt ({prompt_len}) plus max_new_tokens ({max_new_tokens}) "
            f"exceeds the {CONTEXT}-token demo context."
        )
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})
        return "", history
    started = time.perf_counter()
    with torch.inference_mode():
        output = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_to_keep=1,
            pad_token_id=TOKENIZER.pad_token_id,
        )
    ids = output[0, prompt_len:].tolist()
    reply = TOKENIZER.decode(ids, skip_special_tokens=True)
    elapsed = time.perf_counter() - started
    print(
        f"generated tokens={len(ids)} prompt_tokens={prompt_len} seconds={elapsed:.2f}",
        flush=True,
    )
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": reply})
    return "", history


CSS = """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(
    theme=gr.themes.Citrus(),
    css=CSS,
    title="Clawd Spark 4B LoRA",
) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
# 🦞 Clawd Spark 4B LoRA

Private corrective adapter [`ordlibrary/clawd-spark-4b-lora-pilot`](https://huggingface.co/ordlibrary/clawd-spark-4b-lora-pilot)
on pinned [`XHToken/Spark-X2.5-4B`](https://huggingface.co/XHToken/Spark-X2.5-4B).

Pilot result: **10/10** fixed diagnostics after the mint-field correction
(`decimals`, not `decimal_precision`). This is a bounded eval demo, not a
trading or chart-OCR service.
            """
        )
        chatbot = gr.Chatbot(type="messages", height=420, show_label=False)
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask for a Solana mint field, an exact address copy, or JSON…",
                show_label=False,
                scale=4,
                container=False,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        with gr.Accordion("Settings", open=False):
            max_new_tokens = gr.Slider(32, 512, value=256, step=32, label="max_new_tokens")
        gr.Examples(examples=EXAMPLES, inputs=[msg], cache_examples=False)
        send.click(
            chat,
            inputs=[msg, chatbot, max_new_tokens],
            outputs=[msg, chatbot],
            api_name="chat",
            concurrency_limit=1,
        )
        msg.submit(
            chat,
            inputs=[msg, chatbot, max_new_tokens],
            outputs=[msg, chatbot],
            concurrency_limit=1,
        )

if __name__ == "__main__":
    demo.launch(mcp_server=True)
