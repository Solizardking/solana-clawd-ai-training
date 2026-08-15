"""Official Qwen/Qwen3.8-27B multimodal load, chat, and generate helpers.

Weight I/O stays behind the factories named on the Hugging Face model card:

  pipeline("image-text-to-text", model="Qwen/Qwen3.8-27B")
  AutoProcessor.from_pretrained("Qwen/Qwen3.8-27B")
  AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.8-27B", device_map="auto")

This module does not import those classes at module load so dry-run and unit
tests can exercise the helpers without downloading the 27B checkpoint.
"""
from __future__ import annotations

from typing import Any

QWEN38_BASE_MODEL = "Qwen/Qwen3.8-27B"
QWEN38_PIPELINE_TASK = "image-text-to-text"
CANDY_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/p-blog/candy.JPG"
)
CANDY_PROMPT = "What animal is on the candy?"


def is_qwen38_multimodal_base(model_id: str | None) -> bool:
    return model_id == QWEN38_BASE_MODEL


def import_qwen38_transformers() -> dict[str, Any]:
    """Return the official factories. Isolated so tests can spy/patch them."""
    from transformers import AutoModelForMultimodalLM, AutoProcessor, pipeline

    return {
        "pipeline": pipeline,
        "AutoProcessor": AutoProcessor,
        "AutoModelForMultimodalLM": AutoModelForMultimodalLM,
    }


def construct_qwen38_pipeline(model_id: str = QWEN38_BASE_MODEL, **kwargs: Any) -> Any:
    factories = import_qwen38_transformers()
    return factories["pipeline"](QWEN38_PIPELINE_TASK, model=model_id, **kwargs)


def load_qwen38_processor_and_model(
    model_id: str = QWEN38_BASE_MODEL,
    device_map: str = "auto",
    **kwargs: Any,
) -> tuple[Any, Any]:
    factories = import_qwen38_transformers()
    processor = factories["AutoProcessor"].from_pretrained(model_id)
    model = factories["AutoModelForMultimodalLM"].from_pretrained(
        model_id,
        device_map=device_map,
        **kwargs,
    )
    return processor, model


def build_user_message(*, text: str, image_url: str | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image_url:
        content.append({"type": "image", "url": image_url})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def official_candy_messages() -> list[dict[str, Any]]:
    return [build_user_message(image_url=CANDY_IMAGE_URL, text=CANDY_PROMPT)]


def as_text_content(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def normalize_clawd_content(content: Any) -> list[dict[str, Any]] | Any:
    """Keep multimodal lists; wrap text-only Clawd strings as text content."""
    if isinstance(content, str):
        return [as_text_content(content)]
    return content


def normalize_clawd_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**message, "content": normalize_clawd_content(message.get("content", ""))}
        for message in messages
    ]


def apply_official_chat_template(processor: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    )


def official_generate(
    processor: Any,
    model: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 40,
) -> Any:
    """Official generate path: chat-template → device → generate → suffix decode."""
    inputs = apply_official_chat_template(processor, messages)
    to_device = getattr(inputs, "to", None)
    if callable(to_device):
        inputs = to_device(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.decode(outputs[0][prompt_len:])
