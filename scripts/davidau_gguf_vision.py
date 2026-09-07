#!/usr/bin/env python3
"""Run the requested DavidAU IQ2_M model with its required vision projector."""
import argparse
from pathlib import Path

REPO = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
REVISION = "a51791d22b62b03aa5132feaac27147f32f289f7"
GGUF = "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-IQ2_M.gguf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--gpu-layers", type=int, default=-1)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    try:
        from llama_cpp.llama_chat_format import MTMDChatHandler
    except ImportError as e:
        raise RuntimeError("Install a current llama-cpp-python build exposing MTMDChatHandler before downloading model weights") from e
    model = hf_hub_download(REPO, GGUF, revision=REVISION)
    projector = hf_hub_download(REPO, "mmproj-F16.gguf", revision=REVISION)
    image_url = args.image
    if not image_url.startswith(("https://", "http://", "data:")):
        import base64
        import mimetypes
        path = Path(image_url)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        image_url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    handler = MTMDChatHandler(clip_model_path=projector)
    with Llama(model_path=model, chat_handler=handler, n_ctx=4096, n_gpu_layers=args.gpu_layers, verbose=False) as llm:
        result = llm.create_chat_completion(messages=[{"role": "user", "content": [
            {"type": "text", "text": args.prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}],
            max_tokens=256, temperature=0.2)
        print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
