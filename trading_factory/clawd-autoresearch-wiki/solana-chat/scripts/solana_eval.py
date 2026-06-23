"""
Solana-specific evaluation script for solana-chat models.

Evaluates a trained model on Solana domain knowledge (MCQ format).
Can also run the standard CORE metric alongside Solana tasks.

Usage:
    python -m scripts.solana_eval --model-tag d12 --device-batch-size=16
    python -m scripts.solana_eval --hf-path NousResearch/Hermes-3-Llama-3.1-8B
    python -m scripts.solana_eval --hf-path Qwen/Qwen2.5-1.5B-Instruct \
      --adapter ../../../outputs/clawd-autoresearch-wiki-qwen15-lora-mac \
      --device-type mps
"""
import argparse
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import (autodetect_device_type, compute_init,
                             compute_cleanup, print0, get_base_dir)
from nanochat.tokenizer import HuggingFaceTokenizer
from solana.tasks import SolanaKnowledgeTask


def load_hf_model_simple(hf_path: str, device, adapter: str | None = None):
    """Load a HuggingFace model with minimal wrapper."""
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if str(device).startswith(("mps", "cuda")) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel
        print0(f"Attaching LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter, torch_dtype=dtype)
    model.to(device)
    model.eval()

    class ModelWrapper:
        def __init__(self, m):
            self.m = m
        def get_device(self):
            return next(self.m.parameters()).device
        def __call__(self, input_ids, targets=None, loss_reduction='mean'):
            logits = self.m(input_ids).logits
            if targets is None:
                return logits
            batch, seq_len = targets.shape
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            if loss_reduction == 'none':
                return loss.view(batch, seq_len)
            return loss

    return ModelWrapper(model)


def main():
    parser = argparse.ArgumentParser(description="Solana knowledge evaluation")
    parser.add_argument('--model-tag', type=str, default=None,
                        help='nanochat model tag')
    parser.add_argument('--step', type=int, default=None,
                        help='Model step to load')
    parser.add_argument('--hf-path', type=str, default=None,
                        help='HuggingFace model path')
    parser.add_argument('--adapter', type=str, default=None,
                        help='Optional PEFT/LoRA adapter path or Hub repo for --hf-path')
    parser.add_argument('--max-questions', type=int, default=-1,
                        help='Max questions to evaluate')
    parser.add_argument('--device-type', type=str, default='',
                        help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    device_type = (autodetect_device_type()
                   if args.device_type == '' else args.device_type)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(
        device_type)

    # Load model
    is_hf = args.hf_path is not None
    if is_hf:
        model = load_hf_model_simple(args.hf_path, device, adapter=args.adapter)
        from nanochat.tokenizer import HuggingFaceTokenizer
        tokenizer = HuggingFaceTokenizer.from_pretrained(args.hf_path)
    else:
        from nanochat.tokenizer import get_tokenizer
        model, _, meta = load_model("base", device, phase="eval",
                                    model_tag=args.model_tag, step=args.step)
        tokenizer = get_tokenizer()

    print0(f"Evaluating Solana knowledge...")
    print0(f"  Model: {'HF: ' + args.hf_path if is_hf else 'nanochat: ' + str(args.model_tag)}")
    if args.adapter:
        print0(f"  Adapter: {args.adapter}")
    print0(f"  Device: {device}")

    # Run Solana knowledge evaluation
    task = SolanaKnowledgeTask()
    results = task.evaluate(model, tokenizer, device,
                            max_questions=args.max_questions)

    sk = results["solana_knowledge"]
    print0(f"\n{'='*60}")
    print0(f"Solana Knowledge Benchmark Results")
    print0(f"{'='*60}")
    print0(f"  Overall: {sk['overall']:.4f} ({sk['overall']*100:.1f}%)")
    print0(f"  Questions: {sk['num_questions']}")
    print0(f"\n  Per-topic breakdown:")
    for topic, acc in sk["per_topic"].items():
        label = topic.replace('_', ' ').title()
        bar = '█' * int(acc * 20) + '░' * (20 - int(acc * 20))
        print0(f"    {label:<20} {acc:.4f}  {bar}")
    print0(f"{'='*60}")

    # Log to nanochat report if available
    try:
        from nanochat.report import get_report
        get_report().log(section="Solana evaluation", data=[
            {"model": args.model_tag or args.hf_path or "unknown"},
            {"solana_knowledge_overall": sk["overall"]},
            sk["per_topic"],
        ])
    except Exception:
        pass

    compute_cleanup()


if __name__ == "__main__":
    main()
