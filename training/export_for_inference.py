"""
Convert a training checkpoint (training/checkpoints/best.pt or latest.pt) into
the inference export the agents expect at training/finetuned_model/:

    finetuned_model/
      model.pt        <- raw model weights only (state_dict)
      config.json     <- model architecture config
      <tokenizer files>

Usage:
    python training/export_for_inference.py            # uses best.pt
    python training/export_for_inference.py latest.pt  # use a specific checkpoint
"""

import os
import sys
import json
import torch
from transformers import GPT2TokenizerFast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "training", "finetuned_model")


def main():
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    # Own file, trusted: load full dict (contains optimizer state etc.)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(f"  step={ckpt.get('step')}  epoch={ckpt.get('epoch')}  loss={ckpt.get('loss')}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) weights only
    torch.save(ckpt["model_state_dict"], os.path.join(OUT_DIR, "model.pt"))
    print(f"  wrote model.pt ({os.path.getsize(os.path.join(OUT_DIR, 'model.pt'))/1e6:.1f} MB)")

    # 2) model config
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(ckpt["model_config"], f, indent=2)
    print("  wrote config.json")

    # 3) tokenizer (GPT-2 BPE + tool-aware special tokens)
    sys.path.insert(0, PROJECT_ROOT)
    from training.model import add_special_tokens

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    add_special_tokens(tok)
    tok.save_pretrained(OUT_DIR)
    print(f"  wrote tokenizer files (vocab size: {len(tok)})")

    print(f"\nDone. Export ready at: {OUT_DIR}")
    print("Now run:  python agents/local_agent.py")


if __name__ == "__main__":
    main()
