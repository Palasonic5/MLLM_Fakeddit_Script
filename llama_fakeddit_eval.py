import os
import re
import warnings
import argparse

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor

from dataloader import load_split, get_label_names
from prompts import get_prompts, parse_response
from metrics import compute_and_save

os.environ["HF_HOME"] = "/scratch/YOUR-NETID/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/YOUR-NETID/hf_cache"

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",  default="/scratch/YOUR-NETID/mllm/models/llama/llama-3.2-11B-Vision-Instruct")
parser.add_argument("--test_tsv",    default="/scratch/YOUR-NETID/Fakeddit/multimodal_only_samples/test.tsv")
parser.add_argument("--image_dir",   default="/scratch/YOUR-NETID/Fakeddit/public_image_set")
parser.add_argument("--output_path", default="/scratch/YOUR-NETID/Fakeddit/eval/outputs/llama_results.csv")
parser.add_argument("--plot_dir",    default="/scratch/YOUR-NETID/Fakeddit/eval/outputs/plots")
parser.add_argument("--label_type",  default="2_way", choices=["2_way", "6_way"])
parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples (e.g. 100 for a quick run)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading  (identical to llama_inference.py)
# ---------------------------------------------------------------------------
def load_model_and_processor(model_id):
    print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Loading model {model_id} in bfloat16 ...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Single inference  (identical message format to llama_inference.py)
# ---------------------------------------------------------------------------
def run_single_inference(model, processor, image_path, title, system_msg, user_template):
    image = Image.open(image_path).convert("RGB")
    user_prompt = user_template.format(title=title)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{system_msg}\n\n{user_prompt}"},
            ],
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading test split ({args.label_type}) ...")
    df = load_split(args.test_tsv, args.image_dir, label_type=args.label_type)
    label_names = get_label_names(args.label_type)
    system_msg, user_template = get_prompts(args.label_type)
    if args.max_samples is not None:
        df = df.head(args.max_samples)
        print(f"Using first {args.max_samples} samples (--max_samples).")
    print(f"Test samples: {len(df)}")

    # Load model
    model, processor = load_model_and_processor(args.model_path)

    # Inference
    raw_responses, predicted_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        try:
            response = run_single_inference(
                model, processor,
                row["image_path"], row["clean_title"],
                system_msg, user_template,
            )
            raw_responses.append(response)
            predicted_labels.append(parse_response(response, args.label_type))
        except Exception as e:
            raw_responses.append(f"ERROR: {e}")
            predicted_labels.append(None)

    # Save results
    df["raw_response"]     = raw_responses
    df["predicted_label"]  = predicted_labels
    df["correct"]          = df.apply(
        lambda r: int(r["label"]) == r["predicted_label"]
        if r["predicted_label"] is not None else None,
        axis=1,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"\nResults saved → {args.output_path}")

    # Evaluate
    prefix = f"llama_{args.label_type}_"
    compute_and_save(
        y_true=df["label"].tolist(),
        y_pred=predicted_labels,
        label_names=label_names,
        output_dir=args.plot_dir,
        prefix=prefix,
    )


if __name__ == "__main__":
    main()
