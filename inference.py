import os
import json
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for PPR LoRA weights")
    parser.add_argument("--prompt_file", type=str, required=True, help="TXT file, one prompt per line")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to PEFT LoRA safetensors")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--precision", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="How many images per prompt")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Optional negative prompt")
    return parser.parse_args()


def read_prompts(prompt_file):
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line != "":
                prompts.append(line)
    return prompts


def load_peft_lora_into_unet(pipe, lora_path, lora_rank):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    peft_unet = get_peft_model(pipe.unet, lora_config)

    state_dict = load_file(lora_path)

    # Diffusers may prefix PEFT keys with "unet." when saving LoRA weights.
    if all(k.startswith("unet.") for k in state_dict.keys()):
        state_dict = {k[len("unet."):]: v for k, v in state_dict.items()}

    incompatible = set_peft_model_state_dict(
        peft_unet,
        state_dict,
        adapter_name="default",
    )

    print("Loaded PEFT LoRA.")
    print("missing_keys:", getattr(incompatible, "missing_keys", None))
    print("unexpected_keys:", getattr(incompatible, "unexpected_keys", None))

    # Put the UNet with injected LoRA layers back into the pipeline.
    pipe.unet = peft_unet.base_model.model
    return pipe


def setup_pipeline(args):
    inference_dtype = torch.float16 if args.precision == "float16" else torch.float32
    huggingface_cache_dir = (
        os.environ.get("PPR_CACHE_DIR")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HOME")
        or os.environ.get("HUGGING_FACE_CACHE_DIR")
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )

    pipe.safety_checker = None

    pipe = load_peft_lora_into_unet(
        pipe=pipe,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
    )

    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def main():
    args = parse_args()

    prompts = read_prompts(args.prompt_file)
    if len(prompts) == 0:
        raise ValueError(f"No valid prompts found in {args.prompt_file}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact prompt list used for this run.
    prompts_copy_path = output_dir / "prompts_used.txt"
    with open(prompts_copy_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")

    pipe = setup_pipeline(args)
    torch.cuda.empty_cache()

    autocast_ctx = torch.cuda.amp.autocast if args.precision == "float16" else nullcontext

    metadata = []

    total = len(prompts)
    print(f"Loaded {total} prompts from {args.prompt_file}")

    for idx, prompt in enumerate(prompts):
        print(f"[{idx + 1}/{total}] Prompt: {prompt}")

        for img_j in range(args.num_images_per_prompt):
            cur_seed = args.seed + idx * 10000 + img_j
            generator = torch.Generator(device=args.device).manual_seed(cur_seed)

            with autocast_ctx():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                )

            image = result.images[0]

            if args.num_images_per_prompt == 1:
                filename = f"{idx:05d}.png"
            else:
                filename = f"{idx:05d}_{img_j:02d}.png"

            save_path = output_dir / filename
            image.save(save_path)

            item = {
                "index": idx,
                "sub_index": img_j,
                "seed": cur_seed,
                "file_name": filename,
                "prompt": prompt,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_inference_steps,
                "height": args.height,
                "width": args.width,
            }
            metadata.append(item)

            print(f"Saved: {save_path}")

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Images saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()