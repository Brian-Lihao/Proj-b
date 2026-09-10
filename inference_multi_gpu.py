import os
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from contextlib import nullcontext

import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU inference for PPR LoRA weights")

    parser.add_argument("--prompt_file", type=str, required=True, help="TXT file, one prompt per line")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")

    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to PEFT LoRA safetensors")

    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help='GPU list, e.g. "0,1,2,3". Use "auto" to use all visible GPUs.',
    )

    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)

    parser.add_argument("--precision", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=None)

    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["default", "ddim"],
        default="default",
        help="Use default SD1.5 scheduler or DDIM scheduler.",
    )

    parser.add_argument(
        "--enable_attention_slicing",
        action="store_true",
        help="Enable attention slicing to reduce VRAM usage.",
    )

    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Enable xformers memory efficient attention if installed.",
    )

    parser.add_argument(
        "--show_denoising_progress",
        action="store_true",
        help="Show diffusers denoising progress bars. Usually messy in multi-GPU mode.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images. If not set, existing images are skipped.",
    )

    return parser.parse_args()


def read_prompts(prompt_file):
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
    return prompts


def resolve_devices(gpus_arg):
    if gpus_arg.lower() == "auto":
        n = torch.cuda.device_count()
        if n <= 0:
            raise RuntimeError("No CUDA GPU detected. Please check your environment.")
        return list(range(n))

    devices = []
    for x in gpus_arg.split(","):
        x = x.strip()
        if x == "":
            continue
        devices.append(int(x))

    if len(devices) == 0:
        raise ValueError(f"Invalid --gpus value: {gpus_arg}")

    return devices


def split_prompts_round_robin(prompts, devices):
    chunks = [[] for _ in devices]
    for idx, prompt in enumerate(prompts):
        worker_idx = idx % len(devices)
        chunks[worker_idx].append((idx, prompt))
    return chunks


def load_peft_lora_into_unet(pipe, lora_path, lora_rank, lora_alpha):
    if lora_alpha is None:
        lora_alpha = lora_rank

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    peft_unet = get_peft_model(pipe.unet, lora_config)

    state_dict = load_file(lora_path)

    # Your saved LoRA keys look like:
    # unet.base_model.model.down_blocks...
    # set_peft_model_state_dict usually expects:
    # base_model.model.down_blocks...
    if all(k.startswith("unet.") for k in state_dict.keys()):
        state_dict = {k[len("unet."):]: v for k, v in state_dict.items()}

    incompatible = set_peft_model_state_dict(
        peft_unet,
        state_dict,
        adapter_name="default",
    )

    missing_keys = getattr(incompatible, "missing_keys", None)
    unexpected_keys = getattr(incompatible, "unexpected_keys", None)

    print("Loaded PEFT LoRA.")
    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)

    # Put the UNet with injected LoRA layers back into the pipeline.
    pipe.unet = peft_unet.base_model.model
    return pipe


def setup_pipeline(args_dict, device_id):
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)

    precision = args_dict["precision"]
    inference_dtype = torch.float16 if precision == "float16" else torch.float32

    huggingface_cache_dir = (
        os.environ.get("PPR_CACHE_DIR")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HOME")
        or os.environ.get("HUGGING_FACE_CACHE_DIR")
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        args_dict["model"],
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )

    pipe.safety_checker = None

    if args_dict["scheduler"] == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe = load_peft_lora_into_unet(
        pipe=pipe,
        lora_path=args_dict["lora_path"],
        lora_rank=args_dict["lora_rank"],
        lora_alpha=args_dict["lora_alpha"],
    )

    if args_dict["enable_attention_slicing"]:
        pipe.enable_attention_slicing()

    if args_dict["enable_xformers"]:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print(f"[GPU {device_id}] xformers enabled.")
        except Exception as e:
            print(f"[GPU {device_id}] Failed to enable xformers: {e}")

    pipe = pipe.to(device)

    pipe.set_progress_bar_config(
        disable=not args_dict["show_denoising_progress"]
    )

    return pipe, device


def image_filename(prompt_idx, image_idx, num_images_per_prompt, index_width):
    if num_images_per_prompt == 1:
        return f"{prompt_idx:0{index_width}d}.png"
    return f"{prompt_idx:0{index_width}d}_{image_idx:02d}.png"


def worker_process(worker_rank, device_id, assigned_prompts, args_dict):
    output_dir = Path(args_dict["output_dir"])
    metadata_dir = output_dir / "_metadata_shards"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    shard_path = metadata_dir / f"metadata_worker_{worker_rank}_gpu_{device_id}.jsonl"

    print(f"[Worker {worker_rank} | GPU {device_id}] Assigned prompts: {len(assigned_prompts)}")

    pipe, device = setup_pipeline(args_dict, device_id)

    precision = args_dict["precision"]
    autocast_ctx = torch.cuda.amp.autocast if precision == "float16" else nullcontext

    num_images_per_prompt = args_dict["num_images_per_prompt"]
    index_width = args_dict["index_width"]

    with open(shard_path, "w", encoding="utf-8") as meta_f:
        for local_i, (prompt_idx, prompt) in enumerate(assigned_prompts):
            print(
                f"[Worker {worker_rank} | GPU {device_id}] "
                f"{local_i + 1}/{len(assigned_prompts)} | global index {prompt_idx}"
            )

            for image_idx in range(num_images_per_prompt):
                cur_seed = args_dict["seed"] + prompt_idx * 10000 + image_idx

                filename = image_filename(
                    prompt_idx=prompt_idx,
                    image_idx=image_idx,
                    num_images_per_prompt=num_images_per_prompt,
                    index_width=index_width,
                )
                save_path = output_dir / filename

                if save_path.exists() and not args_dict["overwrite"]:
                    print(f"[Worker {worker_rank} | GPU {device_id}] Skip existing: {save_path}")
                    item = {
                        "index": prompt_idx,
                        "sub_index": image_idx,
                        "seed": cur_seed,
                        "file_name": filename,
                        "prompt": prompt,
                        "status": "skipped_existing",
                        "gpu": device_id,
                    }
                    meta_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    meta_f.flush()
                    continue

                generator = torch.Generator(device=device).manual_seed(cur_seed)

                with autocast_ctx():
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=args_dict["negative_prompt"],
                        generator=generator,
                        guidance_scale=args_dict["guidance_scale"],
                        num_inference_steps=args_dict["num_inference_steps"],
                        height=args_dict["height"],
                        width=args_dict["width"],
                    )

                image = result.images[0]
                image.save(save_path)

                item = {
                    "index": prompt_idx,
                    "sub_index": image_idx,
                    "seed": cur_seed,
                    "file_name": filename,
                    "prompt": prompt,
                    "status": "generated",
                    "gpu": device_id,
                    "guidance_scale": args_dict["guidance_scale"],
                    "num_inference_steps": args_dict["num_inference_steps"],
                    "height": args_dict["height"],
                    "width": args_dict["width"],
                    "scheduler": args_dict["scheduler"],
                    "model": args_dict["model"],
                    "lora_path": args_dict["lora_path"],
                }

                meta_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                meta_f.flush()

                print(f"[Worker {worker_rank} | GPU {device_id}] Saved: {save_path}")

    del pipe
    torch.cuda.empty_cache()

    print(f"[Worker {worker_rank} | GPU {device_id}] Done.")


def merge_metadata(output_dir):
    output_dir = Path(output_dir)
    metadata_dir = output_dir / "_metadata_shards"
    final_metadata_path = output_dir / "metadata.json"

    records = []

    if metadata_dir.exists():
        for jsonl_path in sorted(metadata_dir.glob("metadata_worker_*.jsonl")):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

    records.sort(key=lambda x: (x["index"], x["sub_index"]))

    with open(final_metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Merged metadata saved to: {final_metadata_path}")


def save_run_config(args, prompts, devices):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_copy_path = output_dir / "prompts_used.txt"
    with open(prompts_copy_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")

    run_config = vars(args).copy()
    run_config["resolved_devices"] = devices
    run_config["num_prompts"] = len(prompts)

    config_path = output_dir / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"Prompts copied to: {prompts_copy_path}")
    print(f"Run config saved to: {config_path}")


def main():
    args = parse_args()

    prompts = read_prompts(args.prompt_file)
    if len(prompts) == 0:
        raise ValueError(f"No valid prompts found in {args.prompt_file}")

    devices = resolve_devices(args.gpus)

    args.index_width = max(5, len(str(len(prompts) - 1)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(args, prompts, devices)

    chunks = split_prompts_round_robin(prompts, devices)

    args_dict = vars(args)

    print(f"Total prompts: {len(prompts)}")
    print(f"Using GPUs: {devices}")
    for worker_rank, device_id in enumerate(devices):
        print(f"GPU {device_id}: {len(chunks[worker_rank])} prompts")

    ctx = mp.get_context("spawn")
    processes = []

    for worker_rank, device_id in enumerate(devices):
        assigned_prompts = chunks[worker_rank]
        if len(assigned_prompts) == 0:
            continue

        p = ctx.Process(
            target=worker_process,
            args=(worker_rank, device_id, assigned_prompts, args_dict),
        )
        p.start()
        processes.append(p)

    failed = False
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed = True
            print(f"Process {p.pid} failed with exit code {p.exitcode}")

    merge_metadata(output_dir)

    if failed:
        raise RuntimeError("At least one worker process failed. Check logs above.")

    print("All workers finished successfully.")


if __name__ == "__main__":
    main()