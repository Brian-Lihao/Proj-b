import contextlib
import copy
import json
import math
import os
import re
from functools import partial

import torch
import torch.nn.functional as F
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from mmengine.config import Config
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, broadcast
from accelerate.logging import get_logger
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
from peft import LoraConfig
from peft.utils import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from ppr.preference_models import get_preference_model_func, get_compare_func
from ppr.datasets import build_dataset
from ppr.losses import (
    LPIPS_METHODS,
    PPR_METHODS,
    REFERENCE_LPIPS_METHODS,
    preference_loss,
    ppr_weight,
    validate_method,
)
from ppr.utils import (
    huggingface_cache_dir,
    UNET_CKPT_NAME,
    UNET_LORA_CKPT_NAME,
    gather_tensor_with_diff_shape,
)
from ppr.custom_diffusers import (
    multi_sample_pipeline,
    ddim_step_with_logprob,
)
from ppr.custom_diffusers.ddim_sampling import ddim_step_fetch_x0

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "configs/ppr_sd15.py",
    "Training configuration.",
)

logger = get_logger(__name__)


def resolve_resume_checkpoint(resume_from):
    """Resolve a checkpoint directory or the latest checkpoint in a run directory."""
    resume_path = os.path.normpath(os.path.expanduser(resume_from))
    if not os.path.isdir(resume_path):
        raise ValueError(f"Resume path is not a directory: {resume_path}")

    match = re.fullmatch(r"checkpoint_(\d+)", os.path.basename(resume_path))
    if match:
        return resume_path, int(match.group(1))

    checkpoints = []
    for entry in os.scandir(resume_path):
        match = re.fullmatch(r"checkpoint_(\d+)", entry.name)
        if entry.is_dir() and match:
            checkpoints.append((int(match.group(1)), entry.path))
    if not checkpoints:
        raise ValueError(f"No checkpoint_<EPOCH> directories found in {resume_path}")
    return max(checkpoints)[1], max(checkpoints)[0]


def validate_resume_checkpoint(checkpoint_dir, num_processes, logger):
    required_files = (UNET_LORA_CKPT_NAME, "optimizer.bin", "scaler.pt", "global_step.json")
    missing_files = [
        filename
        for filename in required_files
        if not os.path.isfile(os.path.join(checkpoint_dir, filename))
    ]
    if missing_files:
        raise ValueError(
            f"Incomplete checkpoint {checkpoint_dir}: missing {', '.join(missing_files)}"
        )

    missing_rng_ranks = [
        rank
        for rank in range(num_processes)
        if not os.path.isfile(os.path.join(checkpoint_dir, f"random_states_{rank}.pkl"))
    ]
    if missing_rng_ranks:
        logger.warning(
            "Legacy checkpoint is missing RNG states for ranks %s. Model, optimizer, "
            "and scaler state will resume, but exact random-stream reproducibility is unavailable.",
            missing_rng_ranks,
        )


def main(_):
    config = FLAGS.config
    config = Config(config.to_dict())
    method = validate_method(config.train.method)

    resume_epoch = None
    if config.resume_from:
        config.resume_from, resume_epoch = resolve_resume_checkpoint(config.resume_from)

    # timesteps used for training: [divert_start_step: num_sample_timesteps]
    divert_start_step = config.train.divert_start_step

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.wandb_project_name, 
            config=config, 
            init_kwargs={"wandb": {
                "name": config.run_name, 
                "entity": config.wandb_entity_name
            }}
        )
        os.makedirs(os.path.join(config.logdir, config.run_name), exist_ok=True)
        with open(os.path.join(config.logdir, config.run_name, "exp_config.py"), "w") as f:
            f.write(config.pretty_text)
    logger.info(f"Training method: {method}")
    logger.info(f"\n{config.pretty_text}")

    set_seed(config.seed, device_specific=True)
    
    # For mixed precision training we cast all non-trainable weigths (vae, text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, 
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained.model,
        subfolder="unet",
        cache_dir=huggingface_cache_dir,
    )
    pipeline.unet = unet
    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    if config.use_checkpointing:
        unet.enable_gradient_checkpointing()
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=2,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Sampling Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    
    preference_model_fn = get_preference_model_func(
        config.preference_model_func_cfg, accelerator.device
    )
    compare_func = get_compare_func(config.compare_func_cfg)

    lpips_fn = None
    if method in LPIPS_METHODS:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(accelerator.device).eval()
        lpips_fn.requires_grad_(False)

    vae_scale = float(getattr(pipeline.vae.config, "scaling_factor", 0.18215))
    lpips_size = int(config.train.lpips_size)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.vae.eval()
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        unet.to(accelerator.device, dtype=inference_dtype)
        unet.requires_grad_(False)
    else:
        unet.requires_grad_(True)
    #### Prepare reference model
    ref = copy.deepcopy(unet)
    ref.to(accelerator.device)
    ref.requires_grad_(False)
    
    if config.use_lora:
        unet_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        from peft import get_peft_model
        unet = get_peft_model(unet, unet_lora_config)
        if accelerator.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            for p in unet.parameters():
                if p.requires_grad:
                    p.data = p.to(torch.float32)
    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if isinstance(models[0], type(accelerator.unwrap_model(unet))):
            if accelerator.is_main_process:
                if config.use_lora:
                    unet_lora_layers_to_save = get_peft_model_state_dict(models[0])
                    lora_path = os.path.join(output_dir, UNET_LORA_CKPT_NAME)
                    torch.save(unet_lora_layers_to_save, lora_path)
                    logger.info(f"Saved LoRA checkpoint to {lora_path}")
                else:
                    models[0].save_pretrained(os.path.join(output_dir, UNET_CKPT_NAME))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if isinstance(models[0], type(accelerator.unwrap_model(unet))):
            if config.use_lora:
                unet_lora_layers_para = torch.load(os.path.join(input_dir, UNET_LORA_CKPT_NAME), map_location='cpu')
                incompatible_keys = set_peft_model_state_dict(models[0], unet_lora_layers_para, adapter_name="default")
                if getattr(incompatible_keys, 'unexpected_keys', []) == []:
                    logger.info(f"loaded unet_lora_layers_para from {os.path.join(input_dir, UNET_LORA_CKPT_NAME)}")
                else:
                    logger.warning(f"unet_lora_layers has unexpected_keys: {getattr(incompatible_keys, 'unexpected_keys', None)}")
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder=UNET_CKPT_NAME)
                models[0].register_to_config(**load_model.config)
                models[0].load_state_dict(load_model.state_dict())
                del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    trainable_params = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = optimizer_cls(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    prompt_dataset = build_dataset(config.dataset_cfg)
    collate_fn = partial(
        prompt_dataset.collate_fn,
        tokenizer=pipeline.tokenizer,
    )

    data_loader = torch.utils.data.DataLoader(
        prompt_dataset,
        collate_fn=collate_fn,
        batch_size=config.sample.sample_batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=config.dataloader_shuffle,
        pin_memory=config.dataloader_pin_memory,
        drop_last=config.dataloader_drop_last,
    )
    
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    # for some reason, autocast is necessary for non-lora training but not for lora training, and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with `accelerator`.
    unet, optimizer, data_loader = accelerator.prepare(unet, optimizer, data_loader)
        
    # Train!
    total_train_batch_size = (
        config.train.train_batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sampling batch size per device = {config.sample.sample_batch_size}")
    logger.info(f"  Training batch size per device = {config.train.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")

    if config.resume_from:
        validate_resume_checkpoint(config.resume_from, accelerator.num_processes, logger)
        logger.info(f"Resuming from completed epoch checkpoint: {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = resume_epoch + 1
        with open(os.path.join(config.resume_from, "global_step.json"), "r") as f:
            global_step = json.load(f)["global_step"]
    else:
        first_epoch = 0
        global_step = 0
    
    for epoch in tqdm(
        range(first_epoch, config.num_epochs),
        total=config.num_epochs,
        initial=first_epoch,
        disable=not accelerator.is_local_main_process,
        desc="Epoch",
        position=0,
    ):
        train_loss = 0.0
        train_ratio_win = 0.0
        train_ratio_lose = 0.0
        for batch in tqdm(
            data_loader, 
            disable=not accelerator.is_local_main_process,
            desc="Batch",
            position=1,
        ):
            #################### SAMPLING ####################
            unet.eval()
            pipeline.unet.eval()
            batch_size = batch['input_ids'].shape[0]
            prompt_ids = batch['input_ids']
            # encode prompts
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            sample_neg_prompt_embeds = neg_prompt_embed.repeat(batch_size, 1, 1)
            
            # prepare extra_info for the preference model
            extra_info = batch['extra_info']
            for k, v in extra_info.items():
                if isinstance(v, torch.Tensor):
                    other_dim = [1 for _ in range(v.dim() - 1)]
                    extra_info[k] = v.repeat(config.sample.num_sample_each_step, *other_dim)
                elif isinstance(v, list):
                    extra_info[k] = v * config.sample.num_sample_each_step
                else:
                    raise ValueError(f"Unknown type {type(v)} for extra_info[{k}]")
            sampling_lpips_kwargs = {}
            if method in PPR_METHODS:
                sampling_lpips_kwargs = {
                    "lpips_fn": lpips_fn,
                    "lpips_size": lpips_size,
                    "collect_pair_lpips": True,
                }

            with autocast():
                sampling_outputs = multi_sample_pipeline(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    divert_start_step=divert_start_step,
                    num_samples_each_step=config.sample.num_sample_each_step,
                    preference_model_fn=preference_model_fn,
                    compare_fn=compare_func,
                    extra_info=extra_info,
                    **sampling_lpips_kwargs,
                )

            pair_lpips = None
            if method in PPR_METHODS:
                (
                    timesteps,
                    current_latents,
                    next_latents,
                    prompt_embeds,
                    preference_score_logs,
                    pair_lpips,
                ) = sampling_outputs
            else:
                (
                    timesteps,
                    current_latents,
                    next_latents,
                    prompt_embeds,
                    preference_score_logs,
                ) = sampling_outputs

            preference_score_logs = accelerator.gather(preference_score_logs).detach()
            accelerator.log(
                {
                    "preference_scores_mean": preference_score_logs.mean().item(), 
                    "preference_scores_std": preference_score_logs.std().item(),
                },
                step=global_step,
            )
            del preference_score_logs
            
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()
                local_valid_samples_num_list = [
                    torch.tensor([next_latents.shape[0]], dtype=torch.int, device=accelerator.device) 
                    for _ in range(accelerator.num_processes)
                ]
                for process_idx in range(accelerator.num_processes):
                    broadcast(local_valid_samples_num_list[process_idx], from_process=process_idx)
                
                local_valid_samples_num_list = [sample_num.item() for sample_num in local_valid_samples_num_list]

                # total_valid_samples_num, 1
                timesteps = gather_tensor_with_diff_shape(timesteps, local_valid_samples_num_list)
                # total_valid_samples_num, 1, c, h, w
                current_latents = gather_tensor_with_diff_shape(current_latents, local_valid_samples_num_list)
                # total_valid_samples_num, 2, c, h, w
                next_latents = gather_tensor_with_diff_shape(next_latents, local_valid_samples_num_list)
                # total_valid_samples_num,1,l,c
                prompt_embeds = gather_tensor_with_diff_shape(prompt_embeds, local_valid_samples_num_list)
                if pair_lpips is not None:
                    # total_valid_samples_num, 1
                    pair_lpips = gather_tensor_with_diff_shape(pair_lpips, local_valid_samples_num_list)

            total_valid_samples_num = timesteps.shape[0]
            
            if total_valid_samples_num < accelerator.num_processes:
                continue
            
            sample = {
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": current_latents,  # x_t
                "next_latents": next_latents,  # x_{t-1}
            }
            if pair_lpips is not None:
                sample["pair_lpips"] = pair_lpips
            
            if accelerator.is_main_process:
                valid_perm = torch.randperm(total_valid_samples_num, device=accelerator.device)
                accelerator.wait_for_everyone()
                broadcast(valid_perm, from_process=0)
                accelerator.wait_for_everyone()
            else:
                valid_perm = torch.ones(
                    total_valid_samples_num,
                    dtype=torch.int,
                    device=accelerator.device,
                ) * -1
                accelerator.wait_for_everyone()
                broadcast(valid_perm, from_process=0)
                accelerator.wait_for_everyone()
                assert not torch.any(valid_perm == -1)
            
            num_items_per_gpu = total_valid_samples_num // accelerator.num_processes
            valid_start_index = accelerator.process_index * num_items_per_gpu
            valid_end_index = valid_start_index + num_items_per_gpu
            for key, value in sample.items():
                sample[key] = value[valid_perm]
                sample[key] = sample[key][valid_start_index: valid_end_index]
            del prompt_embeds
            del timesteps
            del current_latents
            del next_latents
            
            sample_0 = {}
            sample_1 = {}
            for key, value in sample.items():
                if value.shape[1] == 1:
                    sample_0[key] = value[:, 0]
                    sample_1[key] = value[:, 0]
                else:
                    sample_0[key] = value[:, 0]
                    sample_1[key] = value[:, 1]
            del sample

            # Log pairwise LPIPS statistics for calibrating PPR-Mid.
            if "pair_lpips" in sample_0:
                _d_log = accelerator.gather(sample_0["pair_lpips"].detach().reshape(-1).float())
                if _d_log.numel() > 0:
                    accelerator.log(
                        {
                            "lpips_pair_mean": _d_log.mean().item(),
                            "lpips_pair_std":  _d_log.std(unbiased=False).item(),
                            "lpips_pair_p50":  torch.quantile(_d_log, 0.50).item(),
                            "lpips_pair_p90":  torch.quantile(_d_log, 0.90).item(),
                            "lpips_pair_max":  _d_log.max().item(),
                        },
                        step=global_step,
                    )

            torch.cuda.empty_cache()
            
            num_train_batches = math.ceil(sample_0['latents'].shape[0] / config.train.train_batch_size)
            
            ############ Training ############
            unet.train()
            pipeline.unet.train()
            for train_batch_idx in tqdm(
                range(num_train_batches),
                desc="Training Small Batches",
                position=2,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                train_b_start = config.train.train_batch_size * train_batch_idx
                train_b_end = config.train.train_batch_size * (train_batch_idx + 1)
                if config.train.cfg:
                    train_neg_prompt_embeds = neg_prompt_embed.repeat(
                        sample_0["prompt_embeds"][train_b_start: train_b_end].shape[0], 
                        1, 1,
                    )
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds_0 = torch.cat([train_neg_prompt_embeds, sample_0["prompt_embeds"][train_b_start: train_b_end]])
                    embeds_1 = torch.cat([train_neg_prompt_embeds, sample_1["prompt_embeds"][train_b_start: train_b_end]])
                else:
                    embeds_0 = sample_0["prompt_embeds"][train_b_start: train_b_end]
                    embeds_1 = sample_1["prompt_embeds"][train_b_start: train_b_end]
                
                with accelerator.accumulate(unet):
                    with autocast():
                        if config.train.cfg:
                            noise_pred_0 = unet(
                                torch.cat([sample_0["latents"][train_b_start: train_b_end]] * 2),
                                torch.cat([sample_0["timesteps"][train_b_start: train_b_end]] * 2),
                                embeds_0,
                            ).sample
                            noise_pred_uncond_0, noise_pred_text_0 = noise_pred_0.chunk(2)
                            noise_pred_0 = noise_pred_uncond_0 + config.sample.guidance_scale * (
                                noise_pred_text_0 - noise_pred_uncond_0
                            )
                            noise_ref_pred_0 = ref(
                                torch.cat([sample_0["latents"][train_b_start: train_b_end]] * 2),
                                torch.cat([sample_0["timesteps"][train_b_start: train_b_end]] * 2),
                                embeds_0,
                            ).sample
                            noise_ref_pred_uncond_0, noise_ref_pred_text_0 = noise_ref_pred_0.chunk(2)
                            noise_ref_pred_0 = noise_ref_pred_uncond_0 + config.sample.guidance_scale * (
                                noise_ref_pred_text_0 - noise_ref_pred_uncond_0
                            )
                            
                            noise_pred_1 = unet(
                                torch.cat([sample_1["latents"][train_b_start: train_b_end]] * 2),
                                torch.cat([sample_1["timesteps"][train_b_start: train_b_end]] * 2),
                                embeds_1,
                            ).sample
                            noise_pred_uncond_1, noise_pred_text_1 = noise_pred_1.chunk(2)
                            noise_pred_1 = noise_pred_uncond_1 + config.sample.guidance_scale * (
                                noise_pred_text_1 - noise_pred_uncond_1
                            )
                            
                            noise_ref_pred_1 = ref(
                                torch.cat([sample_1["latents"][train_b_start: train_b_end]] * 2),
                                torch.cat([sample_1["timesteps"][train_b_start: train_b_end]] * 2),
                                embeds_1,
                            ).sample
                            noise_ref_pred_uncond_1, noise_ref_pred_text_1 = noise_ref_pred_1.chunk(2)
                            noise_ref_pred_1 = noise_ref_pred_uncond_1 + config.sample.guidance_scale * (
                                noise_ref_pred_text_1 - noise_ref_pred_uncond_1
                            )
                        else:
                            noise_pred_0 = unet(
                                sample_0["latents"][train_b_start: train_b_end], 
                                sample_0["timesteps"][train_b_start: train_b_end], 
                                embeds_0,
                            ).sample
                            noise_ref_pred_0 = ref(
                                sample_0["latents"][train_b_start: train_b_end], 
                                sample_0["timesteps"][train_b_start: train_b_end], 
                                embeds_0,
                            ).sample
                            
                            noise_pred_1 = unet(
                                sample_1["latents"][train_b_start: train_b_end], 
                                sample_1["timesteps"][train_b_start: train_b_end], 
                                embeds_1,
                            ).sample
                            noise_ref_pred_1 = ref(
                                sample_1["latents"][train_b_start: train_b_end], 
                                sample_1["timesteps"][train_b_start: train_b_end], 
                                embeds_1,
                            ).sample
                    
                    # compute the log prob of next_latents given latents under the current model
                    total_prob_0 = ddim_step_with_logprob(
                        pipeline.scheduler,
                        noise_pred_0,
                        sample_0["timesteps"][train_b_start: train_b_end],
                        sample_0["latents"][train_b_start: train_b_end],
                        eta=config.sample.eta,
                        prev_sample=sample_0["next_latents"][train_b_start: train_b_end],
                    )
                    total_ref_prob_0 = ddim_step_with_logprob(
                        pipeline.scheduler,
                        noise_ref_pred_0,
                        sample_0["timesteps"][train_b_start: train_b_end],
                        sample_0["latents"][train_b_start: train_b_end],
                        eta=config.sample.eta,
                        prev_sample=sample_0["next_latents"][train_b_start: train_b_end],
                    )
                    total_prob_1 = ddim_step_with_logprob(
                        pipeline.scheduler,
                        noise_pred_1,
                        sample_1["timesteps"][train_b_start: train_b_end],
                        sample_1["latents"][train_b_start: train_b_end],
                        eta=config.sample.eta,
                        prev_sample=sample_1["next_latents"][train_b_start: train_b_end],
                    )
                    total_ref_prob_1 = ddim_step_with_logprob(
                        pipeline.scheduler,
                        noise_ref_pred_1,
                        sample_1["timesteps"][train_b_start: train_b_end],
                        sample_1["latents"][train_b_start: train_b_end],
                        eta=config.sample.eta,
                        prev_sample=sample_1["next_latents"][train_b_start: train_b_end],
                    )
                    
                    # Reference-normalized likelihood ratios for winner and loser.
                    ratio_win = torch.clamp(
                        torch.exp(total_prob_0 - total_ref_prob_0),
                        1 - config.train.eps,
                        1 + config.train.eps,
                    )
                    ratio_lose = torch.clamp(
                        torch.exp(total_prob_1 - total_ref_prob_1),
                        1 - config.train.eps,
                        1 + config.train.eps,
                    )
                    log_ratio_diff = torch.log(ratio_win) - torch.log(ratio_lose)
                    base_loss = preference_loss(log_ratio_diff, config.train.beta)

                    if method == "spo":
                        loss = base_loss.mean()
                    elif method in PPR_METHODS:
                        distance = sample_0["pair_lpips"][train_b_start:train_b_end]
                        weight = ppr_weight(
                            distance,
                            method,
                            scale=config.train.ppr_lambda,
                            max_distance=config.train.lpips_max_distance,
                            mid_mu=config.train.ppr_mid_mu,
                            mid_sigma=config.train.ppr_mid_sigma,
                            hard_gamma=config.train.ppr_hard_gamma,
                        )
                        loss = (weight * base_loss).mean()
                    elif method in REFERENCE_LPIPS_METHODS:
                        # The reference image is detached; the current-policy image
                        # remains differentiable through the frozen VAE and LPIPS net.
                        vae_dtype = next(pipeline.vae.parameters()).dtype
                        current_latents = sample_0["latents"][train_b_start:train_b_end]
                        current_timesteps = sample_0["timesteps"][train_b_start:train_b_end]

                        with torch.no_grad():
                            reference_x0 = ddim_step_fetch_x0(
                                pipeline.scheduler,
                                noise_ref_pred_0.detach(),
                                current_timesteps,
                                current_latents,
                            )["pred_original_sample"]
                            reference_image = pipeline.vae.decode(
                                reference_x0.to(vae_dtype) / vae_scale,
                                return_dict=False,
                            )[0].float().clamp(-1.0, 1.0)
                            if lpips_size > 0 and reference_image.shape[-2:] != (
                                lpips_size,
                                lpips_size,
                            ):
                                reference_image = F.interpolate(
                                    reference_image,
                                    (lpips_size, lpips_size),
                                    mode="bilinear",
                                    align_corners=False,
                                )

                        current_x0 = ddim_step_fetch_x0(
                            pipeline.scheduler,
                            noise_pred_0,
                            current_timesteps,
                            current_latents,
                        )["pred_original_sample"]
                        current_image = pipeline.vae.decode(
                            current_x0.to(vae_dtype) / vae_scale,
                            return_dict=False,
                        )[0].float().clamp(-1.0, 1.0)
                        if lpips_size > 0 and current_image.shape[-2:] != (
                            lpips_size,
                            lpips_size,
                        ):
                            current_image = F.interpolate(
                                current_image,
                                (lpips_size, lpips_size),
                                mode="bilinear",
                                align_corners=False,
                            )

                        reference_distance = lpips_fn(
                            current_image, reference_image
                        ).reshape(-1)
                        reference_distance = torch.nan_to_num(
                            reference_distance,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        ).clamp_min(0.0)

                        regularizer = reference_distance.mean()
                        if method == "snr_lpips":
                            alphas = pipeline.scheduler.alphas_cumprod.to(
                                device=current_timesteps.device,
                                dtype=torch.float32,
                            )
                            alpha_t = alphas[current_timesteps].reshape(-1)
                            snr = alpha_t / (1.0 - alpha_t).clamp_min(1e-12)
                            confidence = (
                                snr / (snr + float(config.train.snr_tau))
                            ).detach()
                            regularizer = (confidence * reference_distance).mean()

                        loss = (
                            base_loss.mean()
                            + float(config.train.reference_lpips_lambda) * regularizer
                        )

                    avg_loss = accelerator.reduce(loss.detach(), reduction='mean')
                    train_loss += avg_loss.item() / accelerator.gradient_accumulation_steps
                    
                    # batch size              
                    win_ratio_sum =  accelerator.reduce(ratio_win.detach(), reduction='sum')       
                    lose_ratio_sum =  accelerator.reduce(ratio_lose.detach(), reduction='sum')       
                    
                    avg_win_ratio = (win_ratio_sum.sum() / (win_ratio_sum.shape[0] * accelerator.num_processes)).item()
                    avg_lose_ratio = (lose_ratio_sum.sum() / (lose_ratio_sum.shape[0] * accelerator.num_processes)).item()

                    train_ratio_win += avg_win_ratio / accelerator.gradient_accumulation_steps
                    train_ratio_lose += avg_lose_ratio / accelerator.gradient_accumulation_steps

                    # backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    # log training-related stuff
                    info = {
                        "epoch": epoch, 
                        "global_step": global_step, 
                        "train_loss": train_loss,
                        "train_ratio_win": train_ratio_win,
                        "train_ratio_lose": train_ratio_lose,
                        "lr": optimizer.param_groups[0]['lr'],
                    }
                    accelerator.log(info, step=global_step)
                    global_step += 1
                    train_loss = 0.0
                    train_ratio_win = 0.0
                    train_ratio_lose = 0.0
        
        # Save a complete distributed checkpoint at the epoch boundary.
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_dir = os.path.join(
                config.logdir, config.run_name, f"checkpoint_{epoch}"
            )
            accelerator.wait_for_everyone()
            accelerator.save_state(checkpoint_dir)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                global_step_path = os.path.join(checkpoint_dir, "global_step.json")
                with open(global_step_path, "w") as f:
                    json.dump({"global_step": global_step}, f)
            accelerator.wait_for_everyone()

        # Validation runs on the main process while all other ranks wait.
        should_validate = (
            (epoch + 1) % config.eval_interval == 0
            and config.validation_prompts is not None
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and should_validate:
            prompt_info = "Running validation...\nGenerating images with prompts:\n"
            prompt_info += "\n".join(config.validation_prompts)
            logger.info(prompt_info)
            unet.eval()
            pipeline.unet.eval()
            generator = (
                torch.Generator(device=accelerator.device).manual_seed(config.seed)
                if config.seed
                else None
            )

            image_logs = []
            for validation_prompt in config.validation_prompts:
                with torch.cuda.amp.autocast():
                    images = [
                        pipeline(
                            prompt=validation_prompt,
                            num_inference_steps=config.sample.num_steps,
                            generator=generator,
                            guidance_scale=config.sample.guidance_scale,
                        ).images[0]
                        for _ in range(config.num_validation_images)
                    ]
                image_logs.append({"images": images, "prompts": validation_prompt})

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    formatted_images = []
                    for log in image_logs:
                        for image in log["images"]:
                            formatted_images.append(
                                wandb.Image(image, caption=log["prompts"])
                            )
                    tracker.log({"validation": formatted_images})
            unet.train()
            pipeline.unet.train()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = get_peft_model_state_dict(unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=os.path.join(config.logdir, config.run_name),
            unet_lora_layers=unet_lora_state_dict,
        )
    
    accelerator.end_training()

if __name__ == "__main__":
    app.run(main)
