from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
from .ddim_sampling import ddim_step_fetch_x0, ddim_step_fetch_x_t_1


def _resize_lpips(images: torch.Tensor, size: int) -> torch.Tensor:
    if size and size > 0 and images.shape[-2:] != (size, size):
        images = F.interpolate(images, (size, size), mode="bilinear", align_corners=False)
    return images.float().clamp(-1.0, 1.0)


def _gather_selected_images(
    images_flat: torch.Tensor,
    indices: torch.Tensor,
    valid_samples: torch.Tensor,
    num_samples_each_step: int,
):
    """Gather winner/loser images selected by compare_fn.

    images_flat: [K*B, C, H, W], in K-major order.
    indices:     [2, B], candidate index of winner/loser per prompt.
    valid_samples: [B] bool mask.
    """
    batch_size = indices.shape[1]
    images_kb = images_flat.reshape(
        num_samples_each_step,
        batch_size,
        *images_flat.shape[1:],
    )
    gather_index = indices[..., None, None, None].expand(
        2,
        batch_size,
        *images_flat.shape[1:],
    )
    selected = torch.gather(images_kb, dim=0, index=gather_index)
    selected = selected[:, valid_samples]
    return selected[0], selected[1]


def _selected_pair_lpips(
    images_flat: torch.Tensor,
    indices: torch.Tensor,
    valid_samples: torch.Tensor,
    num_samples_each_step: int,
    lpips_fn,
    lpips_size: int,
) -> torch.Tensor:
    """LPIPS of the selected preferred/dispreferred predicted-clean images.

    Returns [valid_num, 1]. The enclosing pipeline runs under no_grad, so this
    tensor is a detached sampling-side signal by construction.
    """
    valid_num = int(valid_samples.sum().item())
    if valid_num == 0:
        return torch.empty((0, 1), device=images_flat.device, dtype=torch.float32)

    winner, loser = _gather_selected_images(
        images_flat,
        indices,
        valid_samples,
        num_samples_each_step,
    )
    winner = _resize_lpips(winner, lpips_size)
    loser = _resize_lpips(loser, lpips_size)
    d = lpips_fn(winner, loser).reshape(-1, 1)
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return d.float()



@torch.no_grad()
def multi_sample_pipeline(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    divert_start_step=0,
    num_samples_each_step=2,
    preference_model_fn=None,
    compare_fn=None,
    extra_info=None,
    lpips_fn=None,
    lpips_size: int = 256,
    collect_pair_lpips: bool = False,
    **kwargs,
):
    if collect_pair_lpips and lpips_fn is None:
        raise ValueError("lpips_fn is required when collect_pair_lpips is enabled")

    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )
    log_prompt_embeds = prompt_embeds
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

    all_prev_latents = None
    current_latents = latents if divert_start_step == 0 else None
    last_timestep = None
    denoise_idx = None

    valid_timesteps = []
    valid_current_latents = []
    valid_next_latents = []
    valid_prompt_embeds = []
    preference_score_logs = []
    valid_pair_lpips = []

    with self.progress_bar(total=timesteps.shape[0]) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_text,
                    guidance_rescale=guidance_rescale,
                )

            if i >= divert_start_step:
                pred_dict = ddim_step_fetch_x0(
                    self.scheduler,
                    noise_pred,
                    t,
                    latents,
                )

                if i == divert_start_step:
                    prev_latents = ddim_step_fetch_x_t_1(
                        self.scheduler,
                        dtype=latents.dtype,
                        num_sample_per_step=num_samples_each_step,
                        timestep=t,
                        **extra_step_kwargs,
                        **pred_dict,
                    )
                    if do_classifier_free_guidance:
                        prompt_embeds = torch.cat([
                            negative_prompt_embeds.repeat(num_samples_each_step, 1, 1),
                            log_prompt_embeds.repeat(num_samples_each_step, 1, 1),
                        ])

                elif i > divert_start_step:
                    # PPR reuses the predicted-clean images already decoded for the
                    # step-aware preference model, avoiding another VAE decode.
                    pred_x0_latents = pred_dict["pred_original_sample"]
                    pred_images = self.vae.decode(
                        pred_x0_latents.to(self.vae.dtype) / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]

                    preference_timestep = t.repeat(pred_images.shape[0])
                    extra_info["timesteps"] = preference_timestep
                    preference_scores_flat = preference_model_fn(pred_images, extra_info)
                    preference_score_logs.append(preference_scores_flat)
                    preference_scores = preference_scores_flat.reshape(num_samples_each_step, -1)

                    indices, valid_samples = compare_fn(preference_scores)

                    if collect_pair_lpips:
                        valid_pair_lpips.append(
                            _selected_pair_lpips(
                                pred_images,
                                indices,
                                valid_samples,
                                num_samples_each_step,
                                lpips_fn,
                                lpips_size,
                            )
                        )

                    valid_next_latents.append(torch.gather(
                        all_prev_latents,
                        dim=0,
                        index=indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                            -1,
                            -1,
                            *all_prev_latents.shape[2:],
                        ),
                    )[valid_samples.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        2,
                        -1,
                        *all_prev_latents.shape[2:],
                    )].reshape(2, -1, *all_prev_latents.shape[2:]))

                    valid_current_latents.append(current_latents[valid_samples].unsqueeze(1))
                    valid_timesteps.append(
                        last_timestep.repeat(valid_current_latents[-1].shape[0]).unsqueeze(1)
                    )
                    valid_prompt_embeds.append(log_prompt_embeds[valid_samples].unsqueeze(1))

                    denoise_idx = torch.randint(
                        0,
                        num_samples_each_step,
                        size=(all_prev_latents.shape[1],),
                        device=all_prev_latents.device,
                    )[None, :, None, None, None].expand(
                        -1,
                        -1,
                        *all_prev_latents.shape[2:],
                    )

                    for k, v in pred_dict.items():
                        if k != "prev_timestep":
                            v = v.reshape(num_samples_each_step, -1, *v.shape[1:])
                            pred_dict[k] = torch.gather(v, dim=0, index=denoise_idx)[0]

                    current_latents = torch.gather(
                        all_prev_latents,
                        dim=0,
                        index=denoise_idx,
                    )[0]

                    prev_latents = ddim_step_fetch_x_t_1(
                        self.scheduler,
                        dtype=latents.dtype,
                        num_sample_per_step=num_samples_each_step,
                        timestep=t,
                        **extra_step_kwargs,
                        **pred_dict,
                    )

                latents = prev_latents.flatten(0, 1)
                all_prev_latents = prev_latents
                last_timestep = t

                if i == len(timesteps) - 1:
                    final_images = self.vae.decode(
                        prev_latents.flatten(0, 1).to(self.vae.dtype)
                        / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]
                    preference_timestep = torch.zeros_like(t).repeat(final_images.shape[0])
                    extra_info["timesteps"] = preference_timestep
                    preference_scores_flat = preference_model_fn(final_images, extra_info)
                    preference_score_logs.append(preference_scores_flat)
                    preference_scores = preference_scores_flat.reshape(num_samples_each_step, -1)

                    indices, valid_samples = compare_fn(preference_scores)

                    if collect_pair_lpips:
                        valid_pair_lpips.append(
                            _selected_pair_lpips(
                                final_images,
                                indices,
                                valid_samples,
                                num_samples_each_step,
                                lpips_fn,
                                lpips_size,
                            )
                        )

                    valid_next_latents.append(torch.gather(
                        all_prev_latents,
                        dim=0,
                        index=indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                            -1,
                            -1,
                            *all_prev_latents.shape[2:],
                        ),
                    )[valid_samples.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        2,
                        -1,
                        *all_prev_latents.shape[2:],
                    )].reshape(2, -1, *all_prev_latents.shape[2:]))
                    valid_current_latents.append(current_latents[valid_samples].unsqueeze(1))
                    valid_timesteps.append(
                        last_timestep.repeat(valid_current_latents[-1].shape[0]).unsqueeze(1)
                    )
                    valid_prompt_embeds.append(log_prompt_embeds[valid_samples].unsqueeze(1))

            else:
                pred_dict = ddim_step_fetch_x0(
                    self.scheduler,
                    noise_pred,
                    t,
                    latents,
                )
                latents = ddim_step_fetch_x_t_1(
                    self.scheduler,
                    dtype=latents.dtype,
                    num_sample_per_step=1,
                    timestep=t,
                    **extra_step_kwargs,
                    **pred_dict,
                )
                if i == divert_start_step - 1:
                    current_latents = latents

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps
                and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    valid_timesteps = torch.cat(valid_timesteps, dim=0)
    valid_current_latents = torch.cat(valid_current_latents, dim=0)
    valid_next_latents = torch.cat(valid_next_latents, dim=1).transpose(0, 1).contiguous()
    valid_prompt_embeds = torch.cat(valid_prompt_embeds, dim=0)
    preference_score_logs = torch.cat(preference_score_logs, dim=0)

    if collect_pair_lpips:
        pair_lpips = torch.cat(valid_pair_lpips, dim=0)
        if pair_lpips.shape[0] != valid_timesteps.shape[0]:
            raise RuntimeError(
                f"pair_lpips/sample mismatch: {pair_lpips.shape[0]} vs {valid_timesteps.shape[0]}"
            )
        return (
            valid_timesteps,
            valid_current_latents,
            valid_next_latents,
            valid_prompt_embeds,
            preference_score_logs,
            pair_lpips,
        )

    return (
        valid_timesteps,
        valid_current_latents,
        valid_next_latents,
        valid_prompt_embeds,
        preference_score_logs,
    )
