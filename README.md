# PPR: Pairwise Perceptual Reweighting for Text-to-Image Preference Alignment

PPR is a perceptual reweighting method for step-wise preference optimization of text-to-image diffusion models. At each eligible denoising step, the current policy samples multiple candidates from a shared latent. A step-aware preference model selects the highest- and lowest-scoring candidates, and LPIPS measures the perceptual distance between their predicted-clean images.

**PPR-Linear is the primary method.** PPR-Mid and PPR-Hard are weighting-function ablations. PPR uses perceptual distance only as a detached preference-loss weight; it does not optimize LPIPS as a regression target.

## Method

For winner and loser transitions j in {w, l}, the base objective uses the clipped policy/reference likelihood ratio

```math
r_j = \mathrm{clip}\left(\exp(\log p_\theta^j - \log p_{\mathrm{ref}}^j),\ 1-\epsilon,\ 1+\epsilon\right)
```

and

```math
L_{\mathrm{base}} = \mathrm{softplus}\left[-\beta(\log r_w - \log r_l)\right].
```

The winner and loser share the same current latent, timestep, and prompt. During rollout, candidate-specific predicted-clean latents are decoded and scored by the step-aware preference model. The best and worst candidates form a pair, which is retained when their two-way softmax probability gap exceeds the configured threshold.

For each retained pair,

```math
d = \mathrm{sg}\left[\mathrm{LPIPS}(I_w^{x_0}, I_l^{x_0})\right],
\qquad L_{\mathrm{PPR}} = w(d)L_{\mathrm{base}},
```

where sg denotes stop-gradient. The implemented PPR variants are:

- **PPR-Linear (primary):** `w(d) = 1 + lambda * d`
- **PPR-Mid (ablation):** `w(d) = 1 + lambda * exp(-(d-mu)^2 / (2*sigma^2))`
- **PPR-Hard (ablation):** `w(d) = 1 + lambda * exp(-gamma*d)`

Rollout, candidate selection, and pairwise LPIPS computation run under `torch.no_grad()`. The loss helper explicitly detaches the stored distance before constructing the weight.

## Methods and Ablations

| Config value | Paper-facing name | Role |
|---|---|---|
| `spo` | SPO | Unweighted base preference objective |
| `ppr_linear` | PPR-Linear | Primary PPR method |
| `ppr_mid` | PPR-Mid | Mid-range weighting ablation |
| `ppr_hard` | PPR-Hard | Exponentially decaying weighting ablation |
| `direct_lpips` | Direct-LPIPS | Current/reference LPIPS regularizer |
| `snr_lpips` | SNR-LPIPS | SNR-weighted current/reference LPIPS regularizer |

Direct-LPIPS and SNR-LPIPS are separate alternatives rather than PPR variants. They compare current-policy and frozen-reference predicted-clean images from the same `(x_t, t)`. The reference branch is detached, while the current-policy branch remains differentiable through the frozen VAE and LPIPS network.

## Default Training Setup

The source configuration uses Stable Diffusion v1.5 and rank-4 LoRA adapters on UNet attention projections `to_q`, `to_k`, `to_v`, and `to_out.0`. Only LoRA parameters are optimized. The VAE, text encoder, original UNet parameters, frozen reference UNet, step-aware preference model, and LPIPS network remain frozen.

| Setting | Default |
|---|---:|
| Method | PPR-Linear |
| Epochs | 10 |
| Learning rate | 1e-5 |
| Per-device training batch size | 1 |
| Preference scale beta | 10 |
| Ratio-clipping epsilon | 0.1 |
| Pair acceptance threshold | 0.3 |
| PPR lambda | 1.0 |
| PPR-Mid mu / sigma | 0.3 / 0.15 |
| PPR-Hard gamma | 5.0 |
| Reference LPIPS coefficient | 0.5 |
| SNR tau | 1.0 |
| LPIPS backbone / input size | AlexNet / 256 |
| DDIM steps / eta | 20 / 1.0 |
| Classifier-free guidance | 5.0 |
| Candidates per eligible step | 4 |
| Diversion start step | 4 |
| LoRA rank / alpha | 4 / 4 |

Training data is prompt-only JSON. Preferred and dispreferred transitions are generated and selected online; precomputed preference pairs are not required.

## Installation

Python 3.10 and CUDA 12.1 were used for the provided environment.

```bash
conda env create -f environment.yaml
conda activate ppr
```

Alternatively, install the pinned pip dependencies in a compatible Python environment:

```bash
pip install -r requirements.txt
```

The Conda file defines one PyTorch/CUDA stack. Do not install a second PyTorch build over it.

## Required Models

Defaults use the following Hugging Face identifiers:

- `runwayml/stable-diffusion-v1-5`
- `yuvalkirstain/PickScore_v1`
- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`

Download the SPO step-aware preference checkpoint and place it at:

```text
model_ckpts/sd-v1-5_step-aware_preference_model.bin
```

Paths and model identifiers can be overridden without editing source:

| Variable | Purpose |
|---|---|
| `PPR_BASE_MODEL` | Stable Diffusion model ID or local directory |
| `PPR_PREFERENCE_MODEL` | Step-aware preference model base |
| `PPR_CLIP_MODEL` | CLIP processor/tokenizer model |
| `PPR_PREFERENCE_CKPT` | Step-aware preference checkpoint |
| `PPR_PROMPTS` | Prompt JSON |
| `PPR_OUTPUT_DIR` | Training output root |
| `PPR_CACHE_DIR` | Optional Hugging Face cache |

No credentials or machine-specific paths are stored in the source tree.

## Training

### Logging modes

Set `WANDB_MODE` to control Weights & Biases logging:

| Value | Behavior |
|---|---|
| `offline` | Default. Logs to a local `wandb/` directory; no network access. |
| `online` | Streams to Weights & Biases; requires a prior `wandb login`. |
| `disabled` | Turns run logging off entirely. |

### Quick start

Run PPR-Linear with the default configuration:

```bash
bash scripts/train.sh
```

The equivalent explicit command is:

```bash
PYTHONPATH=$(pwd) accelerate launch \
  --config_file accelerate_cfg/multi_gpu_fp16.yaml \
  train_ppr.py \
  --config configs/ppr_sd15.py \
  --config.train.method=ppr_linear \
  --config.run_name=PPR-Linear
```

The bundled Accelerate profile targets four fp16 GPUs. Edit it or set `ACCELERATE_CONFIG` to a profile appropriate for the available hardware.

Override a method and run name through the launcher:

```bash
METHOD=ppr_mid RUN_NAME=PPR-Mid bash scripts/train.sh
```

Run all retained methods sequentially:

```bash
bash scripts/train_all.sh
```

Successful training writes `pytorch_lora_weights.safetensors` under `<PPR_OUTPUT_DIR>/<RUN_NAME>/`.

### Checkpoints and resuming

A checkpoint is written at every epoch boundary as `checkpoint_<EPOCH>/`, containing the LoRA weights, optimizer state, AMP scaler state, per-rank RNG states, and the global step. `checkpoint_N` means epoch `N` finished, so resuming from it continues at epoch `N+1`.

To avoid silently overwriting results, a fresh launch refuses to start in a run directory that already holds checkpoints or final weights. Resume instead:

```bash
RESUME_FROM=auto bash scripts/train.sh
```

`auto` selects the newest checkpoint in the run directory. An explicit checkpoint or run directory also works:

```bash
RESUME_FROM=<PPR_OUTPUT_DIR>/PPR-Linear/checkpoint_3 bash scripts/train.sh
```

Both forms are accepted by the tmux launcher as well, where a resumed run appends to the existing log.

Resume granularity is one completed epoch. If training is interrupted mid-epoch, earlier epochs are preserved and only the interrupted epoch is replayed from its start.

### Long runs in tmux

`scripts/train_linear_tmux.sh` starts a full PPR-Linear run in a detached tmux session so training survives a disconnected terminal. Before the first use, replace the uppercase placeholder paths at the top of the script, or pass them as environment variables:

```bash
PYTHON_BIN=/PATH/TO/CONDA/ENV/BIN/PYTHON \
PPR_CACHE_DIR=/PATH/TO/HUGGINGFACE_CACHE \
PPR_OUTPUT_DIR=/PATH/TO/PPR_OUTPUTS \
LOG_FILE=/PATH/TO/LOGS/PPR_LINEAR_TRAINING.LOG \
bash scripts/train_linear_tmux.sh
```

The launcher validates its inputs, refuses to reuse an existing session name, and prints the attach command, log path, and output directory. Run `bash scripts/train_linear_tmux.sh --help` for the full list of options.

| Variable | Purpose |
|---|---|
| `PYTHON_BIN` | Python executable of the training environment |
| `PPR_CACHE_DIR` | Hugging Face cache holding the required models |
| `PPR_OUTPUT_DIR` | Root directory for runs and checkpoints |
| `LOG_FILE` | Training log file |
| `SESSION_NAME` | tmux session name (default `ppr-linear`) |
| `METHOD` | Training method (default `ppr_linear`) |
| `RUN_NAME` | Run directory name (default `PPR-Linear`) |
| `WANDB_MODE` | `offline`, `online`, or `disabled` |
| `RESUME_FROM` | `none`, `auto`, or a checkpoint path |
| `OFFLINE_MODELS` | `1` forces offline model loading, `0` allows downloads |

Common combinations:

```bash
# Stream metrics to Weights & Biases (requires `wandb login`)
WANDB_MODE=online bash scripts/train_linear_tmux.sh

# Continue an interrupted run from its newest checkpoint
RESUME_FROM=auto bash scripts/train_linear_tmux.sh

# Train an ablation in its own session and run directory
METHOD=ppr_mid RUN_NAME=PPR-Mid SESSION_NAME=ppr-mid bash scripts/train_linear_tmux.sh
```

Session control:

```bash
tmux attach -t ppr-linear          # watch progress
tmux kill-session -t ppr-linear    # remove the session when finished
```

Inside the session, `Ctrl-b d` detaches and leaves training running, while `Ctrl-C` stops training but keeps the session and its shell so the exit status can be inspected and the run resumed. Sessions therefore persist until removed explicitly.

## Inference

Prompt files contain one prompt per line. Single-GPU inference:

```bash
python inference.py \
  --prompt_file <PROMPT_FILE> \
  --output_dir <OUTPUT_DIR> \
  --model <MODEL_ID_OR_PATH> \
  --lora_path <RUN_DIR>/pytorch_lora_weights.safetensors \
  --device cuda:0 \
  --seed 42 \
  --guidance_scale 6.0 \
  --num_inference_steps 50 \
  --lora_rank 4
```

Multi-GPU inference distributes prompts round-robin across selected devices:

```bash
python inference_multi_gpu.py \
  --prompt_file <PROMPT_FILE> \
  --output_dir <OUTPUT_DIR> \
  --model <MODEL_ID_OR_PATH> \
  --lora_path <RUN_DIR>/pytorch_lora_weights.safetensors \
  --gpus auto \
  --seed 42 \
  --guidance_scale 6.0 \
  --num_inference_steps 50 \
  --precision float16 \
  --lora_rank 4
```

Both entry points save images, the exact prompt list, and metadata. The multi-GPU entry additionally records its run configuration and per-worker metadata shards.

## Repository Structure

```text
.
├── README.md
├── train_ppr.py                  # canonical training entry
├── inference.py                  # single-GPU LoRA inference
├── inference_multi_gpu.py        # multi-GPU LoRA inference
├── configs/
│   ├── basic_config.py           # shared portable defaults
│   └── ppr_sd15.py               # canonical SD 1.5 experiment
├── ppr/
│   ├── losses.py                 # SPO base loss and PPR weights
│   ├── custom_diffusers/         # rollout and DDIM helpers
│   ├── preference_models/        # step-aware scoring and pair selection
│   ├── datasets/                 # prompt-only dataset
│   └── utils/
├── prompts/
├── model_ckpts/
├── accelerate_cfg/
├── scripts/
│   ├── train.sh                    # single-method launcher
│   ├── train_all.sh                # sequential launcher for all methods
│   └── train_linear_tmux.sh        # PPR-Linear tmux launcher
├── environment.yaml
└── requirements.txt
```

## Implementation Notes

- PPR reuses predicted-clean images already decoded for step-aware preference scoring; it does not perform a second VAE decode in the training loss.
- The rollout selects the highest- and lowest-scoring candidates for training, then uniformly samples one candidate to continue the rollout trajectory.
- Pairwise LPIPS values are sanitized for non-finite values and clamped below at zero. Set `train.lpips_max_distance` to a positive value to enable upper clipping.
- Each policy/reference likelihood ratio is clipped to `[1-epsilon, 1+epsilon]` before constructing the winner-loser gap.
- The final LoRA file uses Diffusers `save_lora_weights`; the inference entries reconstruct the matching PEFT attention adapters before loading it.

## Packaging

`scripts/package_anonymous.sh` builds a submission archive containing only source, configuration, prompts, and documentation:

```bash
OUTPUT_ARCHIVE=/tmp/ppr-anonymous.tar.gz bash scripts/package_anonymous.sh
```

It excludes version-control data, caches, logs, generated images, model weights, and W&B runs, and it aborts if the tree contains symbolic links or absolute home-directory paths.

## License

Released under the MIT License; see `LICENSE`. Upstream components retain their own licenses, and the base models, preference checkpoint, and prompt data remain subject to the terms of their original providers.

## Acknowledgements

This implementation builds on SPO, D3PO, Diffusion-DPO, Hugging Face Diffusers, PickScore, PEFT, and LPIPS. Please cite the corresponding projects and papers when using their components.
