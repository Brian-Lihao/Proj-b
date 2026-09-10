import os

import ml_collections


def get_config():
    return basic_config()


def basic_config():
    config = ml_collections.ConfigDict()

    # General
    config.seed = 42
    config.num_checkpoint_limit = None
    config.allow_tf32 = True
    config.use_xformers = False
    config.use_checkpointing = False

    # Model
    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = os.environ.get(
        "PPR_BASE_MODEL", "runwayml/stable-diffusion-v1-5"
    )
    config.use_lora = True
    config.lora_rank = 4

    # Step-aware preference model
    config.preference_model_func_cfg = dict(
        type="step_aware_preference_model_func",
        model_pretrained_model_name_or_path=os.environ.get(
            "PPR_PREFERENCE_MODEL", "yuvalkirstain/PickScore_v1"
        ),
        processor_pretrained_model_name_or_path=os.environ.get(
            "PPR_CLIP_MODEL", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        ),
        ckpt_path=os.environ.get(
            "PPR_PREFERENCE_CKPT",
            "model_ckpts/sd-v1-5_step-aware_preference_model.bin",
        ),
    )
    config.compare_func_cfg = dict(
        type="preference_score_compare",
        threshold=0.3,
    )

    # Prompt-only training dataset
    config.dataset_cfg = dict(
        type="PromptDataset",
        meta_json_path=os.environ.get(
            "PPR_PROMPTS", "prompts/4k_training_prompts.json"
        ),
        pretrained_tokenizer_path=os.environ.get(
            "PPR_CLIP_MODEL", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        ),
    )
    config.dataloader_num_workers = 16
    config.dataloader_shuffle = True
    config.dataloader_pin_memory = True
    config.dataloader_drop_last = False

    # Training
    config.num_epochs = 10
    config.resume_from = ""

    config.sample = ml_collections.ConfigDict()
    config.sample.num_steps = 20
    config.sample.eta = 1.0
    config.sample.guidance_scale = 5.0
    config.sample.sample_batch_size = 4
    config.sample.num_sample_each_step = 4

    config.train = ml_collections.ConfigDict()
    config.train.method = "ppr_linear"
    config.train.train_batch_size = 1
    config.train.use_8bit_adam = False
    config.train.learning_rate = 1e-5
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.gradient_accumulation_steps = 1
    config.train.max_grad_norm = 1.0
    config.train.cfg = True
    config.train.divert_start_step = 4
    config.train.beta = 10.0
    config.train.eps = 0.1

    # Perceptual objectives
    config.train.lpips_size = 256
    config.train.lpips_max_distance = 0.0
    config.train.ppr_lambda = 1.0
    config.train.ppr_mid_mu = 0.3
    config.train.ppr_mid_sigma = 0.15
    config.train.ppr_hard_gamma = 5.0
    config.train.reference_lpips_lambda = 0.5
    config.train.snr_tau = 1.0

    # Validation and logging
    config.validation_prompts = ["A beautiful lake"]
    config.num_validation_images = 1
    config.eval_interval = 1
    config.run_name = "PPR-Linear"
    config.wandb_project_name = "PPR"
    config.wandb_entity_name = None
    config.logdir = os.environ.get("PPR_OUTPUT_DIR", "outputs")
    config.save_interval = 1

    return config
