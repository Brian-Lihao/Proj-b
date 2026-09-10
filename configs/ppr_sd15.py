from configs.basic_config import basic_config


def get_config():
    return experiment_config()


def experiment_config():
    config = basic_config()
    config.sample.num_sample_each_step = 4
    config.train.method = "ppr_linear"
    config.run_name = "PPR-Linear"
    return config
