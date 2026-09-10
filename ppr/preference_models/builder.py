from functools import partial
from mmengine import Registry

COMPARE_FUNCS = Registry('compare_funcs')
PREFERENCE_MODEL_FUNC_BUILDERS = Registry('preference_model_func_builders')

def get_compare_func(compare_func_cfg):
    function_type = compare_func_cfg.pop('type')
    compare_func = COMPARE_FUNCS.get(function_type)
    if compare_func_cfg:
        compare_func = partial(compare_func, **compare_func_cfg)
    return compare_func

def get_preference_model_func(cfg, device):
    builder_type = cfg.pop('type')
    cfg.device = device
    preference_model_func_builder = PREFERENCE_MODEL_FUNC_BUILDERS.get(builder_type)
    preference_model_func = preference_model_func_builder(cfg)
    return preference_model_func
