from surfquakecore.data_processing.validators.utils import require_keys, require_type


def validate_deconv(config):
    require_keys(config, ['reference_idx'])
    require_type(config, "reference_idx", int)

    if "water_level" in config.keys():
        require_type(config, "water_level", float)

    if "normalize" in config.keys():
        require_type(config, "normalize", bool)

    if "trim" in config.keys():
        require_type(config, "trim", bool)

    if "gaussian_filter" in config.keys():
        require_type(config, "gaussian_filter", float)