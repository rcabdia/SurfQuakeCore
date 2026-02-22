from surfquakecore.data_processing.validators.utils import optional_type, require_keys


def validate_whitening(config):
    require_keys(config, ['fmin', 'fmax', 'freq_width'])
    if 'taper_edge' in config:
        optional_type(config, 'taper_edge', bool)
    return True