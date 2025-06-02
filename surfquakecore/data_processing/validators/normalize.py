from surfquakecore.data_processing.validators.utils import require_keys, optional_type

def validate_normalize(config):
    require_keys(config, ['norm'])
    if config['norm'] is not None:
        optional_type(config, 'norm', (float, bool))  # <- pass a tuple of accepted types
    return True