from surfquakecore.data_processing.validators.utils import require_keys, optional_type

def validate_normalize(config):
    require_keys(config, ['norm'])
    if config['norm'] is not None:
        optional_type(config, 'norm', (float, bool))  # <- pass a tuple of accepted types

    optional_type(config, 'global_max', bool)
    if config.get('global_max', False) and config.get('norm') not in (True, None):
        raise ValueError("normalize: global_max: True divides by the stream's absolute maximum; "
                         "use it with norm: True (a fixed norm value is already the same for all traces)")

    return True