from surfquakecore.data_processing.validators.utils import require_keys, require_type

def validate_resample(config):
    require_keys(config, ['sampling_rate', 'pre_filter'])
    require_type(config, 'pre_filter', bool)
    return True