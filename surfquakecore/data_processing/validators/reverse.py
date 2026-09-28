from surfquakecore.data_processing.validators.utils import require_keys, require_type

def validate_reverse(config):
    require_keys(config, ['flip'])
    require_type(config, 'flip', bool)
    return True