from surfquakecore.data_processing.validators.utils import require_keys

def validate_differentiate(config):
    require_keys(config, ['method'])
    return True