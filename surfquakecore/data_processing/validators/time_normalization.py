from surfquakecore.data_processing.validators.utils import require_keys
from surfquakecore.data_processing.validators.constants import TIME_METHODS

def validate_time_normalization(config):
    require_keys(config, ['method'])
    if config['method'] not in TIME_METHODS:
        raise ValueError(f"Unsupported time normalization method: {config['method']}")
    return True