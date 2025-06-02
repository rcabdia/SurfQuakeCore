from surfquakecore.data_processing.validators.utils import require_keys
from surfquakecore.data_processing.validators.constants import FILL_GAP_METHODS

def validate_fill_gaps(config):
    require_keys(config, ['method'])
    if config['method'] not in FILL_GAP_METHODS:
        raise ValueError(f"Unsupported fill gaps method: {config['method']}")
    return True