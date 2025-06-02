from surfquakecore.data_processing.validators.utils import require_keys
from surfquakecore.data_processing.validators.constants import INTEGRATE_METHODS

def validate_integrate(config):
    require_keys(config, ['method'])
    if config['method'] not in INTEGRATE_METHODS:
        raise ValueError(f"Unsupported integrate method: {config['method']}")
    return True