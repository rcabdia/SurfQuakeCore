from surfquakecore.data_processing.validators.utils import require_keys
from surfquakecore.data_processing.validators.constants import SMOOTHING_METHODS

def validate_smoothing(config):
    require_keys(config, ['method', 'time_window', 'FWHM'])
    if config['method'] not in SMOOTHING_METHODS:
        raise ValueError(f"Unsupported smoothing method: {config['method']}")
    return True