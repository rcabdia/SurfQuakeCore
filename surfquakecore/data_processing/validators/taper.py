from surfquakecore.data_processing.validators.utils import require_keys, require_type
from surfquakecore.data_processing.validators.constants import TAPER_METHODS, TAPER_SIDE

def validate_taper(config):
    require_keys(config, ['method', 'max_percentage'])

    method = config['method']
    if method not in TAPER_METHODS:
        raise ValueError(f"Unsupported taper method: '{method}'")

    require_type(config, 'max_percentage', float)
    if not (0.0 <= config['max_percentage'] <= 0.5):
        raise ValueError("max_percentage must be between 0.0 and 0.5")

    if 'max_length' in config:
        require_type(config, 'max_length', float)

    if 'side' in config and config['side'] not in TAPER_SIDE:
        raise ValueError(f"Invalid side value: {config['side']}. Must be one of {TAPER_SIDE}")

    return True