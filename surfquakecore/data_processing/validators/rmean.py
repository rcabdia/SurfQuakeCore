from surfquakecore.data_processing.validators.utils import require_keys, require_type
from surfquakecore.data_processing.validators.constants import RMEAN_METHODS

def validate_rmean(config):
    require_keys(config, ['method'])
    method = config['method']

    if method not in RMEAN_METHODS:
        raise ValueError(f"Unsupported rmean method: '{method}'")

    if method == 'polynomial':
        require_keys(config, ['order'])
        require_type(config, 'order', int)
    elif method == 'spline':
        require_keys(config, ['order', 'dspline'])
        require_type(config, 'order', int)
        require_type(config, 'dspline', int)
        if not (1 <= config['order'] <= 5):
            raise ValueError("Spline method: 'order' must be between 1 and 5")

    return True