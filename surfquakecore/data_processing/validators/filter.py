from surfquakecore.data_processing.validators.utils import require_keys, require_type
from surfquakecore.data_processing.validators.constants import FILTER_METHODS

def validate_filter(config):
    require_keys(config, ['type'])
    type = config['type']

    if type not in FILTER_METHODS:
        raise ValueError(f"Unsupported filter method: '{type}'")

    if 'zerophase' in config:
        require_type(config, 'zerophase', bool)

    if type in ['bandpass', 'bandstop', 'lowpass', 'highpass', 'cheby1', 'cheby2', 'elliptic']:
        require_keys(config, ['fmin', 'fmax'])
        require_type(config, 'fmin', float)
        require_type(config, 'fmax', float)


    return True