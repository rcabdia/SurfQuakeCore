from surfquakecore.data_processing.validators.utils import require_keys, require_type
from surfquakecore.data_processing.validators.constants import FILTER_METHODS

def validate_filter(config):
    require_keys(config, ['method'])
    method = config['method']

    if method not in FILTER_METHODS:
        raise ValueError(f"Unsupported filter method: '{method}'")

    if 'zerophase' in config:
        require_type(config, 'zerophase', bool)

    if method in ['bandpass', 'bandstop', 'remez_fir']:
        require_keys(config, ['freqmin', 'freqmax'])
        require_type(config, 'freqmin', float)
        require_type(config, 'freqmax', float)
    elif method in ['lowpass', 'highpass', 'lowpass_fir', 'lowpass_cheby_2']:
        require_keys(config, ['freq'])
        require_type(config, 'freq', float)

    return True