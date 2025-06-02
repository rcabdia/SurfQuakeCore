from surfquakecore.data_processing.validators.utils import require_keys
from surfquakecore.data_processing.validators.constants import WAVELET_METHODS

def validate_wavelet_denoise(config):
    require_keys(config, ['dwt', 'threshold'])
    if config['dwt'] not in WAVELET_METHODS:
        raise ValueError(f"Unsupported wavelet type: {config['dwt']}")
    return True