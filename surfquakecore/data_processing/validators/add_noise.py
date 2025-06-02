from surfquakecore.data_processing.validators.utils import require_keys

def validate_add_white_noise(config):
    require_keys(config, ['noise_type'])
    require_keys(config, ['SNR_dB'])
    return True