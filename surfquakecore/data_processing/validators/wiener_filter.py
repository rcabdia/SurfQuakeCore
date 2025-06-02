from surfquakecore.data_processing.validators.utils import require_keys

def validate_wiener_filter(config):
    require_keys(config, ['time_window', 'noise_power'])
    return True