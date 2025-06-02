from surfquakecore.data_processing.validators.utils import require_keys

def validate_remove_spikes(config):
    require_keys(config, ['window_size', 'sigma'])
    return True