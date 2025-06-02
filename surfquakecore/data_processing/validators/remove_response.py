from surfquakecore.data_processing.validators.utils import require_keys

def validate_remove_response(config):
    require_keys(config, ['water_level', 'units', 'pre_filt'])
    return True