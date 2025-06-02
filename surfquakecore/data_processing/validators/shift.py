from surfquakecore.data_processing.validators.utils import require_keys

def validate_shift(config):
    if not isinstance(config, list):
        raise ValueError("Shift config must be a list of dictionaries")
    for item in config:
        require_keys(item, ['name', 'time'])
    return True