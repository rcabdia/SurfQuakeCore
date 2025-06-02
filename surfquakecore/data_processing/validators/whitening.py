from surfquakecore.data_processing.validators.utils import optional_type

def validate_whitening(config):
    if 'taper_edge' in config:
        optional_type(config, 'taper_edge', bool)
    return True