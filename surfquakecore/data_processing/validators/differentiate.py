from surfquakecore.data_processing.validators.utils import require_keys

def validate_differentiate(config):
    require_keys(config, ['method']) #'reference', 'phase', or 'absolute'
    require_keys(config, ['t1'])
    require_keys(config, ['t2'])
    return True