from surfquakecore.data_processing.validators.utils import require_keys

def validate_shift(config):
    if not isinstance(config['time_shifts'], list):
        raise ValueError("Shift config must be a list of float numbers representing "
                         "the trace start time shifts")

    require_keys(config, ['name', 'time_shifts'])
    return True