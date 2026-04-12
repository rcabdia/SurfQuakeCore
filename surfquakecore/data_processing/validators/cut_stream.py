from datetime import datetime
from surfquakecore.data_processing.validators.constants import CUT_TYPES
from surfquakecore.data_processing.validators.utils import require_keys, require_type

def validate_cut_stream(config):
    require_keys(config, ['method'])

    method = config['method']

    if method not in CUT_TYPES:
        raise ValueError(f"Unsupported cut method: {config['method']}")

    if config['method'] == "absolute":
        require_type(config, 'start', datetime)
        require_type(config, 'end', datetime)

    elif config['method'] == "phase":
        require_type(config, 'phase_name', str)
        require_type(config, 't_before', float)
        require_type(config, 't_after', float)

    elif config['method'] == "reference":
        require_type(config, 't_before', float)
        require_type(config, 't_after', float)

    else:
        print("No valid Config method ", config['method'], "options are: absolute, phase or reference")
        return False

    return True