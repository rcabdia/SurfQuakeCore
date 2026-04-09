from datetime import datetime
from surfquakecore.data_processing.validators.constants import CUT_TYPES
from surfquakecore.data_processing.validators.utils import require_keys, require_type

def validate_cut_stream(config):
    require_keys(config, ['method'])

    method = config['method']
    if method not in CUT_TYPES:
        raise ValueError(f"Unsupported cut method: {config['method']}")

    if config['method'] == "absolute":
        # absolute
        require_type(config, 'start', datetime)
        require_type(config, 'end', datetime)
    else:
        # phase or reference
        require_type(config, 't_before', str)
        require_type(config, 't_end', str)

    return True