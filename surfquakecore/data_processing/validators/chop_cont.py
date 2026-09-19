from surfquakecore.data_processing.validators.utils import require_keys, require_type

def validate_chop_cont(config):
    """
    Validate the CHOP processing config.

    Expected keys:
      - chunk_length  (float)  : Output window length in seconds
      - min_length    (float)  : Minimum amount of REAL recorded data required, in seconds.
      - output_dir    (str)    : target sampling rate; must be positive

    # max_interpolation_gap: float Maximum gap that we allow to interpolate, in seconds.
    # DEFAULT max_interpolation_gap = 2, THIS IS DONE INTERNALLY
    """

    require_keys(config, ['chunk_length', 'min_length', 'output_dir'])

    require_type(config, 'chunk_length', float)
    require_type(config, 'min_length', float)
    require_type(config, 'output_dir', float)

    return True