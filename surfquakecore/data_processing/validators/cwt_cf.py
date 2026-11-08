from surfquakecore.data_processing.validators.utils import require_keys, require_type

def validate_cwt_cf(config: dict) -> bool:

    require_keys(config, ['cycles', 'fmin', 'fmax', "wavelet_type", "voices_per_octave"])

    require_type(config, 'cycles', (int, float))
    require_type(config, 'fmin', (int, float))
    require_type(config, 'fmax', (int, float))
    require_type(config, 'voices_per_octave', int)
    require_type(config, 'wavelet_type', str)

    if config['wavelet_type'] not in ["cm", "mh", "pa"]:

        raise ValueError("'wavelet_type' must be string [cm|mh|pa]")

    return True