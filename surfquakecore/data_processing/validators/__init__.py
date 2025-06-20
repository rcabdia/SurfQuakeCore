from .cut import validate_cut
from .rmean import validate_rmean
from .spectrogram import validate_spectrogram
from .spectrum import validate_spectrum
from .stack import validate_stack
from .synch import validate_synch
from .taper import validate_taper
from .normalize import validate_normalize
from .integrate import validate_integrate
from .filter import validate_filter
from .shift import validate_shift
from .remove_response import validate_remove_response
from .differentiate import validate_differentiate
from .wiener_filter import validate_wiener_filter
from .add_noise import validate_add_white_noise
from .whitening import validate_whitening
from .remove_spikes import validate_remove_spikes
from .time_normalization import validate_time_normalization
from .wavelet_denoise import validate_wavelet_denoise
from .resample import validate_resample
from .fill_gaps import validate_fill_gaps
from .smoothing import validate_smoothing
from .rotate import validate_rotate
from .cross_correlate import validate_cross_correlate
from .envelope import validate_envelope


CHECK_DISPATCH = {
    'rmean': validate_rmean,
    'taper': validate_taper,
    'normalize': validate_normalize,
    'integrate': validate_integrate,
    'filter': validate_filter,
    'shift': validate_shift,
    'remove_response': validate_remove_response,
    'differentiate': validate_differentiate,
    'wiener_filter': validate_wiener_filter,
    'add_noise': validate_add_white_noise,
    'whitening': validate_whitening,
    'remove_spikes': validate_remove_spikes,
    'time_normalization': validate_time_normalization,
    'wavelet_denoise': validate_wavelet_denoise,
    'resample': validate_resample,
    'fill_gaps': validate_fill_gaps,
    'smoothing': validate_smoothing,
    'rotate': validate_rotate,
    'cross_correlate': validate_cross_correlate,
    'stack': validate_stack,
    'synch': validate_synch,
    'envelope': validate_envelope,
    'cut': validate_cut,
    'spectrum': validate_spectrum,
    'spectrogram': validate_spectrogram
}

def validate_step(step_type, config):
    if step_type not in CHECK_DISPATCH:
        raise ValueError(f"No validator implemented for step type '{step_type}'")
    return CHECK_DISPATCH[step_type](config)