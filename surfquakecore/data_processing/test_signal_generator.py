
import numpy as np
from obspy import Trace, UTCDateTime
from scipy.signal import chirp


class TestTraceGenerator:
    """
    Generate synthetic ObsPy traces for testing and signal-processing
    applications.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz.
    duration : float
        Signal duration in seconds.
    starttime : str, datetime or UTCDateTime
        Start time of the generated trace.
    network : str, optional
        Network code.
    station : str, optional
        Station code.
    location : str, optional
        Location code.
    channel : str, optional
        Channel code.
    """

    def __init__(
        self,
        sampling_rate=100.0,
        duration=60.0,
        starttime=None,
        network="XX",
        station="TEST",
        location="",
        channel="BHZ",
    ):
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be > 0")

        if duration <= 0:
            raise ValueError("duration must be > 0")

        self.sampling_rate = float(sampling_rate)
        self.duration = float(duration)

        if starttime is None:
            starttime = UTCDateTime(0)

        self.starttime = UTCDateTime(starttime)

        self.network = network
        self.station = station
        self.location = location
        self.channel = channel

        self.npts = int(round(self.duration * self.sampling_rate))
        self.dt = 1.0 / self.sampling_rate
        self.time = np.arange(self.npts) * self.dt

    def generate(self, signal_type, **kwargs):
        """
        Generate a synthetic ObsPy Trace.

        Available signal types
        ----------------------
        impulse
        impulse_train
        step
        boxcar
        triangle
        sine
        line
        quadratic
        cubic
        random
        ricker
        chirp
        damped_sine
        """

        signal_type = signal_type.lower()

        generators = {
            "impulse": self._impulse,
            "impulse_train": self._impulse_train,
            "step": self._step,
            "boxcar": self._boxcar,
            "triangle": self._triangle,
            "sine": self._sine,
            "line": self._line,
            "quadratic": self._quadratic,
            "cubic": self._cubic,
            "random": self._random,
            "ricker": self._ricker,
            "chirp": self._chirp,
            "damped_sine": self._damped_sine,
        }

        if signal_type not in generators:
            available = ", ".join(generators.keys())
            raise ValueError(
                f"Unknown signal type '{signal_type}'. "
                f"Available types: {available}"
            )

        data = generators[signal_type](**kwargs)
        return self._build_trace(data)

    # ------------------------------------------------------------------
    # Basic signals
    # ------------------------------------------------------------------

    def _impulse(self, amplitude=1.0, sample=None):
        data = np.zeros(self.npts, dtype=float)

        if sample is None:
            sample = self.npts // 2

        if not 0 <= sample < self.npts:
            raise ValueError(
                f"Impulse sample must be between 0 and {self.npts - 1}"
            )

        data[sample] = amplitude
        return data

    def _impulse_train(self, samples, amplitudes=1.0):
        data = np.zeros(self.npts, dtype=float)

        samples = np.atleast_1d(samples).astype(int)

        if np.isscalar(amplitudes):
            amplitudes = np.full(len(samples), amplitudes, dtype=float)
        else:
            amplitudes = np.asarray(amplitudes, dtype=float)

        if len(amplitudes) != len(samples):
            raise ValueError(
                "Number of amplitudes must match number of samples."
            )

        for sample, amplitude in zip(samples, amplitudes):
            if not 0 <= sample < self.npts:
                raise ValueError(
                    f"Impulse sample {sample} outside trace."
                )

            data[sample] += amplitude

        return data

    def _step(self, amplitude=1.0):
        data = np.zeros(self.npts, dtype=float)
        middle = self.npts // 2
        data[middle:] = amplitude
        return data

    def _boxcar(self, amplitude=1.0):
        data = np.zeros(self.npts, dtype=float)

        i1 = self.npts // 3
        i2 = 2 * self.npts // 3

        data[i1:i2] = amplitude
        return data

    def _triangle(self, amplitude=1.0):
        data = np.zeros(self.npts, dtype=float)

        q1 = self.npts // 4
        q2 = self.npts // 2
        q3 = 3 * self.npts // 4

        data[q1:q2] = np.linspace(
            0.0,
            amplitude,
            q2 - q1,
            endpoint=False,
        )

        data[q2:q3] = np.linspace(
            amplitude,
            0.0,
            q3 - q2,
            endpoint=False,
        )

        return data

    def _sine(self, frequency=1.0, phase=0.0, amplitude=1.0):
        self._check_frequency(frequency)

        phase_rad = np.deg2rad(phase)

        return amplitude * np.sin(
            2.0 * np.pi * frequency * self.time + phase_rad
        )

    def _line(self, slope=1.0, intercept=0.0):
        return slope * self.time + intercept

    def _quadratic(self, a=1.0, b=0.0, c=0.0):
        return (
            a * self.time**2
            + b * self.time
            + c
        )

    def _cubic(self, a=1.0, b=0.0, c=0.0, d=0.0):
        return (
            a * self.time**3
            + b * self.time**2
            + c * self.time
            + d
        )

    def _random(
        self,
        mean=0.0,
        std=1.0,
        seed=None,
    ):
        if std < 0:
            raise ValueError("std must be >= 0")

        rng = np.random.default_rng(seed)

        return rng.normal(
            loc=mean,
            scale=std,
            size=self.npts,
        )

    # ------------------------------------------------------------------
    # Seismic / signal-processing functions
    # ------------------------------------------------------------------

    def _ricker(
        self,
        frequency=5.0,
        time=None,
        amplitude=1.0,
    ):
        """
        Generate a Ricker wavelet.

        Parameters
        ----------
        frequency : float
            Central frequency in Hz.
        time : float, optional
            Time of the wavelet maximum in seconds from the beginning
            of the trace. Defaults to the centre of the trace.
        amplitude : float
            Peak scaling factor.
        """

        self._check_frequency(frequency)

        if time is None:
            time = self.duration / 2.0

        if not 0 <= time <= self.duration:
            raise ValueError(
                "time must lie inside the trace duration."
            )

        tau = self.time - time

        x = np.pi * frequency * tau

        data = (
            amplitude
            * (1.0 - 2.0 * x**2)
            * np.exp(-x**2)
        )

        return data

    def _chirp(
        self,
        f0=1.0,
        f1=20.0,
        amplitude=1.0,
        method="linear",
        phase=0.0,
    ):
        """
        Generate a frequency-swept sinusoidal signal.

        Parameters
        ----------
        f0 : float
            Frequency at the beginning of the signal in Hz.
        f1 : float
            Frequency at the end of the signal in Hz.
        amplitude : float
            Signal amplitude.
        method : str
            Frequency sweep type accepted by scipy.signal.chirp:
            'linear', 'quadratic', 'logarithmic', or 'hyperbolic'.
        phase : float
            Initial phase in degrees.
        """

        self._check_frequency(f0)
        self._check_frequency(f1)

        allowed_methods = {
            "linear",
            "quadratic",
            "logarithmic",
            "hyperbolic",
        }

        if method not in allowed_methods:
            raise ValueError(
                f"Unknown chirp method '{method}'. "
                f"Choose from {sorted(allowed_methods)}."
            )

        data = chirp(
            self.time,
            f0=f0,
            f1=f1,
            t1=self.duration,
            method=method,
            phi=phase,
        )

        return amplitude * data

    def _damped_sine(
        self,
        frequency=8.0,
        decay=0.5,
        amplitude=1.0,
        phase=0.0,
        time=0.0,
    ):
        """
        Generate an exponentially damped sinusoid.

        x(t) = A exp(-decay * tau) sin(2*pi*f*tau + phase)

        where tau = t - time and the signal is zero before ``time``.

        Parameters
        ----------
        frequency : float
            Oscillation frequency in Hz.
        decay : float
            Exponential decay coefficient in 1/s.
        amplitude : float
            Initial amplitude.
        phase : float
            Initial phase in degrees.
        time : float
            Start time of the damped oscillation in seconds from the
            beginning of the trace.
        """

        self._check_frequency(frequency)

        if decay < 0:
            raise ValueError("decay must be >= 0")

        if not 0 <= time <= self.duration:
            raise ValueError(
                "time must lie inside the trace duration."
            )

        data = np.zeros(self.npts, dtype=float)

        mask = self.time >= time
        tau = self.time[mask] - time

        phase_rad = np.deg2rad(phase)

        data[mask] = (
            amplitude
            * np.exp(-decay * tau)
            * np.sin(
                2.0 * np.pi * frequency * tau
                + phase_rad
            )
        )

        return data

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_frequency(self, frequency):
        if frequency < 0:
            raise ValueError("frequency must be >= 0")

        nyquist = self.sampling_rate / 2.0

        if frequency > nyquist:
            raise ValueError(
                f"Frequency {frequency} Hz exceeds "
                f"Nyquist frequency {nyquist} Hz."
            )

    # ------------------------------------------------------------------
    # ObsPy Trace
    # ------------------------------------------------------------------

    def _build_trace(self, data):
        data = np.asarray(data, dtype=np.float64)

        header = {
            "network": self.network,
            "station": self.station,
            "location": self.location,
            "channel": self.channel,
            "sampling_rate": self.sampling_rate,
            "starttime": self.starttime,
        }

        return Trace(
            data=data,
            header=header,
        )
