import numpy as np
from functools import lru_cache

from .base import Domain


class UniformTimeDomain(Domain):
    """Defines the physical time domain on which the data of interest live.

    The time bins are assumed to be uniform between [0, duration]
    with spacing 1 / sampling_rate.

    TODO: Add BaseTimeDomain class and make this inherit from it.
    """

    def __init__(self, time_duration: float, sampling_rate: float):
        self._time_duration = time_duration
        self._sampling_rate = sampling_rate

    def update(self, new_settings: dict):
        """
        Update the domain with new settings.

        Parameters
        ----------
        new_settings : dict
            Settings dictionary.
        """
        new_settings = new_settings.copy()
        if "type" in new_settings and new_settings.pop("type") not in [
            "UniformTimeDomain",
            "TimeDomain",
            "TD",
        ]:
            raise ValueError("Cannot update domain to type other than UniformTimeDomain.")

        for k, v in new_settings.items():
            if k == "time_duration":
                self._time_duration = v
            elif k == "sampling_rate":
                self._sampling_rate = v
            else:
                raise KeyError(f"Invalid key for domain update: {k}.")

        # Clear cached values
        self.__len__.cache_clear()
        self.__call__.cache_clear()

    @lru_cache()
    def __len__(self):
        """Number of time bins given duration and sampling rate"""
        return int(self._time_duration * self._sampling_rate)

    @lru_cache()
    def __call__(self) -> np.ndarray:
        """Array of uniform times at which data is sampled"""
        num_bins = self.__len__()
        return np.linspace(
            0.0, self._time_duration, num=num_bins, endpoint=False, dtype=np.float32
        )

    @property
    def time_array(self) -> np.ndarray:
        """Array of uniform times at which data is sampled"""
        n_bins = int(self._time_duration * self._sampling_rate)
        return np.linspace(
            0.0, self._time_duration, num=n_bins, endpoint=False, dtype=np.float32
        )

    @property
    def delta_t(self) -> float:
        """The size of the time bins"""
        return 1.0 / self._sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t: float):
        self._sampling_rate = 1.0 / delta_t

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        return 1.0 / np.sqrt(2.0 * self.delta_t)

    def time_translate_data(self, data, dt) -> np.ndarray:
        """
        Time translate data by dt seconds using circular shift.

        Parameters
        ----------
        data : np.ndarray
            Time-domain data array.
        dt : float
            Time shift in seconds. Positive dt shifts data to later times.

        Returns
        -------
        np.ndarray
            Time-shifted data.
        """
        n_shift = int(round(dt * self._sampling_rate))
        return np.roll(data, n_shift, axis=-1)

    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        return self._sampling_rate / 2.0

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return self._time_duration

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def min_idx(self) -> int:
        return 0

    @property
    def max_idx(self) -> int:
        return round(self._time_duration * self._sampling_rate)

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        return {
            "type": "UniformTimeDomain",
            "time_duration": self._time_duration,
            "sampling_rate": self._sampling_rate,
        }