# Read in libraries
import numpy as np
from matplotlib import pyplot as plt

def poisson_refractory_method1(rate, T, dt=1e-4, tau_ref=0.05, rng=None):
    """
    Generate Poisson spike train of length T, rate rate, and refractory period.

    Parameters
    ----------
    rate : float
        Expected firing rate of the spike train
    T : float
        Length of time for spike train (seconds)
    dt : float
        (optional) Small dt to determine binning of spike train
    tau_ref : float
        (optional) Refractory period, shortest time between spikes
    rng : numpy.random.Generator, optional
        Defaults to None, otherwise sets random seed

    Returns
    --------
    np.ndarray
        array of spike times following poisson distribution
    """
    rng = np.random.default_rng(rng)  # set random seed
    n_bins = int(T / dt)  # determine total bins
    prob = rate * dt  # probability of a spike
    spikes = rng.random(n_bins) < prob  # check if spike occurs in time bin
    times = (np.arange(n_bins) + 0.5) * dt  # create time array corresponding to bins

    final_spike_train = []
    last_spike = -1

    # Iterate through all spikes only add those at least tau_ref apart
    # to final spike train
    for s, t in zip(spikes, times):
        if s and (t - last_spike >= tau_ref):
            final_spike_train.append(t)
            last_spike = t
    return np.array(final_spike_train)

def poisson_refractory_method2(rate, T, tau_ref=0.05, rng=None):
    """
    Generate Poisson spike train of length T, rate rate, and refractory period.

    Parameters
    ----------
    rate : float
        Expected firing rate of the spike train
    T : float
        Length of time for spike train (seconds)
    tau_ref : float
        (optional) Refractory period, shortest time between spikes
    rng : numpy.random.Generator, optional
        Defaults to None, otherwise sets random seed

    Returns
    --------
    np.ndarray
        array of spike times following poisson distribution
    """
    # set random seed
    rng = np.random.default_rng(rng)

    # Maximum possible firing rate and guard against tau_ref = 0
    r_ceil = 1 / max(tau_ref, 1e-12)

    # Enforce feasbility, rate must be < 1/tau_ref
    if rate >= 1 / max(tau_ref, 1e-12):
        raise ValueError(
            f"Requested rate {rate:.3f} Hz >= 1/tau_ref = {r_ceil:.3f} Hz. "
            "Decrease rate or tau_ref."
        )

    # Mean of exponential distribution to draw ISIs from
    r_max = max(1 / rate - tau_ref, 1e-12)

    # Empty array to hold spike times
    spikes = []

    # Evolve time by ISIs to identify spikes
    t = 0
    while t < T:
        if t == 0:  # first spike does not need refractory
            t += rng.exponential(r_max)
        else:  # Subsequent spikes have refractory period
            t += rng.exponential(r_max) + tau_ref

        # keep spike as long smaller than cutoff T
        if t < T:
            spikes.append(t)
    return np.asarray(spikes)

def poisson_burst_method1(
    rate,
    burst_rate,
    T,
    dt=1e-3,
    tau_ref=0.05,
    tau_burst=0.1,
    prob_burst=1,
    prob_end=30,
    rng=None,
):
    """
    Generate Poisson spike train of length T, rate rate, and refractory period.
    There is a chance of bursting at rate burst_rate.

    Parameters
    ----------
    rate : float
        Baseline firing rate of the spike train
    burst_rate : float
        Elevated firing rate used during a burst
    T : float
        Length of time for spike train (seconds)
    dt : float
        (optional) Small dt to determine binning of spike train
    tau_ref : float
        (optional) Refractory period, shortest time between spikes
    tau_burst : float
        (optional) Minimum time between consecutive bursts
    prob_burst : float
        (optional) Per-second hazard to start a burst (converted to per-bin as prob_burst*dt)
    prob_end : float
        (optional) Per-second hazard to end a burst (converted to per-bin as prob_end*dt)
    rng : numpy.random.Generator, optional
        Defaults to None, otherwise sets random seed

    Returns
    --------
    np.ndarray
        array of spike times following poisson distribution
    """
    # set random seed
    rng = np.random.default_rng(rng)

    # Empty array to hold spikes
    spikes = []
    t = 0.0

    # boolean to track if we are in a burst
    in_burst = False

    # Checks for most recent spike time, and most recent burst spike
    last_spike = -1
    last_burst = -1

    # Evolve time by dt to find spike times
    while t < T:

        # If not in burst, see if we enter burst, assuming time since the
        # last spike of the most recent burst is larger than tau_burst
        if not in_burst and t - tau_burst >= last_burst:
            if rng.random() < prob_burst * dt:
                in_burst = True

        # If in burst, see if we need to exit burst
        elif in_burst and rng.random() < prob_end * dt:
            in_burst = False

        # Determine rate for see if spike occurred, different if in burst or not
        rate_check = burst_rate if in_burst else rate

        # Determine if we spiked in the bin, assuming tau_ref has elapsed
        if rng.random() < rate_check * dt and t - tau_ref >= last_spike:
            spikes.append(t)
            last_spike = t

            # Update last spike in burst
            if in_burst:
                last_burst = t

        # Advance time by one step
        t += dt
    return np.array(spikes)

def poisson_burst_method2(
    rate,
    burst_rate,
    T,
    tau_ref=0.05,
    tau_burst=0.1,
    prob_burst=0.2,
    prob_end=0.5,
    rng=None,
):
    """
    Generate Poisson spike train of length T, rate rate, and refractory period.
    There is a chance of bursting at rate burst_rate

    Parameters
    ----------
    rate : float
        Baseline firing rate of the spike train
    burst_rate : float
        Elevated firing rate used during a burst
    T : float
        Length of time for spike train (seconds)
    dt : float
        (optional) Small dt to determine binning of spike train
    tau_ref : float
        (optional) Refractory period, shortest time between spikes
    tau_burst : float
        (optional) Minimum time between consecutive bursts
    prob_burst : float
        (optional) Probability to enter a burst.
    prob_end : float
        (optional) Probability to end the burst.
    rng : numpy.random.Generator, optional
        Defaults to None, otherwise sets random seed

    Returns
    --------
    np.ndarray
        array of spike times following poisson distribution
    """

    # Set random seed
    rng = np.random.default_rng(rng)

    # Maximum possible firing rate and guard against tau_ref = 0
    r_ceil = 1 / max(tau_ref, 1e-12)

    # Enforce feasbility, rate must be < 1/tau_ref
    if rate >= 1 / max(tau_ref, 1e-12):
        raise ValueError(
            f"Requested rate {rate:.3f} Hz >= 1/tau_ref = {r_ceil:.3f} Hz. Decrease rate or tau_ref."
        )
    if burst_rate > r_ceil:
        raise ValueError(
            f"Requested burst rate {burst_rate:.3f} Hz > 1/tau_ref = {r_ceil:.3f} Hz. Decrease burst rate or tau_ref"
        )

    # Empty array to hold spike times
    spikes = []
    t = 0

    # bool for in burst
    in_burst = False

    # time of last spike in burst
    last_burst = -1

    # Loop through time and draw ISIs
    while t < T:

        # Enter burst if prob_burst met and time since last_burst is greater than tau_burst
        if rng.random() < prob_burst and (t - last_burst) >= tau_burst:
            in_burst = True

            # In the burst, update spike times
            while in_burst and t < T:

                # Draw ISIs from burst_rate distribution and include tau_ref
                t += rng.exponential(1 / burst_rate - tau_ref) + tau_ref

                # Only add spike if smaller than cutoff time
                if t < T:
                    spikes.append(t)

                # Check if probability met to end the burst
                # Record exiting burst and last spike in burst
                if rng.random() < prob_end:
                    in_burst = False
                    last_burst = t
        else:
            # Draw ISI from background rate if not in burst
            t += rng.exponential(1 / rate - tau_ref) + tau_ref

            # Only add spike if smaller than cutoff time
            if t < T:
                spikes.append(t)
    return np.array(spikes)

def psth(spikes, bin_size, T0, T1):
    """
    Computes peri-stimulus time histogram from multiple spike trains.

    Parameters
    ----------
    spikes : list of np.ndarray
        List where each element is 1d array of spike times between T0 and T1
    bin_size : float
        Width of each bin for the histogram
    T0 : float
        Start time of the analysis window
    T1 : float
        End time of the analysis window.

    Returns
    -------
    bin_centers : np.ndarray
        1d array of time points representing centers of bins for histogram
    counts : np.ndarray
        1d array of average firing rates across all trials aligned at bin_centers
    """

    # Create bin edges from T0 to T1 by bin_size
    bin_edges = np.arange(T0, T1 + bin_size, bin_size)

    # Empty array to hold all spike counts
    counts = np.zeros(len(bin_edges) - 1)

    # Loop through spike trains in spikes list
    # Determine histogram for each spike train and update counts
    for spks in spikes:
        hist_counts, _ = np.histogram(spks, bins=bin_edges)
        counts += hist_counts

    # Divide counts by bin_size and number of trials to return firing rate
    counts = counts / (bin_size * len(spikes))

    # Determine bin_centers for better plotting
    bin_centers = bin_edges[:-1] + bin_size / 2

    return bin_centers, counts

def autocorrelogram(spikes, T, max_lag=0.02, bin_size=0.005):
    """
    Computer the spike trian autocorrelogram (ACG). This shows probability of
    finding a spike at a time lag relative of another spike. Elucidates
    temporal properties of spike train.

    Parameters
    ----------
    spikes : array_like
        1D array of sorted spike times
    T : float
        Total duration of spike train
    max_lag : float
        (optional) Maximum time lag to consider
    bin_size : float
        (optioal) Bin width for histogram of spike differences

    Returns
    -------
    bin_centers : np.ndarray
        1d array of time points representing centers of bins for histogram
    acorr : np.ndarray
        Normalized autocorrelogram values, units are spike density.
    """

    # Empty array to hold spike differences
    diffs = []

    # Change list to numpy array
    spikes = np.asarray(spikes)

    # Iterate thorugh all spike times
    for i in range(len(spikes)):

        # Determine lags to all later spikes
        lags = spikes[i + 1 :] - spikes[i]

        # Keep lags above some threshold and remove above max_lag
        lags = lags[(np.abs(lags) >= 1e-12) & (np.abs(lags) <= max_lag)]
        diffs.append(lags)

    # Flatten list into one large array
    diffs = np.concatenate(diffs) if len(diffs) > 0 else np.array([])

    # Default to symmetric ACG, provide negative lags
    diffs = np.concatenate([diffs, -diffs])

    # Define bins and bins centers as determined by max_lag
    bin_edges = np.arange(-max_lag, max_lag + bin_size, bin_size)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Calculate the ACG by bins
    acorr, _ = np.histogram(diffs, bins=bin_edges)

    # Determine rate of spike train
    rate = len(spikes) / T

    # Normalize histogram for better comparison across datasets
    edge = np.maximum(1e-12, T - np.abs(bin_centers))
    acorr = acorr / (rate**2 * (edge * bin_size))
    return bin_centers, acorr


def cv(spikes):
    """
    Compute coefficient of variation (CV) of a spike train.

    Parameters
    ----------
    spikes : array_like
        1D array of sorted spike times

    Returns
    -------
    float
        Coefficient of variation of ISIs
    """
    # Compute ISIs
    diff = np.diff(spikes)

    # Calculate and return CV
    return np.std(diff) / np.mean(diff)

def sliding_window(spikes, T0, T1, window, step=None):
    """
    Computes the firing rate over sliding windows seperated of step

    Parameters
    ----------
    spikes : np.ndarray
        Sorted array of spike times
    T0 : float
        Beginning of spike train to consider, start of bins
    T1 : float
        End of spike train to consider, end of bins
    window : float
        Size of window to compute firing rate
    step : float
        (optional) How far the window moves between bins

    Returns
    -------
    bin_centers : np.ndarray
        Array of midpoints between bin edges
    rates : np.ndarray
        Array of spike count for each bin
    """

    # if no step, define step as the window
    if step is None:
        step = window

    # Create beginnig of window for all bins
    bin_starts = np.arange(T0, T1 - window, step)

    # Determine bin centers for plotting
    bin_centers = bin_starts + window / 2

    # Iterate through bins and determine rate in that bin
    counts = []
    for bs in bin_starts:
        bin_spikes = spikes[(spikes < bs + window) & (spikes >= bs)]
        counts.append(len(bin_spikes))
    return bin_centers, np.array(counts)

def fano_factor(spikes, T0, T1, bin_size):
    """
    Compute fano factor of a spike train.

    Parameters
    ----------
    spikes : array_like
        1D array of sorted spike times

    Returns
    -------
    float
        Coefficient of variation of ISIs
    """
    # Compute counts for a sliding window
    bin_centers, counts = sliding_window(spikes, T0, T1, bin_size)

    # Evaluate fano factor
    ff = np.std(counts) ** 2 / np.mean(counts)
    return ff
