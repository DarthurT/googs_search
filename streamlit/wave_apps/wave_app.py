import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq
from scipy.stats import norm


def jonswap_spectrum(frequency, Hs, Tp, gamma):
    """
    Calculate JONSWAP spectrum.

    :param frequency: Frequency array
    :param Hs: Significant wave height (meters)
    :param Tp: Peak period (seconds)
    :param gamma: Shape parameter
    :return: JONSWAP spectrum values
    """
    g = 9.81  # acceleration due to gravity
    sigma = np.where(frequency <= 1 / Tp, 0.07, 0.09)
    alpha = 0.076 * (Hs**2 / Tp**4) ** (-0.22)
    r = np.exp(-(0.5 * ((frequency * Tp - 1) / sigma) ** 2))
    S = (
        alpha
        * g**2
        * frequency ** (-5)
        * np.exp(-5 / 4 * (Tp * frequency) ** (-4))
        * gamma**r
    )
    return S


def discretize_spectrum(frequencies, spectrum, num_waves):
    """
    Discretize the spectrum into a set of linear waves.

    :param frequencies: Frequency array
    :param spectrum: Spectrum values
    :param num_waves: Number of linear waves
    :return: Tuple of arrays (amplitudes, frequencies)
    """
    df = frequencies[1] - frequencies[0]
    wave_energies = spectrum * df
    sorted_indices = np.argsort(wave_energies)[::-1]
    top_indices = sorted_indices[:num_waves]
    top_frequencies = frequencies[top_indices]
    top_energies = wave_energies[top_indices]
    top_amplitudes = np.sqrt(2 * top_energies)
    return top_amplitudes, top_frequencies


def wave_field(x, y, t, amplitude, period, direction, phase):
    frequency = 2 * np.pi / period
    k = 2 * np.pi / (period * np.sqrt(direction[0] ** 2 + direction[1] ** 2))

    # Stokes' Second Order Theory Components
    stokes_second_order_correction = (k * amplitude**2) / 2
    wave_component = np.sin(
        frequency * t - k * (direction[0] * x + direction[1] * y) + phase
    )
    stokes_component = np.sin(
        2 * (frequency * t - k * (direction[0] * x + direction[1] * y) + phase)
    )

    return (
        amplitude * wave_component + stokes_second_order_correction * stokes_component
    )


def plot_waves(waves, t, x_range, y_range):
    X, Y = np.meshgrid(x_range, y_range)
    Z = sum(
        wave_field(
            X, Y, t, wave["amplitude"], wave["period"], wave["direction"], wave["phase"]
        )
        for wave in waves
    )

    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(121, projection="3d")  # Adjust for subplot
    surf = ax.plot_surface(X, Y, Z, cmap="ocean")
    ax.set_title(f"Wave Field at t={t} seconds")
    ax.set_zlim(-4, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    surf.axes.set_box_aspect([50, 50, 3])

    return fig


st.title("Wave Superposition Visualizer")

# Add toggle for JONSWAP spectrum-based wave generation
jonswap_based = st.sidebar.checkbox("Use JONSWAP Spectrum")

if jonswap_based:
    # Input fields for JONSWAP spectrum parameters
    Hs = st.sidebar.number_input("Significant Wave Height", 0.1, 10.0, 2.0)
    Tp = st.sidebar.number_input("Peak Period (s)", 5, 30, 12)
    gamma = st.sidebar.number_input("Peak Enhancement Factor (gamma)", 1.0, 7.0, 3.3)
    n_waves = st.sidebar.number_input("Number of waves", 1, 200, 128)
    direction_range = st.sidebar.slider(
        "Direction Range (degrees)", 0, 360, (0, 180), step=1
    )
    phase_range = st.sidebar.slider(
        "Phase Range (radians)", 0.0, 2 * np.pi, (0.0, np.pi)
    )
    # Frequency range
    f_min = 0.01
    f_max = 0.5
    num_freq = 500
    frequencies = np.linspace(f_min, f_max, num_freq)

    # Calculate JONSWAP spectrum
    spectrum = jonswap_spectrum(frequencies, Hs, Tp, gamma)
    # Discretize the spectrum
    amplitudes, wave_frequencies = discretize_spectrum(frequencies, spectrum, n_waves)
    amplitudes = amplitudes / 1000
    waves = []
    for i in range(n_waves):
        amplitude = amplitudes[i]
        period = wave_frequencies[i]
        direction_angle = np.random.uniform(direction_range[0], direction_range[1])
        direction = (
            np.cos(np.radians(direction_angle)),
            np.sin(np.radians(direction_angle)),
        )
        phase = np.random.uniform(phase_range[0], phase_range[1])
        waves.append(
            {
                "amplitude": amplitude,
                "period": period,
                "direction": direction,
                "phase": phase,
            }
        )

else:
    n_waves = st.sidebar.number_input("Number of waves", 1, 200, 128)
    amplitude_range = st.sidebar.slider("Amplitude Range", 0.01, 0.5, (0.01, 0.05))
    period_range = st.sidebar.slider("Period Range (s)", 5, 30, (10, 20))
    direction_range = st.sidebar.slider(
        "Direction Range (degrees)", 0, 360, (0, 180), step=1
    )
    phase_range = st.sidebar.slider(
        "Phase Range (radians)", 0.0, 2 * np.pi, (0.0, np.pi)
    )

    waves = []
    for i in range(n_waves):
        amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
        period = np.random.uniform(period_range[0], period_range[1])
        direction_angle = np.random.uniform(direction_range[0], direction_range[1])
        direction = (
            np.cos(np.radians(direction_angle)),
            np.sin(np.radians(direction_angle)),
        )
        phase = np.random.uniform(phase_range[0], phase_range[1])
        waves.append(
            {
                "amplitude": amplitude,
                "period": period,
                "direction": direction,
                "phase": phase,
            }
        )

t = st.slider("Time", 0.0, 30.0, 0.0, step=0.5)

x_range = np.linspace(-100, 100, 250)
y_range = np.linspace(-100, 100, 250)

fig = plot_waves(waves, t, x_range, y_range)
st.pyplot(fig)
