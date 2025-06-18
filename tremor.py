import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, detrend
import matplotlib.pyplot as plt
import io

# ---------- Funções de processamento ----------


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)


def compute_resultant(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


def compute_rms(signal):
    return np.sqrt(np.mean(signal**2))


def compute_envelope(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope


def compute_fft(signal, fs):
    N = len(signal)
    freq = np.fft.rfftfreq(N, d=1/fs)
    fft_magnitude = np.abs(np.fft.rfft(signal)) / N
    return freq, fft_magnitude


def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')


def detect_separator(file_content):
    first_line = file_content.decode('utf-8').split('\n')[0]
    if first_line.count(';') > first_line.count(',') and first_line.count(';') > first_line.count('\t'):
        return ';'
    elif first_line.count('\t') > first_line.count(','):
        return '\t'
    else:
        return ','


def interpolate_signal(time, signal, target_fs):
    target_time = np.arange(time[0], time[-1], 1/target_fs)
    interpolated = np.interp(target_time, time, signal)
    return target_time, interpolated


# ---------- Interface Streamlit ----------
st.set_page_config(layout="wide")
st.title("Análise de Tremor de Mãos")

uploaded_file = st.file_uploader(
    "📂 Faça o upload do arquivo (.csv ou .txt)", type=["csv", "txt"])

if uploaded_file is not None:
    file_content = uploaded_file.read()
    sep = detect_separator(file_content)
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), sep=sep)

    # ---------- Configurações ----------
    st.sidebar.header("Configurações de Análise")
    target_fs = 100  # Frequência final após interpolação
    lowcut = st.sidebar.slider(
        "Frequência de Corte Inferior (Hz)", 1.0, 10.0, 4.0)
    highcut = st.sidebar.slider(
        "Frequência de Corte Superior (Hz)", 8.0, 20.0, 12.0)
    window_size = st.sidebar.slider(
        "Tamanho da Janela para Média Móvel (Envelope Espectral)", 3, 50, 20)

    # ---------- Carregar e interpolar ----------
    time_raw = df.iloc[500:, 0].values
    x_raw = df.iloc[500:, 1].values
    y_raw = df.iloc[500:, 2].values
    z_raw = df.iloc[500:, 3].values
    time_raw = time_raw/1000

    time_interp, x_interp = interpolate_signal(time_raw, x_raw, target_fs)
    _, y_interp = interpolate_signal(time_raw, y_raw, target_fs)
    _, z_interp = interpolate_signal(time_raw, z_raw, target_fs)

    # ---------- Detrend ----------
    x_detrended = detrend(x_interp)
    y_detrended = detrend(y_interp)
    z_detrended = detrend(z_interp)

    # ---------- Filtro Banda Passa ----------
    x_f = bandpass_filter(x_detrended, lowcut, highcut, target_fs)
    y_f = bandpass_filter(y_detrended, lowcut, highcut, target_fs)
    z_f = bandpass_filter(z_detrended, lowcut, highcut, target_fs)

    # ---------- Resultante no Tempo ----------
    resultant_time = compute_resultant(x_f, y_f, z_f)
    rms_value = compute_rms(resultant_time)
    envelope_time = compute_envelope(resultant_time)
    mean_envelope = np.mean(envelope_time)

    # ---------- FFT por eixo ----------
    freqx, fft_x = compute_fft(x_f, target_fs)
    freqy, fft_y = compute_fft(y_f, target_fs)
    freqz, fft_z = compute_fft(z_f, target_fs)

    # ---------- Resultante espectral ----------
    spectral_resultant = np.sqrt(fft_x**2 + fft_y**2 + fft_z**2)
    envelope_spectral = moving_average(
        spectral_resultant, window_size=window_size)

    peak_freq_resultant = freqx[np.argmax(envelope_spectral)]
    peak_magnitude_resultant = np.max(envelope_spectral)

    band_mask = (freqx >= 3) & (freqx <= 12)
    band_power_resultant = np.trapz(
        envelope_spectral[band_mask], freqx[band_mask])

    # ---------- Resultados ----------
    st.subheader("📈 Resultados Globais")
    st.markdown(f"**RMS (3–12 Hz):** {rms_value:.4f} m/s²")
    st.markdown(f"**Média do Envelope (tempo):** {mean_envelope:.4f} m/s²")
    st.markdown(
        f"**Frequência de Pico (Resultante Espectral):** {peak_freq_resultant:.2f} Hz")
    st.markdown(
        f"**Magnitude de Pico (Resultante Espectral):** {peak_magnitude_resultant:.4e}")
    st.markdown(f"**Área Espectral (3–12 Hz):** {band_power_resultant:.4e}")

    # ---------- Plots ----------
    st.subheader("📊 Gráficos")
    col1, col2 = st.columns(2)

    with col1:
        fig_time, ax_time = plt.subplots(figsize=(8, 5))
        ax_time.plot(time_interp, x_f, label='X')
        ax_time.plot(time_interp, y_f, label='Y')
        ax_time.plot(time_interp, z_f, label='Z')
        ax_time.set_xlabel('Tempo (s)')
        ax_time.set_ylabel('Aceleração (m/s²)')
        ax_time.legend()
        st.pyplot(fig_time)

    with col2:
        fig_spec, ax_spec = plt.subplots(figsize=(8, 5))
        ax_spec.plot(freqx, spectral_resultant, label='FFT Resultante')
        ax_spec.plot(freqx, envelope_spectral, color='red', linewidth=2,
                     label=f'Envelope Espectral (Janela={window_size})')
        ax_spec.set_xlim(0, 20)
        ax_spec.set_xlabel('Frequência (Hz)')
        ax_spec.set_ylabel('Magnitude')
        ax_spec.set_title('FFT Resultante com Envelope Espectral')
        ax_spec.legend()
        st.pyplot(fig_spec)

    # ---------- Exportação ----------

        results = pd.DataFrame({
            "RMS": [rms_value],
            "Mean_Envelope_Tempo": [mean_envelope],
            "Peak_Frequency_Resultant": [peak_freq_resultant],
            "Peak_Magnitude_Resultant": [peak_magnitude_resultant],
            "Band_Power_3_12Hz": [band_power_resultant]
        })

        st.download_button(
            label="Download CSV",
            data=results.to_csv(index=False).encode('utf-8'),
            file_name='features_tremor_fft_envelope.csv',
            mime='text/csv'
        )
