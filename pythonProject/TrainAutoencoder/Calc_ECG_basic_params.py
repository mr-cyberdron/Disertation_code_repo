import numpy as np
import neurokit2 as nk
import pandas as pd
from sklearn.manifold import TSNE  # Для выполнения t-SNE
import matplotlib.pyplot as plt    # Для построения графиков

# Предположим, у вас есть массив сигналов размером (100, 3, 5000)
np.random.seed(42)
ecg_signals = np.load('./AutoencoderData/X_train.npy')
# ecg_signals = ecg_signals[0:200,:,500:]
ecg_signals = ecg_signals[:,:,500:]
# Частота дискретизации
sampling_rate = 500  # 500 Гц

# Список для хранения результатов
results = []

# Обработка каждой записи
counter = 0
for record_idx in range(ecg_signals.shape[0]):  # По 100 записям
    counter +=1
    print(f'{counter}/{ecg_signals.shape[0]}')
    record_data = {}

    # Обработка каждого канала (отведения)
    for channel_idx in range(ecg_signals.shape[1]):  # По 3 каналам
        ecg_signal = ecg_signals[record_idx, channel_idx, :]
        signal_length = len(ecg_signal)  # 5000 в данном случае

        # NeuroKit2: базовая обработка
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        hr = signals["ECG_Rate"].mean() if not signals["ECG_Rate"].empty else np.nan

        # Позиции пиков с фильтрацией по границам
        r_peaks = [peak for peak in info["ECG_R_Peaks"] if 0 <= peak < signal_length]
        p_peaks = [peak for peak in info["ECG_P_Peaks"] if 0 <= peak < signal_length]
        q_peaks = [peak for peak in info["ECG_Q_Peaks"] if 0 <= peak < signal_length]
        s_peaks = [peak for peak in info["ECG_S_Peaks"] if 0 <= peak < signal_length]
        t_peaks = [peak for peak in info["ECG_T_Peaks"] if 0 <= peak < signal_length]

        # Интервалы (в миллисекундах) с синхронизацией массивов
        if len(p_peaks) > 0 and len(r_peaks) > 0:
            min_len_pr = min(len(p_peaks), len(r_peaks))
            pr_interval = np.nanmean(
                np.array(r_peaks[:min_len_pr]) - np.array(p_peaks[:min_len_pr])) / sampling_rate * 1000
        else:
            pr_interval = np.nan

        if len(q_peaks) > 0 and len(s_peaks) > 0:
            min_len_qs = min(len(q_peaks), len(s_peaks))
            qrs_duration = np.nanmean(
                np.array(s_peaks[:min_len_qs]) - np.array(q_peaks[:min_len_qs])) / sampling_rate * 1000
        else:
            qrs_duration = np.nan

        if len(q_peaks) > 0 and len(t_peaks) > 0:
            min_len_qt = min(len(q_peaks), len(t_peaks))
            qt_interval = np.nanmean(
                np.array(t_peaks[:min_len_qt]) - np.array(q_peaks[:min_len_qt])) / sampling_rate * 1000
        else:
            qt_interval = np.nan

        # HRV из NeuroKit2
        if len(r_peaks) > 1:
            hrv_metrics = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)
            rmssd = hrv_metrics["HRV_RMSSD"].values[0]
            sdnn = hrv_metrics["HRV_SDNN"].values[0]
            pnn50 = hrv_metrics["HRV_pNN50"].values[0]
        else:
            rmssd = sdnn = pnn50 = np.nan

        # Амплитуды пиков
        p_amplitude = np.nanmean(ecg_signal[p_peaks]) if len(p_peaks) > 0 else np.nan
        q_amplitude = np.nanmean(ecg_signal[q_peaks]) if len(q_peaks) > 0 else np.nan
        r_amplitude = np.nanmean(ecg_signal[r_peaks]) if len(r_peaks) > 0 else np.nan
        s_amplitude = np.nanmean(ecg_signal[s_peaks]) if len(s_peaks) > 0 else np.nan
        t_amplitude = np.nanmean(ecg_signal[t_peaks]) if len(t_peaks) > 0 else np.nan

        # Дополнительные точки: длительности волн
        p_onsets = [peak for peak in info.get("ECG_P_Onsets", []) if 0 <= peak < signal_length]
        p_offsets = [peak for peak in info.get("ECG_P_Offsets", []) if 0 <= peak < signal_length]
        t_onsets = [peak for peak in info.get("ECG_T_Onsets", []) if 0 <= peak < signal_length]
        t_offsets = [peak for peak in info.get("ECG_T_Offsets", []) if 0 <= peak < signal_length]

        if len(p_onsets) > 0 and len(p_offsets) > 0:
            min_len_p = min(len(p_onsets), len(p_offsets))
            p_duration = np.nanmean(
                np.array(p_offsets[:min_len_p]) - np.array(p_onsets[:min_len_p])) / sampling_rate * 1000
        else:
            p_duration = np.nan

        if len(t_onsets) > 0 and len(t_offsets) > 0:
            min_len_t = min(len(t_onsets), len(t_offsets))
            t_duration = np.nanmean(
                np.array(t_offsets[:min_len_t]) - np.array(t_onsets[:min_len_t])) / sampling_rate * 1000
        else:
            t_duration = np.nan

        # Добавляем параметры для текущего канала с суффиксом (_1, _2, _3)
        channel_suffix = f"_{channel_idx + 1}"
        record_data.update({
            f"Heart_Rate{channel_suffix}": hr,
            f"PR_Interval{channel_suffix}": pr_interval,
            f"QRS_Duration{channel_suffix}": qrs_duration,
            f"QT_Interval{channel_suffix}": qt_interval,
            f"RMSSD{channel_suffix}": rmssd,
            f"SDNN{channel_suffix}": sdnn,
            f"pNN50{channel_suffix}": pnn50,
            f"P_Amplitude{channel_suffix}": p_amplitude,
            f"Q_Amplitude{channel_suffix}": q_amplitude,
            f"R_Amplitude{channel_suffix}": r_amplitude,
            f"S_Amplitude{channel_suffix}": s_amplitude,
            f"T_Amplitude{channel_suffix}": t_amplitude,
            f"P_Duration{channel_suffix}": p_duration,
            f"T_Duration{channel_suffix}": t_duration
        })

    # Добавляем номер записи
    record_data["Record"] = record_idx
    results.append(record_data)

# Преобразуем в DataFrame
df = pd.DataFrame(results)

# Переставляем колонку Record в начало
df = df[["Record"] + [col for col in df.columns if col != "Record"]]

# Вывод первых строк
print(df.head())

print('Fix_NAN')


columns_to_process = [col for col in df.columns if col != "Record"]

# Проходим по каждому столбцу
for col in columns_to_process:
    # Вычисляем медиану столбца (игнорируя NaN)
    median_value = df[col].median()

    # 1. Заменяем NaN на медиану
    df[col] = df[col].fillna(median_value)

    # 2. Определяем 1-й и 99-й процентили
    p1 = df[col].quantile(0.01)  # 1-й процентиль
    p99 = df[col].quantile(0.99)  # 99-й процентиль

    # Заменяем значения, выходящие за пределы 1-99 процентиля, на медиану
    df[col] = df[col].apply(lambda x: median_value if x < p1 or x > p99 else x)

print(df.head())
df.to_csv('ECG_params_df.csv')

print('T-sne...')

# Подготовка данных для t-SNE
X = df.drop(columns=["Record"]).values

# Применяем t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Построение графика
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='blue', alpha=0.6)
plt.title("t-SNE Visualization of ECG Parameters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

# Добавляем номера записей
for i, record in enumerate(df["Record"]):
    plt.annotate(record, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8)

plt.grid(True)
plt.show()