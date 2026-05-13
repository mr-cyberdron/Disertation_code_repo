import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def limit_spectrum(signal, cutoff_freq_hz, fs):
    """
    Ограничивает спектр сигнала до заданной частоты в герцах.

    Аргументы:
        signal: тензор [batch_size, signal_length] — входной сигнал (например, [200, 1000])
        cutoff_freq_hz: float — частота среза в герцах (например, 50.0)
        fs: float — частота дискретизации в герцах (например, 500.0)

    Возвращает:
        filtered_signal: тензор [batch_size, signal_length] — сигнал с ограниченным спектром
    """
    signal_length = signal.size(1)  # Длина сигнала (например, 1000)

    # Преобразуем сигнал в частотную область
    signal_fft = torch.fft.rfft(signal, dim=1)  # [batch_size, signal_length//2 + 1], комплексные числа

    # Частотный шаг (Гц) = fs / signal_length
    freq_step = fs / signal_length  # Например, 500 / 1000 = 0.5 Гц

    # Индекс, соответствующий частоте среза
    freq_limit_idx = int(cutoff_freq_hz / freq_step)  # Например, 50 / 0.5 = 100
    freq_limit_idx = min(freq_limit_idx, signal_fft.size(1))  # Ограничиваем размером спектра

    # Ограничиваем спектр
    signal_fft_limited = torch.zeros_like(signal_fft)
    signal_fft_limited[:, :freq_limit_idx] = signal_fft[:, :freq_limit_idx]

    # Обратное преобразование в пространственную область
    filtered_signal = torch.fft.irfft(signal_fft_limited, n=signal_length, dim=1)  # [batch_size, signal_length]

    return filtered_signal

class CrossAttentionBlock(nn.Module):
    def __init__(self, latent_dim=1000, feature_dim=32, embed_dim=256, num_heads=8):
        super(CrossAttentionBlock, self).__init__()

        self.query_proj = nn.Linear(latent_dim, embed_dim)     # [batch, latent_dim] → [batch, embed_dim]
        self.kv_proj = nn.Linear(feature_dim, embed_dim)       # [batch, seq_len, feature_dim] → [batch, seq_len, embed_dim]

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, latent, features):
        """
        latent:   [batch, latent_dim] — сжатый вектор после conv + fc
        features: [batch, seq_len, feature_dim] — признаки до сжатия (например, [batch, 1375, 32])
        """
        # 1. Преобразуем latent → query (добавим временную размерность)
        q = self.query_proj(latent).unsqueeze(1)   # [batch, 1, embed_dim]

        # 2. Преобразуем признаки (conv-выход) → key и value
        kv = self.kv_proj(features)                # [batch, seq_len, embed_dim]

        # 3. Вызов внимания
        attn_output, attn_weights = self.attn(q, kv, kv,need_weights=True, average_attn_weights=True)  # Q: [batch, 1, embed_dim], K/V: [batch, seq_len, embed_dim]

        # 4. Удаляем временное измерение
        attended_latent = attn_output.squeeze(1)   # [batch, embed_dim]
        return attended_latent, attn_weights

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Слои энкодера
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, stride=2, padding=1)  # [batch, 16, 2750]
        self.ln2 = nn.LayerNorm(2750)
        self.ln3 = nn.LayerNorm(1375)
        self.af1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)  # [batch, 32, 1375]
        self.fc_compress = nn.Linear(32 * 1375, 1000)
        self.af3 = nn.Softsign()#nn.LeakyReLU()

        self.atention = CrossAttentionBlock(latent_dim=1000, feature_dim=32, embed_dim=256, num_heads=8)

        self.fc1 = nn.Linear(256,1000)

        self.af4 = nn.Softmax(dim=1)

        self.flat1 = nn.Flatten()

        self.ln1 = nn.LayerNorm(1000)

        self.drop1 = nn.Dropout(0.2)



    def forward(self, x):
        x = self.conv1(x)
        x = self.af1(x)  # Первая свертка + активация
        x = self.ln2(x)
        x = self.conv2(x)              # Вторая свертка
        x = self.ln3(x)
        x_flat = x.view(x.size(0), -1)  # Разворачиваем [batch, 32 * 1375]
        x_seq = x.permute(0, 2, 1)
        x = self.fc_compress(x_flat)  # Сжимаем до [batch, 1000]
        latent = self.af3(x)

        # latent = self.drop1(latent)

        attention_out, w = self.atention(latent,x_seq)

        att = self.fc1(attention_out)
        att = self.af4(att)


        x = self.ln1(att+latent)

        # x = limit_spectrum(x,150,500)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Слои декодера
        self.fc_expand = nn.Linear(1000, 32 * 1375)  # [batch, 32 * 1375]
        self.af3 = nn.ReLU()
        self.conv_trans1 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # [batch, 16, 2750]
        self.af1 = nn.ReLU()
        self.conv_trans2 = nn.ConvTranspose1d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)  # [batch, 3, 5500]
        self.af2 = nn.LeakyReLU()

    def forward(self, latent, batch_size):
        x =  self.fc_expand(latent)  # Расширяем [batch, 1000] -> [batch, 32 * 1375]
        expanded = self.af3(x)
        x = expanded.view(batch_size, 32, 1375)  # Преобразуем в 3D [batch, 32, 1375]
        x = self.conv_trans1(x) # Первая транспонированная свертка + активация
        x = self.af1(x)
        x = self.conv_trans2(x)  # Вторая транспонированная свертка
        x = self.af2(x)
        return x

class SimpleECG_Autoencoder(nn.Module):
    def __init__(self, num_classes = 2):
        super(SimpleECG_Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # Классификационная голова: из latent [batch, 1000] в [batch, num_classes]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        # Проход через энкодер
        latent = self.encoder(x)  # [batch, 1000]
        # Проход через декодер
        batch_size = x.size(0)
        output = self.decoder(latent, batch_size)  # [batch, 3, 5500]
        class_logits = self.classifier(latent)  # [batch, num_classes]
        return output, latent, class_logits