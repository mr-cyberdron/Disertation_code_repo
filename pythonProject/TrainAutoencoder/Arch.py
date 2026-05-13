import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline


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


class EncoderLinear(nn.Module):
    def __init__(self, logit_len=1000, input_len=5500):
        super(EncoderLinear, self).__init__()
        # Calculate neuron counts automatically
        input_size = 3 * input_len  # 16500
        hidden1_size = input_size // 2  # First reduction: 8250
        hidden2_size = hidden1_size // 2  # Second reduction: 4125

        self.input_len = input_len

        # Linear layers with automatically calculated sizes
        self.linear1 = nn.Linear(input_size, hidden1_size)
        self.ln2 = nn.LayerNorm(int(input_len / 2))
        self.ln3 = nn.LayerNorm(int(input_len / 4))
        self.af1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)
        self.af2 = nn.ReLU()
        self.fc_compress = nn.Linear(hidden2_size, logit_len)
        self.af3 = nn.Softsign()

        self.atention = CrossAttentionBlock(latent_dim=logit_len, feature_dim=1, embed_dim=256, num_heads=8)

        self.fc1 = nn.Linear(256, logit_len)
        self.af4 = nn.Softmax(dim=1)
        self.flat1 = nn.Flatten()
        self.ln1 = nn.LayerNorm(logit_len)
        self.drop1 = nn.Dropout(0.2)

    def forward(self, x):
        # Assuming x shape is [batch, 3, 5500]
        x = x.view(x.size(0), -1)  # Flatten to [batch, 16500]
        x = self.linear1(x)  # [batch, 8250]
        x = self.af1(x)
        x = self.linear2(x)  # [batch, 4125]
        x = self.af2(x)
        x = x.unsqueeze(1)
        x_seq = x.permute(0, 2, 1)  # [batch, 1375, 32]
        x = self.fc_compress(x.squeeze(1))  # [batch, 1000]
        latent = self.af3(x)

        attention_out, w = self.atention(latent, x_seq)
        att = self.fc1(attention_out)
        att = self.af4(att)
        # att = self.drop1(att)

        x = self.ln1(att + latent)

        return x

class Encoder(nn.Module):
    def __init__(self,logit_len = 1000, input_len = 5500):
        super(Encoder, self).__init__()
        # Слои энкодера
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, stride=2, padding=1)  # [batch, 16, 2750]
        self.ln2 = nn.LayerNorm(int(input_len/2))
        self.ln3 = nn.LayerNorm(int(input_len/4))
        self.af1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)  # [batch, 32, 1375]

        self.fc_compress = nn.Linear(32 * int(input_len/4), logit_len)
        self.af3 = nn.Softsign()

        self.atention = CrossAttentionBlock(latent_dim=logit_len, feature_dim=32, embed_dim=256, num_heads=8)

        self.fc1 = nn.Linear(256,logit_len)

        self.af4 = nn.Softmax(dim=1)

        self.flat1 = nn.Flatten()

        self.ln1 = nn.LayerNorm(logit_len)

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
        att = self.drop1(att)



        x = self.ln1(att+latent)

        return x

class Decoder(nn.Module):
    def __init__(self, logit_len = 1000, input_len = 5500):
        super(Decoder, self).__init__()

        self.input_len = input_len
        # Слои декодера
        self.fc_expand = nn.Linear(logit_len, 32 * int(input_len/4))  # [batch, 32 * 1375]
        self.af3 = nn.ReLU()
        self.conv_trans1 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # [batch, 16, 2750]
        self.af1 = nn.ReLU()
        self.conv_trans2 = nn.ConvTranspose1d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)  # [batch, 3, 5500]
        self.af2 = nn.LeakyReLU()

        self.drop1 = nn.Dropout(0.2)

    def forward(self, latent, batch_size):
        latent = self.drop1(latent)
        x = self.fc_expand(latent)  # Расширяем [batch, 1000] -> [batch, 32 * 1375]
        expanded = self.af3(x)
        x = expanded.view(batch_size, 32, int(self.input_len/4))  # Преобразуем в 3D [batch, 32, 1375]
        x = self.conv_trans1(x) # Первая транспонированная свертка + активация
        x = self.af1(x)
        x = self.conv_trans2(x)  # Вторая транспонированная свертка
        x = self.af2(x)
        return x


class SplineDecoder(nn.Module):
    def __init__(self, logit_len=1000, input_len=5000, out_channels=3):
        super(SplineDecoder, self).__init__()
        self.input_len = input_len
        self.out_channels = out_channels
        self.total_len = out_channels * input_len

    def forward(self, latent, batch_size):
        # latent: (batch, 1000)
        x = latent.unsqueeze(1)  # (batch, 1, 1000)
        x = F.interpolate(x, size=self.total_len, mode='linear', align_corners=False)  # (batch, 1, 16500)
        x = x.view(batch_size, self.out_channels, self.input_len)  # (batch, 3, 5500)
        return x

class ChannelwiseSplineDecoder(nn.Module):
    def __init__(self, logit_len=1000, input_len=5500, out_channels=3):
        super(ChannelwiseSplineDecoder, self).__init__()
        self.input_len = input_len
        self.out_channels = out_channels

        if logit_len % out_channels != 0:
            raise ValueError(f"logit_len ({logit_len}) должно делиться на out_channels ({out_channels}) без остатка.")

        self.per_channel_latent = logit_len // out_channels

    def forward(self, latent, batch_size):
        # latent: (batch, 1000)
        splits = torch.chunk(latent, self.out_channels, dim=1)  # -> список (batch, 1000 // 3)
        upsampled_channels = []
        for ch_latent in splits:
            x = ch_latent.unsqueeze(1)  # (batch, 1, per_channel_latent)
            x_up = F.interpolate(x, size=self.input_len, mode='linear', align_corners=False)  # (batch, 1, 5500)
            upsampled_channels.append(x_up)

        # Объединяем в (batch, 3, 5500)
        return torch.cat(upsampled_channels, dim=1)

class FlatLinearDecoder(nn.Module):
    def __init__(self, logit_len=1000, input_len=5500, out_channels=3):
        super(FlatLinearDecoder, self).__init__()
        self.input_len = input_len
        self.out_channels = out_channels
        self.linear = nn.Linear(logit_len, input_len * out_channels, bias=False)

    def forward(self, latent, batch_size):
        # latent: (batch, 1000)
        x = self.linear(latent)  # (batch, 16500)
        x = x.view(batch_size, self.out_channels, self.input_len)  # (batch, 3, 5500)
        return x


def interp_using_patterns():
    """Каждому сгенерированому компоненту свой вес"""
    class PatternDecoder(nn.Module):
        def __init__(self, latent_dim=1000, out_channels=3, out_length=5000):
            super().__init__()
            # Предзаданный банк паттернов: [latent_dim, channels, time]
            patterns = torch.randn(latent_dim, out_channels, out_length)

            # Можно здесь зафиксировать шаблоны (или включить обучение)
            self.register_buffer("patterns", patterns)  # не обучается

        def forward(self, z):
            # z: (batch, latent_dim)
            # Выход: (batch, out_channels, out_length)
            return torch.einsum('bl,lco->bco', z, self.patterns)

    def generate_wavelet_patterns(latent_dim, channels, length, base_wave='gaussian'):
        patterns = np.zeros((latent_dim, channels, length))
        x = np.linspace(-1, 1, length)

        for i in range(latent_dim):
            scale = np.random.uniform(0.05, 0.3)
            shift = np.random.uniform(-0.7, 0.7)

            if base_wave == 'gaussian':
                wave = np.exp(-((x - shift) ** 2) / (2 * scale ** 2))
            elif base_wave == 'morlet':
                wave = np.cos(5 * np.pi * (x - shift)) * np.exp(-((x - shift) ** 2) / (2 * scale ** 2))
            else:
                wave = np.sin(2 * np.pi * (x - shift) / scale)

            for c in range(channels):
                amp = np.random.uniform(0.5, 1.5)
                patterns[i, c, :] = amp * wave

        return torch.tensor(patterns, dtype=torch.float32)
class SimpleECG_Autoencoder(nn.Module):
    def __init__(self, num_classes = 2, logit_len = 1000, input_len = 5500):
        super(SimpleECG_Autoencoder, self).__init__()
        # self.encoder = Encoder(logit_len=logit_len, input_len=input_len)
        self.encoder = EncoderLinear(logit_len=logit_len, input_len=input_len)

        # self.decoder = Decoder(logit_len=logit_len, input_len=input_len)
        # self.decoder = SplineDecoder(logit_len=logit_len, input_len=input_len)
        self.decoder = ChannelwiseSplineDecoder(logit_len=logit_len, input_len=input_len)
        # self.decoder = FlatLinearDecoder(logit_len=logit_len, input_len=input_len)
        # interp_using_patterns() - для дальнейшей разработки
        # Классификационная голова: из latent [batch, 1000] в [batch, num_classes]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(logit_len, num_classes)
        )

    def forward(self, x):
        # Проход через энкодер
        latent = self.encoder(x,)
        # Проход через декодер
        batch_size = x.size(0)
        output = self.decoder(latent, batch_size)  # [batch, 3, 5500]
        class_logits = self.classifier(latent)  # [batch, num_classes]
        return output, latent, class_logits