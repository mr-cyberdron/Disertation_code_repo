import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricAutoencoderWithClassifier(nn.Module):
    def __init__(self, channels, signal_size, hidden_dims, num_classes=4, latent_coeff_size=150):
        """
        channels: количество каналов (3)
        signal_size: размер сигнала (500)
        hidden_dims: список размеров скрытых слоев ([300, 150, 90])
        num_classes: количество классов для классификации (4)
        latent_coeff_size: размер сжатого вектора коэффициентов (90)
        """
        super(SymmetricAutoencoderWithClassifier, self).__init__()

        # Сохраняем параметры
        self.channels = channels
        self.signal_size = signal_size
        self.input_dim = channels * signal_size  # 3 * 500 = 1500
        self.hidden_dims = hidden_dims
        self.latent_coeff_size = latent_coeff_size
        self.num_classes = num_classes

        # Энкодер
        self.encoder_weights = nn.ParameterList()
        dims = [self.input_dim] + hidden_dims  # [1500, 300, 150, 90]
        for i in range(len(dims) - 1):
            weight = nn.Parameter(torch.empty(dims[i], dims[i + 1]))
            nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('relu'))
            self.encoder_weights.append(weight)

        # Слой для сжатия коэффициентов
        total_weight_size = sum(w.numel() for w in self.encoder_weights)
        self.coeff_compressor = nn.Linear(total_weight_size, latent_coeff_size)

        # Классификатор
        latent_size = hidden_dims[-1] + latent_coeff_size  # 90 + 90 = 180
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Активационная функция
        self.activation = nn.ReLU()

    def get_compressed_coeffs(self):
        flat_weights = torch.cat([w.view(-1) for w in self.encoder_weights])
        compressed_coeffs = self.coeff_compressor(flat_weights)
        return compressed_coeffs

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.channels * self.signal_size)

        # Энкодер
        h = x
        for i, weight in enumerate(self.encoder_weights):
            h = torch.matmul(h, weight)
            h = self.activation(h)

        latent_data = h  # [50, 90]
        latent_coeffs = self.get_compressed_coeffs().unsqueeze(0).expand(batch_size, -1)  # [50, 90]
        latent = torch.cat([latent_data, latent_coeffs], dim=1)  # [50, 180]

        # Классификация
        class_logits = self.classifier(latent)  # [50, 4]

        # Декодер
        decoded = latent_data
        for i in range(len(self.encoder_weights) - 1, 0, -1):
            decoded = torch.matmul(decoded, self.encoder_weights[i].t())
            decoded = self.activation(decoded)
        decoded = torch.matmul(decoded, self.encoder_weights[0].t())
        decoded = decoded.view(batch_size, self.channels, self.signal_size)

        return decoded, latent, class_logits



class SymmetricAutoencoderWithClassifier2(nn.Module):
    def __init__(self, channels = 12, signal_size = 500,
                 hidden_dims = [1200,600,160],
                 # hidden_dims=[3000, 1500, 800],
                 num_classes=15,
                 latent_coeff_size=80,
                 # latent_coeff_size=240,
                 latent_att_cof_size = None
                 ):


        super(SymmetricAutoencoderWithClassifier2, self).__init__()

        # Сохраняем параметры
        self.latent_att_cof_size = latent_att_cof_size
        self.channels = channels
        self.signal_size = signal_size
        self.hidden_dims = hidden_dims
        self.input_dim = int(self.hidden_dims[-1]/8) * int(signal_size/4)
        self.latent_coeff_size = latent_coeff_size
        self.num_classes = num_classes

        # Энкодер
        self.encoder_weights = nn.ParameterList()
        dims = [self.input_dim] + hidden_dims  # [1500, 300, 150, 90]
        for i in range(len(dims) - 1):
            weight = nn.Parameter(torch.empty(dims[i], dims[i + 1]))
            nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('leaky_relu'))
            self.encoder_weights.append(weight)

        # Слой для сжатия коэффициентов


        # Классификатор
        # latent_size = hidden_dims[-1] + latent_coeff_size+(self.latent_att_cof_size*8)  # 90 + 90 = 180
        latent_size = hidden_dims[-1] + latent_coeff_size  # 90 + 90 = 180
        # self.classifier = nn.Sequential(
        #     nn.Linear(latent_size, 128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128,50),
        #     nn.LeakyReLU(),
        #     nn.Linear(50, num_classes)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(latent_size, num_classes),
        )

        # Активационная функция
        self.activation = nn.LeakyReLU()

        self.shared_weights = nn.Parameter(torch.empty(int(self.hidden_dims[-1]/8), channels, 4))
        nn.init.xavier_normal_(self.shared_weights, gain=nn.init.calculate_gain('leaky_relu'))

        self.conv1 = nn.Conv1d(channels,int(self.hidden_dims[-1]/8),5,4,)
        self.conv1.weight = self.shared_weights

        self.mha = nn.MultiheadAttention(embed_dim=int(self.hidden_dims[-1]/8), num_heads=5, batch_first=True)
        self.conv1_t = nn.ConvTranspose1d(int(self.hidden_dims[-1]/8), channels, 5, 4, )
        self.conv1_t.weight = self.shared_weights

        self.coeff_compressor = nn.AdaptiveMaxPool1d(latent_coeff_size)

    def get_compressed_coeffs(self):
        flat_weights = torch.cat([w.view(-1) for w in self.encoder_weights])
        flat_weights2 = torch.cat([w.view(-1) for w in self.shared_weights])
        fw = torch.cat([flat_weights, flat_weights2])
        fw = fw.unsqueeze(0).unsqueeze(0)
        compressed_coeffs = self.coeff_compressor(fw)
        compressed_coeffs = compressed_coeffs.squeeze()
        return compressed_coeffs



    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)

        x = self.activation(x)

        query = x.permute(0, 2, 1)
        x = x.view(batch_size, int(self.hidden_dims[-1]/8) * int(self.signal_size/4))

        # Энкодер
        h = x
        for i, weight in enumerate(self.encoder_weights):
            h_1 = torch.matmul(h, weight)
            # h = self.bn_mass[i](h)
            h_1 = self.activation(h_1)
            h_residual = F.adaptive_max_pool1d(h,self.hidden_dims[i])
            h = h_residual+h_1

        k_v = h.view(batch_size, 8,int(self.hidden_dims[-1]/8))

        attn_output, attn_weights = self.mha(k_v,query,query)
        # attn_weights_compressed = F.adaptive_max_pool1d(attn_weights,self.latent_att_cof_size).reshape(batch_size, -1)
        attn_output = attn_output+k_v
        attn_output = F.layer_norm(attn_output,[8,int(self.hidden_dims[-1]/8)])
        h = attn_output.reshape(batch_size, self.hidden_dims[-1])


        latent_data = h  # [50, 90]
        latent_coeffs = self.get_compressed_coeffs().unsqueeze(0).expand(batch_size, -1)  # [50, 90]
        latent = torch.cat([latent_data, latent_coeffs], dim=1)  # [50, 180]
        # latent = torch.cat([latent_data, latent_coeffs, attn_weights_compressed], dim=1)  # [50, 180]

        # def center_and_boost(x, power=0.1):
        #
        #     # Центрування по останній осі
        #     mean = x[..., 160:200].mean(dim=-1, keepdim=True)
        #     centered = x - mean
        #
        #     # Нелінійне підсилення малих значень
        #     boosted = torch.sign(centered) * (torch.abs(centered) ** power)
        #     return boosted

        # latent = center_and_boost(latent,power=0.3)

        # print(latent.size())

        # Классификация
        class_logits = self.classifier(latent)  # [50, 4]

        # Декодер
        decoded = latent_data
        for i in range(len(self.encoder_weights) - 1, 0, -1):
            decoded = torch.matmul(decoded, self.encoder_weights[i].t())
            decoded = self.activation(decoded)
        decoded = torch.matmul(decoded, self.encoder_weights[0].t())


        decoded = decoded.view(batch_size, int(self.hidden_dims[-1]/8), int(self.signal_size/4))
        decoded = self.conv1_t(decoded)

        return decoded, latent, class_logits