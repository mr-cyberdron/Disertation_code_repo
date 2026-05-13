import pandas as pd
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px


# Загрузка данных
df = pd.read_csv('./ECG_params_df.csv')

# Подготовка данных
columns_to_process = [col for col in df.columns if col != "Record"]
X = df[columns_to_process].values



# Кластеризация
n_clusters = 8
kmeans = KMeans(
    n_clusters=n_clusters,          # Кол-во кластеров — выбери по смыслу или через метрику (см. ниже)
    init='k-means++',      # Умный выбор начальных центров
    random_state=42        # Фиксируем для повторяемости
)
df["Cluster"] = kmeans.fit_predict(X)

# Вывод DataFrame с кластерами
print("DataFrame с метками кластеров:")
print(df)
df.to_csv('ECG_params_df_clusterised.csv')

# Применяем t-SNE для визуализации
tsne = TSNE(n_components=2, random_state=42)  # Для 5 строк
X_tsne = tsne.fit_transform(X)

# Визуализация кластеров с легендой
plt.figure(figsize=(10, 6))

# Список цветов для кластеров (можно расширить при необходимости)
colors = [
    "#1f77b4",  # синий
    "#ff7f0e",  # оранжевый
    "#2ca02c",  # зеленый
    "#d62728",  # красный
    "#9467bd",  # фиолетовый
    "#8c564b",  # коричневый
    "#e377c2",  # розовый
    "#7f7f7f",  # серый
    "#bcbd22",  # оливковый
    "#17becf",  # голубой
    "#aec7e8",  # светло-синий
    "#ffbb78",  # светло-оранжевый
    "#98df8a",  # светло-зеленый
    "#ff9896",  # светло-красный
    "#c5b0d5",  # светло-фиолетовый
]

# Рисуем точки для каждого кластера отдельно
for cluster in range(n_clusters):
    cluster_points = X_tsne[df["Cluster"] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)

# Добавляем номера записей
# for i, record in enumerate(df["Record"]):
#     plt.annotate(record, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8)

# Настройки графика
plt.title("t-SNE Visualization with K-Means Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()  # Добавляем легенду
plt.grid(True)
plt.show()

#
# t-SNE в 3D
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X)

# Добавим координаты в DataFrame
df["TSNE-1"] = X_tsne[:, 0]
df["TSNE-2"] = X_tsne[:, 1]
df["TSNE-3"] = X_tsne[:, 2]

# Визуализация через Plotly
fig = px.scatter_3d(
    df, x='TSNE-1', y='TSNE-2', z='TSNE-3',
    color=df["Cluster"].astype(str),
    # text='Record',
    title='3D t-SNE Visualization with K-Means Clusters',
    labels={'color': 'Cluster'}
)

fig.update_traces(marker=dict(size=5), textposition='top center')
fig.update_layout(legend_title_text='Cluster', margin=dict(l=0, r=0, b=0, t=30))

fig.show()


#
# # Визуализация
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# colors = ['blue', 'green', 'red', 'purple', 'orange']
#
# for cluster in range(n_clusters):
#     cluster_points = X_tsne[df["Cluster"] == cluster]
#     ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
#                c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
#
# # Добавляем подписи записей
# for i, record in enumerate(df["Record"]):
#     ax.text(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], record, fontsize=8)
#
# # Настройки графика
# ax.set_title("3D t-SNE Visualization with K-Means Clusters")
# ax.set_xlabel("t-SNE Component 1")
# ax.set_ylabel("t-SNE Component 2")
# ax.set_zlabel("t-SNE Component 3")
# ax.legend()
# plt.tight_layout()
# plt.show()



#

#
# # Применяем DBSCAN
# dbscan = DBSCAN(eps=30, min_samples=2)  # eps и min_samples нужно подобрать
# dbscan_labels = dbscan.fit_predict(X)
#
# # Добавляем метки DBSCAN в DataFrame
# df["DBSCAN_Cluster"] = dbscan_labels
#
# # Визуализация DBSCAN
# plt.figure(figsize=(10, 6))
# unique_labels = np.unique(dbscan_labels)
# colors_dbscan = matplotlib.colormaps.get_cmap('hsv')
#
# for i, cluster in enumerate(unique_labels):
#     cluster_points = X_tsne[df["DBSCAN_Cluster"] == cluster]
#     label = "Noise" if cluster == -1 else f"Cluster {cluster}"
#     color_idx = cluster if cluster >= 0 else 3  # -1 -> gray
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
#                 c=[colors_dbscan(i / len(unique_labels))], label=label, alpha=0.6)
#
# # for i, record in enumerate(df["Record"]):
# #     plt.annotate(record, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8)
#
# plt.title("t-SNE Visualization with DBSCAN Clusters")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Вывод DataFrame с кластерами
# print("DataFrame с метками кластеров (K-Means и DBSCAN):")
# print(df)