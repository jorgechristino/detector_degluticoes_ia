import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Função para aplicar filtro de mediana para remover ruído em um arquivo de áudio
def denoise_audio(audio_data, sample_rate):
    # Aplicar filtro de mediana para remover ruído
    denoised_audio = medfilt(audio_data, kernel_size=3)  
    return denoised_audio

# Função para extrair características de áudio
def extract_features(audio_data, sample_rate, mfcc=True, chroma=True, mel=True, spectral_centroid=True, zero_crossing_rate=True):
        features = []
        
        # Calcular a média da amplitude do sinal de áudio
        amplitude_mean = np.mean(np.abs(audio_data))
        features.append(amplitude_mean)
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
            features.extend(mfccs)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sample_rate).T,axis=0)
            features.extend(chroma)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T,axis=0)
            features.extend(mel)
        if spectral_centroid:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0])
            features.append(spectral_centroid)
        if zero_crossing_rate:
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data))
            features.append(zero_crossing_rate)
            
        return features

# Função para detectar períodos de deglutição em arquivos de áudio
def detect_swallow_periods(audio_files, interval_duration=0.1, threshold=0.5, min_duration=0.5, max_duration=2.0):
    all_swallow_periods = []
    
    for audio_file in audio_files:
        swallow_periods = []
        # Carregar o arquivo de áudio
        y, sr = librosa.load(audio_file)
        
        # Calcular a média da amplitude do sinal de áudio
        amplitude_mean = np.mean(np.abs(y))
        
        # Definir o limite para detecção de deglutições
        threshold = amplitude_mean * threshold
        interval_samples = int(interval_duration * sr)
        
        # Encontrar períodos de deglutição
        swallow_indices = np.where(np.abs(y) > threshold)[0]
        
        # Agrupar períodos de deglutição em intervalos maiores
        for i in range(0, len(swallow_indices), interval_samples):
            start = swallow_indices[i] / sr
            end = swallow_indices[min(i + interval_samples, len(swallow_indices) - 1)] / sr
            duration = end - start
            
            # Verificar se a duração do intervalo é maior que o mínimo
            if duration >= min_duration and duration <= max_duration:
                # Verificar se o início do intervalo atual é posterior ao final do intervalo anterior
                if start > swallow_periods[-1][1] if swallow_periods else True:
                    swallow_periods.append((start, end))

        all_swallow_periods.append(swallow_periods)

    return all_swallow_periods

# Diretório onde os arquivos de áudio estão localizados
audio_dir = "MP3"

# Lista para armazenar as características extraídas
features = []

# Lista para armazenar os nomes dos arquivos de áudio válidos
audio_files = []

# Extrair características de cada arquivo de áudio na pasta
for file_name in os.listdir(audio_dir):
    if file_name.endswith(".mp3"):
        # Carregar o áudio usando librosa
        audio_data, sample_rate = librosa.load(os.path.join(audio_dir, file_name))
        # denoised_audio = denoise_audio(audio_data, sample_rate)

        # Extrair características do áudio
        features.append(extract_features(audio_data, sample_rate))

        if(file_name == 'a00200.mp3'):
            break

# Convertendo a lista de características para uma matriz numpy
features = np.array(features)

# Aplicar K-Means para agrupar os dados
kmeans = KMeans(n_clusters=2)  # número de clusters pode ser ajustado
kmeans.fit(features)

# Rotular os clusters
labels = kmeans.labels_

# Identificar outliers como pontos que estão longe do centro do cluster
# distances = kmeans.transform(features)
# threshold = np.percentile(distances, 95)  # Ajuste o limite conforme necessário
# outliers = np.where(distances > threshold)[0]

# Identificar os arquivos de áudio que são identificados como áudios válidos
audio_files_indices = np.where(labels == 1)[0]

# Exibir os arquivos de áudio que são identificados como outliers
print("Deglutições detectadas nos seguintes arquivos:")
for idx in audio_files_indices:
    audio_files.append(os.path.join(audio_dir, os.listdir(audio_dir)[idx]))
    print(os.listdir(audio_dir)[idx])

# Aplicar PCA para reduzir a dimensionalidade dos dados para visualização
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Plotar os clusters
plt.figure(figsize=(8, 6))
for i in range(len(features_pca)):
    if labels[i] == 0:
        plt.scatter(features_pca[i, 0], features_pca[i, 1], color='blue', alpha=0.5)
    else:
        plt.scatter(features_pca[i, 0], features_pca[i, 1], color='red', alpha=0.5)

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Clusters K-Means')
plt.show()

true_labels = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0,    # 1 - 10
               1, 1, 0, 1, 1, 0, 0,             # 11 - 20
               0, 0, 0, 0, 0, 1, 1, 1, 1,       # 21 - 30
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,    # 31 - 40
               0, 1, 1, 1, 0, 0, 0, 0, 1, 0,    # 41 - 50
               1, 1, 1, 0, 1, 1, 1, 1, 0,       # 51 - 60
               1, 1, 1, 0, 1, 0, 1, 0, 1, 0,    # 61 - 70
               0, 0, 1, 1, 1, 1, 0, 0, 0, 1,    # 71 - 80
               0, 0, 1, 0, 0, 0, 0, 1, 1, 1,    # 81 - 90
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,    # 91 - 100
               1, 1, 0, 1, 1, 1, 1, 0, 1,       # 101 - 110
               1, 1, 1, 1, 0, 1, 1, 1, 1,       # 111 - 120
               1, 1, 1, 1, 1, 0, 1, 1, 0, 1,    # 121 - 130
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,    # 131 - 140
               1, 1, 1, 1, 1, 1, 1, 0, 1, 1,    # 141 - 150
               1, 0, 1, 1, 1, 1, 1, 1, 1,       # 151 - 160
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,    # 161 - 170
               1, 0, 1, 1, 1, 0, 0, 0, 1, 0,    # 171 - 180
               1, 1, 1, 1, 1, 1, 0, 0, 1,       # 181 - 190
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,    # 191 - 200
               ]    

# Calcular a acurácia
accuracy_1 = accuracy_score(true_labels, labels)

# Calcular a pontuação F1
f1 = f1_score(true_labels, labels, average='weighted')  # ou 'macro', 'micro', dependendo do seu caso

print("Modelo não supervisionado:")
print("Acurácia :", accuracy_1)
print("Pontuação F1:", f1)


# Detectar períodos de deglutição em arquivos de áudio
# swallow_periods = detect_swallow_periods(audio_files)

# print("Períodos de deglutição identificados:")
# for idx, swallow_periods in enumerate(swallow_periods):
#     print(f"Arquivo {audio_files[idx]}:")
#     for start, end in swallow_periods:
#         print(f"De {start:.2f} segundos até {end:.2f} segundos.")

# Separar os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, true_labels, test_size=0.2, random_state=42)

# Instanciar um objeto RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Ajustar o modelo aos dados de treinamento
random_forest.fit(X_train, y_train)

# Fazer previsões nos dados de teste
predictions = random_forest.predict(X_test)

# Avaliar o desempenho do modelo
accuracy_2 = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)


print("\nModelo supervisionado:")
print("Acurácia:", accuracy_2)
print("Relatório de Classificação:")
print(report)