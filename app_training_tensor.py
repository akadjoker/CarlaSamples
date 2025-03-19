"""
Modelo de Direção Autônoma
--------------------------
Este script implementa um modelo de direção autônoma baseado na arquitetura NVIDIA,
utilizando imagens de câmera para predizer o ângulo de direção.
Configurado para rodar apenas na CPU.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import glob
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

# Forçar uso de CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Importações do Keras
import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input

# Configurações
DATASET_DIR = 'dataset'  # Diretório principal com sessões de dados
BATCH_SIZE = 64  # Tamanho de batch reduzido para CPU
EPOCHS = 10
LEARNING_RATE = 1e-3
IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3

# =========================================================================
# Funções de carregamento e processamento de dados
# =========================================================================

def load_sessions_data():
    """
    Carrega dados de todas as sessões na pasta dataset.
    Cada sessão contém uma pasta 'images' e um arquivo 'steering_data.csv'.
    """
    all_image_paths = []
    all_steering_angles = []
    
    # Encontra todas as pastas de sessão
    session_dirs = glob.glob(os.path.join(DATASET_DIR, 'session_*'))
    
    if not session_dirs:
        raise ValueError(f"Nenhuma pasta de sessão encontrada em {DATASET_DIR}")
    
    print(f"Encontradas {len(session_dirs)} sessões de dados")
    
    # Processa cada sessão
    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        csv_path = os.path.join(session_dir, 'steering_data.csv')
        
        if not os.path.exists(csv_path):
            print(f"Aviso: Arquivo CSV não encontrado em {session_dir}, pulando sessão")
            continue
        
        print(f"Carregando dados da sessão: {session_name}")
        
        # Carrega o CSV
        try:
            df = pd.read_csv(csv_path)
            
            # Verifica se o CSV tem as colunas esperadas
            if 'image_path' not in df.columns or 'steering' not in df.columns:
                print(f"Aviso: Formato de CSV inválido em {session_name}, pulando sessão")
                continue
            
            # Processa as linhas do CSV
            for _, row in df.iterrows():
                # Constrói o caminho completo para a imagem
                image_path = os.path.join(session_dir, row['image_path'])
                
                # Verifica se a imagem existe
                if os.path.exists(image_path):
                    all_image_paths.append(image_path)
                    all_steering_angles.append(float(row['steering']))
                else:
                    # Verifica se há um caminho alternativo
                    alt_image_path = os.path.join(session_dir, 'images', os.path.basename(row['image_path']))
                    if os.path.exists(alt_image_path):
                        all_image_paths.append(alt_image_path)
                        all_steering_angles.append(float(row['steering']))
            
            print(f"Adicionadas {len(all_image_paths)} imagens da sessão {session_name}")
            
        except Exception as e:
            print(f"Erro ao processar sessão {session_name}: {str(e)}")
    
    return np.array(all_image_paths), np.array(all_steering_angles)

def balance_data(image_paths, steering_angles, num_bins=25, samples_per_bin=400):
    """
    Balanceia os dados para evitar viés no treino.
    Retorna arrays balanceados de caminhos de imagem e ângulos de direção.
    """
    # Histograma dos ângulos de direção
    hist, bins = np.histogram(steering_angles, num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5
    
    # Visualização antes do balanceamento
    plt.figure(figsize=(10, 5))
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(steering_angles), np.max(steering_angles)), 
            (samples_per_bin, samples_per_bin))
    plt.title('Distribuição de ângulos de direção (antes do balanceamento)')
    plt.xlabel('Ângulo de direção')
    plt.ylabel('Frequência')
    
    print(f'Total de dados original: {len(steering_angles)}')
    
    # Remoção de exemplos excessivos para cada bin
    remove_indices = []
    for j in range(num_bins):
        bin_indices = []
        for i in range(len(steering_angles)):
            if steering_angles[i] >= bins[j] and steering_angles[i] <= bins[j+1]:
                bin_indices.append(i)
        
        # Se temos mais amostras do que o limite para este bin
        if len(bin_indices) > samples_per_bin:
            bin_indices = shuffle(bin_indices)
            remove_indices.extend(bin_indices[samples_per_bin:])
    
    print(f'Removendo {len(remove_indices)} amostras para balancear os dados')
    
    # Cria máscaras para filtrar os arrays
    keep_mask = np.ones(len(steering_angles), dtype=bool)
    keep_mask[remove_indices] = False
    
    balanced_image_paths = image_paths[keep_mask]
    balanced_steering_angles = steering_angles[keep_mask]
    
    print(f'Restantes após balanceamento: {len(balanced_steering_angles)}')
    
    # Visualização após o balanceamento
    hist, _ = np.histogram(balanced_steering_angles, num_bins)
    plt.figure(figsize=(10, 5))
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(balanced_steering_angles), np.max(balanced_steering_angles)), 
            (samples_per_bin, samples_per_bin))
    plt.title('Distribuição de ângulos de direção (após balanceamento)')
    plt.xlabel('Ângulo de direção')
    plt.ylabel('Frequência')
    
    return balanced_image_paths, balanced_steering_angles

def visualize_distribution(y_train, y_valid, num_bins=25):
    """Visualiza a distribuição dos conjuntos de treino e validação."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
    axes[0].set_title('Conjunto de treino')
    axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
    axes[1].set_title('Conjunto de validação')
    plt.tight_layout()

# =========================================================================
# Funções de aumento de dados (data augmentation)
# =========================================================================

def zoom(image):
    """Aplica zoom aleatório na imagem."""
    zoom = iaa.Affine(scale=(1, 1.3))
    return zoom.augment_image(image)

def pan(image):
    """Desloca a imagem horizontalmente e verticalmente."""
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    return pan.augment_image(image)

def img_random_brightness(image):
    """Altera o brilho da imagem aleatoriamente."""
    brightness = iaa.Multiply((0.2, 1.2))
    return brightness.augment_image(image)

def img_random_flip(image, steering_angle):
    """Inverte a imagem horizontalmente e ajusta o ângulo de direção."""
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

def random_augment(image_path, steering_angle):
    """Aplica transformações aleatórias em uma imagem carregada do caminho."""
    try:
        image = mpimg.imread(image_path)
        
        # Aplica cada transformação com 50% de probabilidade
        if np.random.rand() < 0.5:
            image = pan(image)
        if np.random.rand() < 0.5:
            image = zoom(image)
        if np.random.rand() < 0.5:
            image = img_random_brightness(image)
        if np.random.rand() < 0.5:
            image, steering_angle = img_random_flip(image, steering_angle)
        
        return image, steering_angle
    except Exception as e:
        print(f"Erro ao aumentar imagem {image_path}: {str(e)}")
        # Se houver erro, tenta retornar a imagem original
        try:
            return mpimg.imread(image_path), steering_angle
        except:
            # Em caso de falha completa, retorna uma imagem preta
            return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)), steering_angle

def visualize_augmentation(image_paths, steerings, num_examples=5):
    """Visualiza exemplos de imagens originais e aumentadas."""
    ncol = 2
    nrow = num_examples
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, 4*num_examples))
    fig.tight_layout()
    
    for i in range(num_examples):
        randnum = random.randint(0, len(image_paths) - 1)
        random_image_path = image_paths[randnum]
        random_steering = steerings[randnum]
        
        try:
            original_image = mpimg.imread(random_image_path)
            augmented_image, steering = random_augment(random_image_path, random_steering)
            
            axs[i][0].imshow(original_image)
            axs[i][0].set_title("Imagem Original")
            
            axs[i][1].imshow(augmented_image)
            axs[i][1].set_title("Imagem Aumentada")
        except Exception as e:
            print(f"Erro ao visualizar aumento de dados: {str(e)}")
    
    plt.tight_layout()

# =========================================================================
# Pré-processamento de imagens
# =========================================================================

def img_preprocess(img):
    """Pré-processa a imagem para o formato esperado pelo modelo."""
    try:
        # Verifica se a imagem está no formato correto
        if img is None or img.size == 0:
            return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        
        # Corta a imagem para remover partes irrelevantes (ajuste conforme necessário)
        img = img[60:135, :, :]
        
        # Converte para o espaço de cores YUV (usado pela NVIDIA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        
        # Aplica blur Gaussiano
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Redimensiona para o tamanho esperado pelo modelo
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Normaliza os valores de pixel
        img = img / 255
        
        return img
    except Exception as e:
        print(f"Erro no pré-processamento da imagem: {str(e)}")
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

def visualize_preprocessing(image_paths):
    """Visualiza o resultado do pré-processamento."""
    if len(image_paths) == 0:
        print("Nenhuma imagem disponível para visualizar o pré-processamento")
        return
    
    image_path = image_paths[random.randint(0, len(image_paths) - 1)]
    
    try:
        original_image = mpimg.imread(image_path)
        preprocessed_image = img_preprocess(original_image)
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.tight_layout()
        
        axs[0].imshow(original_image)
        axs[0].set_title('Imagem Original')
        
        axs[1].imshow(preprocessed_image)
        axs[1].set_title('Imagem Pré-processada')
    except Exception as e:
        print(f"Erro ao visualizar pré-processamento: {str(e)}")

# =========================================================================
# Gerador de lotes (batch generator)
# =========================================================================

def batch_generator(image_paths, steering_ang, batch_size, is_training):
    """Gerador que fornece lotes de imagens e ângulos de direção."""
    while True:
        batch_img = []
        batch_steering = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            
            try:
                if is_training:
                    # Aplica aumento de dados no conjunto de treino
                    im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
                else:
                    # Usa imagens originais para validação
                    im = mpimg.imread(image_paths[random_index])
                    steering = steering_ang[random_index]
                
                # Pré-processa a imagem
                im = img_preprocess(im)
                batch_img.append(im)
                batch_steering.append(steering)
            except Exception as e:
                print(f"Erro ao gerar lote: {str(e)}")
                # Em caso de erro, usa uma imagem preta
                batch_img.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
                batch_steering.append(0.0)
        
        yield (np.asarray(batch_img), np.asarray(batch_steering))

# =========================================================================
# Modelo NVIDIA para direção autônoma
# =========================================================================

def nvidia_model():
    """Cria e compila o modelo NVIDIA para direção autônoma."""
    print("Verificando backend para TensorFlow...")
    from tensorflow.python.client import device_lib
    print("Dispositivos disponíveis:", device_lib.list_local_devices())
    
    with keras.utils.CustomObjectScope({'GlorotUniform': keras.initializers.glorot_uniform()}):
        model = Sequential([
            # Define a camada de entrada explicitamente
            Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)),
            
            # Camadas convolucionais
            Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
            Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
            Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
            Conv2D(64, (3, 3), activation='elu'),
            Conv2D(64, (3, 3), activation='elu'),
            
            # Camada de achatamento
            Flatten(),
            
            # Camadas densas
            Dense(100, activation='elu'),
            Dense(50, activation='elu'),
            Dense(10, activation='elu'),
            Dense(1)  # Camada de saída para o ângulo de direção
        ])
    
    # Compilação do modelo
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mse', optimizer=optimizer)
    
    return model

# =========================================================================
# Função principal
# =========================================================================

def main():
    """Função principal que orquestra todo o fluxo de trabalho."""
    try:
        print("Keras em CPU - Modelo de Direção Autônoma")
        print("-----------------------------------------")
        print(f"Versão do Keras: {keras.__version__}")
        print(f"Versão do TensorFlow: {keras.backend.backend()}")
        
        # Verifica dispositivos
        print("\nConfigurando TensorFlow para CPU...")
        import tensorflow as tf
        print("Dispositivos TensorFlow:", tf.config.list_physical_devices())
        
        # Carrega os dados
        print("\nCarregando dados de todas as sessões...")
        image_paths, steering_angles = load_sessions_data()
        
        if len(image_paths) == 0:
            print("Nenhum dado foi carregado. Verifique a estrutura das pastas e arquivos.")
            return
        
        print(f"Total de imagens carregadas: {len(image_paths)}")
        
        # Balanceia os dados
        print("Balanceando dados...")
        image_paths, steering_angles = balance_data(image_paths, steering_angles)
        
        # Divide em conjuntos de treino e validação
        print("Dividindo em conjuntos de treino e validação...")
        X_train, X_valid, y_train, y_valid = train_test_split(
            image_paths, steering_angles, test_size=0.2, random_state=6)
        
        print(f'Amostras de Treino: {len(X_train)}\nAmostras de Validação: {len(X_valid)}')
        
        # Visualiza a distribuição dos conjuntos
        visualize_distribution(y_train, y_valid)
        
        # Visualiza exemplos de aumento de dados
        print("Visualizando exemplos de aumento de dados...")
        visualize_augmentation(X_train, y_train)
        
        # Visualiza o pré-processamento
        print("Visualizando exemplo de pré-processamento...")
        visualize_preprocessing(X_train)
        
        # Validação de batch generator
        print("Testando batch generator...")
        x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
        x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
        
        # Visualiza exemplos do batch generator
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.tight_layout()
        axs[0].imshow(x_train_gen[0])
        axs[0].set_title(f'Imagem de Treino - Ângulo: {y_train_gen[0]:.4f}')
        axs[1].imshow(x_valid_gen[0])
        axs[1].set_title(f'Imagem de Validação - Ângulo: {y_valid_gen[0]:.4f}')
        
        # Criação do modelo
        print("Criando modelo NVIDIA...")
        model = nvidia_model()
        model.summary()
        
        # treino do modelo
        print("Iniciando treino...")
        steps_per_epoch = min(300, len(X_train) // BATCH_SIZE)
        validation_steps = min(200, len(X_valid) // BATCH_SIZE)
        
        print(f"Passos por época: {steps_per_epoch}, Passos de validação: {validation_steps}")
        
        history = model.fit(
            batch_generator(X_train, y_train, BATCH_SIZE, 1),
            steps_per_epoch=steps_per_epoch, 
            epochs=EPOCHS,
            validation_data=batch_generator(X_valid, y_valid, BATCH_SIZE, 0),
            validation_steps=validation_steps,
            verbose=1
        )
        
        # Visualiza a perda durante o treino
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['treino', 'validação'])
        plt.title('Perda durante treino')
        plt.xlabel('Época')
        plt.ylabel('Erro Quadrático Médio')
        
        # Salva o modelo
        print("Salvando modelo...")
        model.save('modelo_direcao_autonoma.keras')
        print("Modelo salvo com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

# Executa a função principal quando o script é executado diretamente
if __name__ == "__main__":
    main()
