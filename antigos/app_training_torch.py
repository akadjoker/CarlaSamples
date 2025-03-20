
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import shutil

# Verificar se GPU está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Configurações
DATASET_DIR = 'dataset'  # Diretório principal com sessões de dados
BATCH_SIZE = 64
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
            bin_indices = np.random.choice(bin_indices, len(bin_indices) - samples_per_bin, replace=False)
            remove_indices.extend(bin_indices)
    
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

# =========================================================================
# Funções de aumento de dados (data augmentation)
# =========================================================================

def zoom_image(image, zoom_factor=None):
    """
    Aplica zoom na imagem usando interpolação do OpenCV.
    Se zoom_factor não for especificado, usa um valor aleatório entre 1.0 e 1.3.
    """
    if zoom_factor is None:
        zoom_factor = random.uniform(1.0, 1.3)
    
    h, w = image.shape[:2]
    
    # Calcula novos limites para o recorte
    h_half, w_half = h / 2, w / 2
    h_crop, w_crop = int(h / zoom_factor), int(w / zoom_factor)
    h_start, w_start = int(h_half - h_crop / 2), int(w_half - w_crop / 2)
    
    # Recorta e redimensiona
    cropped = image[h_start:h_start + h_crop, w_start:w_start + w_crop]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed

def pan_image(image, x_shift=None, y_shift=None):
    """
    Desloca a imagem horizontalmente e verticalmente usando transformação afim.
    Se x_shift ou y_shift não for especificado, usa valores aleatórios entre -10% e 10%.
    """
    h, w = image.shape[:2]
    
    if x_shift is None:
        x_shift = int(w * random.uniform(-0.1, 0.1))
    if y_shift is None:
        y_shift = int(h * random.uniform(-0.1, 0.1))
    
    # Matriz de transformação para translação
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return shifted

def adjust_brightness(image, brightness_factor=None):
    """
    Ajusta o brilho da imagem.
    Se brightness_factor não for especificado, usa um valor aleatório entre 0.2 e 1.2.
    """
    if brightness_factor is None:
        brightness_factor = random.uniform(0.2, 1.2)
    
    # Converte para HSV para ajustar o valor (brilho)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    
    # Converte de volta para RGB
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return adjusted

def flip_image(image, steering_angle):
    """
    Inverte a imagem horizontalmente e ajusta o ângulo de direção.
    """
    flipped_image = cv2.flip(image, 1)
    flipped_angle = -steering_angle
    
    return flipped_image, flipped_angle

def random_augment(image, steering_angle):
    """
    Aplica transformações aleatórias em uma imagem.
    """
    # Aplica cada transformação com 50% de probabilidade
    if random.random() < 0.5:
        image = pan_image(image)
    if random.random() < 0.5:
        image = zoom_image(image)
    if random.random() < 0.5:
        image = adjust_brightness(image)
    if random.random() < 0.5:
        image, steering_angle = flip_image(image, steering_angle)
    
    return image, steering_angle

# =========================================================================
# Pré-processamento de imagens
# =========================================================================

def img_preprocess(img):
    """Pré-processa a imagem para o formato esperado pelo modelo."""
    try:
        # Verifica se a imagem está no formato correto
        if img is None or img.size == 0:
            return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        
        # Corta a imagem para remover partes irrelevantes
        img = img[60:135, :, :]
        
        # Converte para o espaço de cores YUV (usado pela NVIDIA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        
        # Aplica blur Gaussiano
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Redimensiona para o tamanho esperado pelo modelo
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Normaliza os valores de pixel
        img = img / 255.0
        
        return img
    except Exception as e:
        print(f"Erro no pré-processamento da imagem: {str(e)}")
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

# =========================================================================
# Dataset e DataLoader do PyTorch
# =========================================================================

class DrivingDataset(Dataset):
    """Dataset personalizado para os dados de direção autônoma."""
    
    def __init__(self, image_paths, steering_angles, transform=False):
        """
        Inicializa o dataset.
        
        Args:
            image_paths (list): Lista de caminhos para as imagens
            steering_angles (list): Lista de ângulos de direção
            transform (bool): Se True, aplica aumento de dados
        """
        self.image_paths = image_paths
        self.steering_angles = steering_angles
        self.transform = transform
    
    def __len__(self):
        """Retorna o tamanho do dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Retorna um item do dataset pelo índice.
        
        Args:
            idx (int): Índice do item
            
        Returns:
            tuple: (imagem, ângulo de direção)
        """
        img_path = self.image_paths[idx]
        steering_angle = self.steering_angles[idx]
        
        try:
            # Carrega a imagem
            img = mpimg.imread(img_path)
            
            # Aplica aumento de dados se necessário
            if self.transform:
                img, steering_angle = random_augment(img, steering_angle)
            
            # Pré-processa a imagem
            img = img_preprocess(img)
            
            # Converte para tensor do PyTorch (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            img = torch.FloatTensor(img)
            
            return img, torch.FloatTensor([steering_angle])
        
        except Exception as e:
            print(f"Erro ao carregar imagem {img_path}: {str(e)}")
            # Em caso de erro, retorna uma imagem preta e ângulo zero
            img = np.zeros((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
            return torch.FloatTensor(img), torch.FloatTensor([0.0])

# =========================================================================
# Modelo NVIDIA para direção autônoma (implementação PyTorch)
# =========================================================================

class NvidiaModel(nn.Module):
    """Implementação do modelo NVIDIA para direção autônoma usando PyTorch."""
    
    def __init__(self):
        """Inicializa as camadas do modelo."""
        super(NvidiaModel, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        # Camadas densas
        self.fc1 = nn.Linear(1152, 100)  # 1152 = 64 * 3 * 6 (calculado a partir das dimensões de entrada)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass pelo modelo.
        
        Args:
            x (torch.Tensor): Tensor de entrada com shape (batch_size, 3, 66, 200)
            
        Returns:
            torch.Tensor: Tensor de saída com shape (batch_size, 1)
        """
        # Camadas convolucionais com ativação ELU
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Camadas densas com ativação ELU e dropout
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        
        return x

# =========================================================================
# Funções de treino e avaliação
# =========================================================================

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs):
    """
    Treina o modelo.
    
    Args:
        model (nn.Module): Modelo PyTorch
        train_loader (DataLoader): DataLoader para o conjunto de treino
        valid_loader (DataLoader): DataLoader para o conjunto de validação
        criterion: Função de perda
        optimizer: Otimizador
        epochs (int): Número de épocas
        
    Returns:
        dict: Histórico de treino
    """
    # Move o modelo para a GPU se disponível
    model = model.to(device)
    
    # Inicializa o histórico
    history = {
        'train_loss': [],
        'valid_loss': []
    }
    
    # Melhor perda para salvar o melhor modelo
    best_valid_loss = float('inf')
    
    # Loop de épocas
    for epoch in range(epochs):
        start_time = time.time()
        
        # Treino
        model.train()
        train_loss = 0.0
        
        # Barra de progresso para o treino
        train_pbar = tqdm(train_loader, desc=f'Época {epoch+1}/{epochs} (Treino)')
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            # Move os dados para a GPU se disponível
            data, target = data.to(device), target.to(device)
            
            # Zera os gradientes
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calcula a perda
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Atualiza os pesos
            optimizer.step()
            
            # Acumula a perda
            train_loss += loss.item()
            
            # Atualiza a barra de progresso
            train_pbar.set_postfix({'loss': train_loss / (batch_idx + 1)})
        
        # Calcula a perda média
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validação
        model.eval()
        valid_loss = 0.0
        
        # Barra de progresso para a validação
        valid_pbar = tqdm(valid_loader, desc=f'Época {epoch+1}/{epochs} (Valid)')
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_pbar):
                # Move os dados para a GPU se disponível
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Calcula a perda
                loss = criterion(output, target)
                
                # Acumula a perda
                valid_loss += loss.item()
                
                # Atualiza a barra de progresso
                valid_pbar.set_postfix({'loss': valid_loss / (batch_idx + 1)})
        
        # Calcula a perda média
        valid_loss /= len(valid_loader)
        history['valid_loss'].append(valid_loss)
        
        # Salva o melhor modelo
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'melhor_modelo.pth')
            print(f'Modelo salvo com perda de validação: {valid_loss:.6f}')
        
        # Calcula o tempo gasto
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # Mostra o progresso
        print(f'Época {epoch+1}/{epochs} - '
              f'Tempo: {epoch_time:.2f}s - '
              f'Perda (Treino): {train_loss:.6f} - '
              f'Perda (Valid): {valid_loss:.6f}')
    
    # Salva o modelo final
    torch.save(model.state_dict(), 'modelo_final.pth')
    print('Modelo final salvo!')
    
    return history

def plot_training_history(history):
    """
    Plota o histórico de treino.
    
    Args:
        history (dict): Dicionário com as perdas de treino e validação
    """
    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Treino')
    plt.plot(history['valid_loss'], label='Validação')
    plt.title('Histórico de treino')
    plt.xlabel('Época')
    plt.ylabel('Perda (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('historico_treino.png')
    plt.show()

# =========================================================================
# Função principal
# =========================================================================

def main():
    """Função principal que orquestra todo o fluxo de trabalho."""
    try:
        print("PyTorch - Modelo de Direção Autônoma")
        print("---------------------------------")
        
        # Carrega os dados
        print("Carregando dados de todas as sessões...")
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
            image_paths, steering_angles, test_size=0.2, random_state=42)
        
        print(f'Amostras de Treino: {len(X_train)}\nAmostras de Validação: {len(X_valid)}')
        
        # Cria os datasets
        print("Criando datasets...")
        train_dataset = DrivingDataset(X_train, y_train, transform=True)
        valid_dataset = DrivingDataset(X_valid, y_valid, transform=False)
        
        # Cria os dataloaders
        print("Criando dataloaders...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Testa um batch
        print("Testando batch...")
        sample_batch, sample_labels = next(iter(train_loader))
        print(f"Forma do batch: {sample_batch.shape}")
        print(f"Forma das labels: {sample_labels.shape}")
        
        # Cria o modelo
        print("Criando modelo NVIDIA...")
        model = NvidiaModel()
        print(model)
        
        # Define a função de perda e o otimizador
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Treina o modelo
        print("Iniciando treino...")
        history = train_model(model, train_loader, valid_loader, criterion, optimizer, EPOCHS)
        
        # Plota o histórico de treino
        print("Plotando histórico de treino...")
        plot_training_history(history)
        
        print("treino concluído com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

# Executa a função principal quando o script é executado diretamente
if __name__ == "__main__":
    main()
