 

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import time
import shutil
import seaborn as sns
from scipy.stats import norm

from imgaug import augmenters as iaa

# Configurações
DATASET_DIR = 'dataset'  # Diretório principal com sessões de dados
MODEL_PATH = 'modelo_final.pth'  # Caminho para o modelo treinado
TEST_SESSION = None  # Definir para usar uma sessão específica, ou None para usar todas
IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3
BATCH_SIZE = 32

# Verifica se GPU está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# =========================================================================
# Definição do Modelo NVIDIA
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
    
    def get_activation_maps(self, x):
        """
        Obtém os mapas de ativação do modelo para visualização.
        
        Args:
            x (torch.Tensor): Tensor de entrada com shape (batch_size, 3, 66, 200)
            
        Returns:
            dict: Dicionário com os mapas de ativação de cada camada
        """
        activations = {}
        
        # Camadas convolucionais
        x1 = F.elu(self.conv1(x))
        activations['conv1'] = x1
        
        x2 = F.elu(self.conv2(x1))
        activations['conv2'] = x2
        
        x3 = F.elu(self.conv3(x2))
        activations['conv3'] = x3
        
        x4 = F.elu(self.conv4(x3))
        activations['conv4'] = x4
        
        x5 = F.elu(self.conv5(x4))
        activations['conv5'] = x5
        
        return activations

# =========================================================================
# Funções de pré-processamento de imagens
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
# Dataset do PyTorch
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
            
            # Pré-processa a imagem
            img = img_preprocess(img)
            
            # Converte para tensor do PyTorch (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            img = torch.FloatTensor(img)
            
            return img, torch.FloatTensor([steering_angle]), img_path
        
        except Exception as e:
            print(f"Erro ao carregar imagem {img_path}: {str(e)}")
            # Em caso de erro, retorna uma imagem preta e ângulo zero
            img = np.zeros((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
            return torch.FloatTensor(img), torch.FloatTensor([0.0]), img_path

# =========================================================================
# Funções de carregamento de dados
# =========================================================================

def load_test_data():
    """
    Carrega os dados para teste/validação.
    Se TEST_SESSION for definido, carrega apenas essa sessão.
    Caso contrário, carrega todas as sessões.
    """
    all_image_paths = []
    all_steering_angles = []
    
    # Determina quais sessões carregar
    if TEST_SESSION:
        session_dirs = [os.path.join(DATASET_DIR, TEST_SESSION)]
    else:
        session_dirs = glob.glob(os.path.join(DATASET_DIR, 'session_*'))
    
    if not session_dirs:
        raise ValueError(f"Nenhuma pasta de sessão encontrada")
    
    print(f"Carregando dados de {len(session_dirs)} sessões")
    
    # Processa cada sessão
    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        csv_path = os.path.join(session_dir, 'steering_data.csv')
        
        if not os.path.exists(csv_path):
            print(f"Aviso: Arquivo CSV não encontrado em {session_dir}, pulando sessão")
            continue
        
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
    
    return all_image_paths, all_steering_angles

# =========================================================================
# Funções de avaliação do modelo
# =========================================================================

def evaluate_model(model, test_loader):
    """
    Avalia o modelo no conjunto de teste.
    
    Args:
        model: Modelo PyTorch
        test_loader: DataLoader para o conjunto de teste
        
    Returns:
        dict: Métricas de avaliação
        list: Predições
        list: Valores reais
        list: Caminhos das imagens
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_paths = []
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data, target, paths in tqdm(test_loader, desc="Avaliando modelo"):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calcula a perda
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            # Armazena as predições e alvos
            all_predictions.extend(output.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
            all_paths.extend(paths)
    
    # Calcula as métricas
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    metrics = {
        'loss': total_loss / len(test_loader.dataset),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics, all_predictions, all_targets, all_paths

# =========================================================================
# Funções de visualização
# =========================================================================

def plot_prediction_vs_target(predictions, targets, title="Predições vs. Valores Reais"):
    """
    Plota gráfico de dispersão das predições vs. valores reais.
    
    Args:
        predictions: Lista de predições do modelo
        targets: Lista de valores reais
        title: Título do gráfico
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5, color='blue')
    
    # Adiciona linha ideal (y=x)
    min_val = min(min(predictions), min(targets))
    max_val = max(max(predictions), max(targets))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(title)
    plt.xlabel('Ângulos Reais')
    plt.ylabel('Ângulos Preditos')
    plt.grid(True, alpha=0.3)
    
    # Calcula correlação
    correlation = np.corrcoef(targets, predictions)[0, 1]
    plt.figtext(0.15, 0.85, f'Correlação: {correlation:.4f}', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    return plt.gcf()

def plot_error_distribution(predictions, targets, title="Distribuição do Erro de Predição"):
    """
    Plota histograma dos erros de predição.
    
    Args:
        predictions: Lista de predições do modelo
        targets: Lista de valores reais
        title: Título do gráfico
    """
    errors = np.array(predictions) - np.array(targets)
    
    plt.figure(figsize=(10, 6))
    
    # Histograma dos erros
    sns.histplot(errors, kde=True, bins=50, color='blue', alpha=0.6)
    
    # Adiciona curva normal teórica
    mu, sigma = norm.fit(errors)
    x = np.linspace(min(errors), max(errors), 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p * len(errors) * (max(errors) - min(errors)) / 50, 'r-', linewidth=2)
    
    plt.title(title)
    plt.xlabel('Erro (Predito - Real)')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)
    
    # Adiciona estatísticas
    stats_text = (
        f'Média do erro: {np.mean(errors):.6f}\n'
        f'Desvio padrão: {np.std(errors):.6f}\n'
        f'Erro máximo: {np.max(np.abs(errors)):.6f}'
    )
    plt.figtext(0.15, 0.85, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    return plt.gcf()

def plot_predictions_over_time(predictions, targets, title="Predições e Valores Reais ao Longo do Tempo"):
    """
    Plota os ângulos de direção preditos e reais em função do tempo (índice da amostra).
    
    Args:
        predictions: Lista de predições do modelo
        targets: Lista de valores reais
        title: Título do gráfico
    """
    plt.figure(figsize=(15, 6))
    
    # Amostras
    samples = np.arange(len(predictions))
    
    # Plota valores reais
    plt.plot(samples, targets, 'b-', alpha=0.5, label='Ângulos Reais')
    
    # Plota predições
    plt.plot(samples, predictions, 'r-', alpha=0.5, label='Ângulos Preditos')
    
    plt.title(title)
    plt.xlabel('Índice da Amostra')
    plt.ylabel('Ângulo de Direção')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Se houver muitas amostras, limita a visualização a uma parte
    if len(samples) > 500:
        plt.xlim(0, 500)
        plt.figtext(0.15, 0.85, f'Mostrando primeiras 500 de {len(samples)} amostras', 
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    return plt.gcf()

def visualize_worst_predictions(predictions, targets, image_paths, n=10):
    """
    Visualiza as n piores predições (maior erro absoluto).
    
    Args:
        predictions: Lista de predições do modelo
        targets: Lista de valores reais
        image_paths: Lista de caminhos das imagens
        n: Número de imagens para visualizar
    """
    # Calcula o erro absoluto
    abs_errors = np.abs(np.array(predictions) - np.array(targets))
    
    # Encontra os índices das n piores predições
    worst_indices = np.argsort(abs_errors)[-n:][::-1]
    
    # Plota as imagens
    rows = (n + 4) // 5  # Calcula o número de linhas necessárias
    cols = min(n, 5)     # No máximo 5 colunas
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axs = [axs]  # Garante que axs seja um array 2D
    
    for i, idx in enumerate(worst_indices):
        row, col = i // cols, i % cols
        try:
            # Carrega e mostra a imagem
            img = mpimg.imread(image_paths[idx])
            axs[row][col].imshow(img)
            axs[row][col].set_title(f'Pred: {predictions[idx]:.4f}\nReal: {targets[idx]:.4f}\nErro: {abs_errors[idx]:.4f}')
            axs[row][col].axis('off')
        except Exception as e:
            print(f"Erro ao visualizar imagem {image_paths[idx]}: {str(e)}")
    
    # Remove subplots vazios
    for i in range(len(worst_indices), rows*cols):
        row, col = i // cols, i % cols
        fig.delaxes(axs[row][col])
    
    plt.tight_layout()
    plt.suptitle("Piores Predições (Maior Erro Absoluto)", y=1.02)
    
    return plt.gcf()

def visualize_activation_maps(model, image_path):
    """
    Visualiza os mapas de ativação do modelo para uma imagem específica.
    
    Args:
        model: Modelo PyTorch
        image_path: Caminho para a imagem
        
    Returns:
        fig: Figura matplotlib
    """
    try:
        # Carrega a imagem
        img = mpimg.imread(image_path)
        
        # Pré-processa a imagem
        processed_img = img_preprocess(img)
        
        # Converte para tensor
        img_tensor = torch.FloatTensor(np.transpose(processed_img, (2, 0, 1))).unsqueeze(0).to(device)
        
        # Obtém os mapas de ativação
        with torch.no_grad():
            activations = model.get_activation_maps(img_tensor)
        
        # Configura a visualização
        fig, axs = plt.subplots(1, 6, figsize=(15, 4))
        
        # Imagem original
        axs[0].imshow(img)
        axs[0].set_title('Original')
        axs[0].axis('off')
        
        # Mapas de ativação das camadas convolucionais
        for i, layer_name in enumerate(['conv1', 'conv2', 'conv3', 'conv4', 'conv5']):
            if layer_name in activations:
                # Pega o primeiro canal do primeiro batch
                act = activations[layer_name][0, 0, :, :].cpu().numpy()
                axs[i+1].imshow(act, cmap='viridis')
                axs[i+1].set_title(f'{layer_name}')
                axs[i+1].axis('off')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Erro ao visualizar mapas de ativação: {str(e)}")
        return None

def generate_html_report(metrics, model_path, images=None):
    """
    Gera um relatório HTML com os resultados da avaliação.
    
    Args:
        metrics: Dicionário com métricas de avaliação
        model_path: Caminho para o modelo avaliado
        images: Lista de tuplas (nome da imagem, caminho da imagem)
    
    Returns:
        str: Caminho para o relatório HTML
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = f"evaluation_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # Copia as imagens para a pasta do relatório
    img_paths = []
    if images:
        for img_name, img_path in images:
            if os.path.exists(img_path):
                new_path = os.path.join(report_dir, f"{img_name}.png")
                shutil.copy2(img_path, new_path)
                img_paths.append((img_name, os.path.basename(new_path)))
    
    # Gera o conteúdo HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Relatório de Avaliação do Modelo - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
            .metric-card {{ 
                background-color: #f9f9f9; 
                border-radius: 8px; 
                padding: 15px; 
                margin: 10px; 
                width: 200px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-name {{ font-weight: bold; margin-bottom: 5px; }}
            .metric-value {{ font-size: 24px; color: #007bff; }}
            img {{ max-width: 100%; margin: 10px 0; border-radius: 8px; }}
            .image-container {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Relatório de Avaliação do Modelo de Direção Autônoma</h1>
        <p>Data: {time.strftime("%d/%m/%Y %H:%M:%S")}</p>
        <p>Modelo: {model_path}</p>
        
        <h2>Métricas de Desempenho</h2>
        <div class="metrics">
    """
    
    # Adiciona as métricas
    for name, value in metrics.items():
        html_content += f"""
            <div class="metric-card">
                <div class="metric-name">{name.upper()}</div>
                <div class="metric-value">{value:.6f}</div>
            </div>
        """
    
    html_content += """
        </div>
        
        <h2>Visualizações</h2>
    """
    
    # Adiciona as imagens
    for img_name, img_file in img_paths:
        html_content += f"""
        <div class="image-container">
            <h3>{img_name}</h3>
            <img src="{img_file}" alt="{img_name}" />
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Salva o relatório HTML
    report_path = os.path.join(report_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    return report_path

# =========================================================================
# Função principal
# =========================================================================

def main():
    try:
        print("Avaliação do Modelo de Direção Autônoma (PyTorch)")
        print("-----------------------------------------------")
        
        # Verifica se o modelo existe
        if not os.path.exists(MODEL_PATH):
            print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
            return
        
        # Carrega os dados de teste
        print("Carregando dados de teste...")
        image_paths, steering_angles = load_test_data()
        
        if len(image_paths) == 0:
            print("Nenhum dado de teste encontrado. Verifique a estrutura das pastas.")
            return
        
        # Cria o dataset e dataloader
        print(f"Criando dataset com {len(image_paths)} imagens...")
        test_dataset = DrivingDataset(image_paths, steering_angles, transform=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Carrega o modelo
        print(f"Carregando modelo de {MODEL_PATH}...")
        model = NvidiaModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        
        # Avalia o modelo
        print("Avaliando modelo...")
        metrics, predictions, targets, paths = evaluate_model(model, test_loader)
        
        # Exibe as métricas
        print("\nMétricas de Avaliação:")
        print(f"Perda (MSE): {metrics['loss']:.6f}")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"R²: {metrics['r2']:.6f}")
        
        # Cria pasta para salvar as visualizações
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"model_evaluation_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Gera e salva as visualizações
        print("\nGerando visualizações...")
        
        # Predições vs. Valores Reais
        fig_scatter = plot_prediction_vs_target(predictions, targets)
        scatter_path = os.path.join(output_dir, "predictions_vs_targets.png")
        fig_scatter.savefig(scatter_path)
        
        # Distribuição do Erro
        fig_error = plot_error_distribution(predictions, targets)
        error_path = os.path.join(output_dir, "error_distribution.png")
        fig_error.savefig(error_path)
        
        # Predições ao Longo do Tempo
        fig_time = plot_predictions_over_time(predictions, targets)
        time_path = os.path.join(output_dir, "predictions_over_time.png")
        fig_time.savefig(time_path)
        
        # Piores Predições
        fig_worst = visualize_worst_predictions(predictions, targets, paths)
        worst_path = os.path.join(output_dir, "worst_predictions.png")
        fig_worst.savefig(worst_path)
        
        # Mapas de Ativação (para uma imagem aleatória)
        random_idx = np.random.randint(0, len(paths))
        random_img_path = paths[random_idx]
        fig_act = visualize_activation_maps(model, random_img_path)
        if fig_act:
            act_path = os.path.join(output_dir, "activation_maps.png")
            fig_act.savefig(act_path)
        
        # Visualiza os resultados
        plt.show()
        
        # Gera o relatório HTML
        images = [
            ("Predições vs. Valores Reais", scatter_path),
            ("Distribuição do Erro", error_path),
            ("Predições ao Longo do Tempo", time_path),
            ("Piores Predições", worst_path)
        ]
        if fig_act:
            images.append(("Mapas de Ativação", act_path))
        
        report_path = generate_html_report(metrics, MODEL_PATH, images)
        print(f"\nRelatório HTML gerado em: {report_path}")
        
        print("\nAvaliação concluída com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
