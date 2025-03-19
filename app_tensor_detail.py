"""
Avaliação do Modelo de Direção Autônoma (TensorFlow/Keras)
---------------------------------------------------------
Este script carrega um modelo Keras treinado e realiza:
1. Avaliação sobre um conjunto de validação
2. Visualização de predições vs. valores reais
3. Análise de estatísticas de desempenho
4. Visualização de mapas de características
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import glob
import random
import time
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import shutil

# Forçar uso de CPU se necessário
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configurações
DATASET_DIR = 'dataset'  # Diretório principal com sessões de dados
MODEL_PATH = 'modelo_direcao_autonoma.keras'  # Caminho para o modelo treinado
TEST_SESSION = None  # Definir para usar uma sessão específica, ou None para usar todas
IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3
BATCH_SIZE = 32

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
# Funções para criação de lotes de dados
# =========================================================================

def create_test_generator(image_paths, steering_angles, batch_size=32):
    """
    Cria um gerador para avaliação do modelo.
    """
    num_samples = len(image_paths)
    
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_images = []
            batch_angles = []
            batch_paths = []
            
            end = min(offset + batch_size, num_samples)
            
            for i in range(offset, end):
                try:
                    # Carrega a imagem
                    img = mpimg.imread(image_paths[i])
                    
                    # Pré-processa a imagem
                    img = img_preprocess(img)
                    
                    batch_images.append(img)
                    batch_angles.append(steering_angles[i])
                    batch_paths.append(image_paths[i])
                    
                except Exception as e:
                    print(f"Erro ao carregar imagem {image_paths[i]}: {str(e)}")
            
            if batch_images:  # Verifica se o lote não está vazio
                yield np.array(batch_images), np.array(batch_angles), batch_paths

# =========================================================================
# Funções de avaliação do modelo
# =========================================================================

def evaluate_model(model, image_paths, steering_angles):
    """
    Avalia o modelo no conjunto de teste.
    
    Args:
        model: Modelo Keras carregado
        image_paths: Lista de caminhos das imagens
        steering_angles: Lista de ângulos de direção
        
    Returns:
        dict: Métricas de avaliação
        list: Predições
        list: Valores reais
        list: Caminhos das imagens
    """
    all_predictions = []
    all_targets = []
    all_paths = []
    
    # Cria o gerador
    test_gen = create_test_generator(image_paths, steering_angles, BATCH_SIZE)
    
    # Número de lotes
    n_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print("Avaliando modelo...")
    
    # Processa cada lote
    for i in tqdm(range(n_batches)):
        batch_images, batch_angles, batch_paths = next(test_gen)
        
        # Faz a predição
        predictions = model.predict(batch_images, verbose=0)
        
        # Armazena resultados
        all_predictions.extend(predictions.flatten())
        all_targets.extend(batch_angles)
        all_paths.extend(batch_paths)
    
    # Calcula as métricas
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    metrics = {
        'loss': mse,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics, all_predictions, all_targets, all_paths

def get_gradcam(model, img, layer_name):
    """
    Gera um mapa de calor Grad-CAM para visualizar onde o modelo está prestando atenção.
    
    Args:
        model: Modelo Keras
        img: Imagem de entrada
        layer_name: Nome da camada para extrair o mapa de características
        
    Returns:
        numpy.ndarray: Mapa de calor que pode ser sobreposto à imagem original
    """
    try:
        # Certifica-se de que a imagem está na forma correta
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        
        # Cria um modelo que retorna tanto a saída quanto as ativações da camada especificada
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Registra a operação de gradiente
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, 0]  # Ângulo de direção (valor único)
        
        # Gradiente da saída em relação aos mapas de características
        grads = tape.gradient(loss, conv_outputs)
        
        # Promedia os gradientes espacialmente
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiplica cada canal pelo seu 'peso de importância'
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normaliza o mapa de calor
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Redimensiona para o tamanho da imagem original
        heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
        
        # Converte para formato RGB para sobreposição
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap
    except Exception as e:
        print(f"Erro ao gerar Grad-CAM: {str(e)}")
        return None

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
        if col < len(axs[row]):
            fig.delaxes(axs[row][col])
    
    plt.tight_layout()
    plt.suptitle("Piores Predições (Maior Erro Absoluto)", y=1.02)
    
    return plt.gcf()

def visualize_feature_maps(model, image_path):
    """
    Visualiza os mapas de características do modelo para uma imagem específica usando Grad-CAM.
    
    Args:
        model: Modelo Keras
        image_path: Caminho para a imagem
    """
    try:
        # Carrega a imagem original
        original_img = mpimg.imread(image_path)
        
        # Pré-processa a imagem para o modelo
        processed_img = img_preprocess(original_img)
        
        # Escolhe as camadas convolucionais para visualizar
        conv_layers = []
        for i, layer in enumerate(model.layers):
            if 'conv' in layer.name:
                conv_layers.append(layer.name)
        
        # Configura a visualização
        n_layers = len(conv_layers)
        fig, axs = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))
        
        # Mostra a imagem original
        axs[0].imshow(original_img)
        axs[0].set_title('Imagem Original')
        axs[0].axis('off')
        
        # Gera e mostra Grad-CAM para cada camada convolucional
        for i, layer_name in enumerate(conv_layers):
            # Gera o mapa de calor
            heatmap = get_gradcam(model, np.expand_dims(processed_img, axis=0), layer_name)
            
            if heatmap is not None:
                # Converte a imagem original para BGR (para compatibilidade com cv2)
                original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                
                # Redimensiona o heatmap para o tamanho da imagem original
                heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                
                # Combina imagem original e heatmap
                superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_resized, 0.4, 0)
                
                # Converte de volta para RGB para matplotlib
                superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
                
                # Mostra a imagem com sobreposição
                axs[i + 1].imshow(superimposed_rgb)
                axs[i + 1].set_title(f'Camada: {layer_name}')
                axs[i + 1].axis('off')
            else:
                axs[i + 1].text(0.5, 0.5, f'Erro ao gerar\nGrad-CAM para\n{layer_name}', 
                               ha='center', va='center')
                axs[i + 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Erro ao visualizar mapas de características: {str(e)}")
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
        print("Avaliação do Modelo de Direção Autônoma (TensorFlow/Keras)")
        print("-------------------------------------------------------")
        
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
        
        print(f"Total de imagens carregadas: {len(image_paths)}")
        
        # Carrega o modelo
        print(f"Carregando modelo de {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        model.summary()
        
        # Avalia o modelo
        metrics, predictions, targets, paths = evaluate_model(model, image_paths, steering_angles)
        
        # Exibe as métricas
        print("\nMétricas de Avaliação:")
        print(f"Perda (MSE): {metrics['mse']:.6f}")
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
        
        # Mapas de Características (para uma imagem aleatória)
        random_idx = np.random.randint(0, len(paths))
        random_img_path = paths[random_idx]
        fig_features = visualize_feature_maps(model, random_img_path)
        if fig_features:
            features_path = os.path.join(output_dir, "feature_maps.png")
            fig_features.savefig(features_path)
        
        # Visualiza os resultados
        plt.show()
        
        # Gera o relatório HTML
        images = [
            ("Predições vs. Valores Reais", scatter_path),
            ("Distribuição do Erro", error_path),
            ("Predições ao Longo do Tempo", time_path),
            ("Piores Predições", worst_path)
        ]
        if fig_features:
            images.append(("Mapas de Características", features_path))
        
        report_path = generate_html_report(metrics, MODEL_PATH, images)
        print(f"\nRelatório HTML gerado em: {report_path}")
        
        print("\nAvaliação concluída com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
