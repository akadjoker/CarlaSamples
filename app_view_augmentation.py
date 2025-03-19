 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import random
import pandas as pd
from imgaug import augmenters as iaa

# Configuração
DATASET_DIR = 'dataset'  
NUM_SAMPLES = 8  # Número de amostras para visualizar

# =========================================================================
# Funções de carregamento de dados
# =========================================================================

def load_random_images(num_samples=8):
    """
    Carrega imagens aleatórias de todas as sessões disponíveis.
    Retorna uma lista de tuplas (imagem, caminho, ângulo de direção).
    """
    all_images = []
    
    # Encontra todas as pastas de sessão
    session_dirs = glob.glob(os.path.join(DATASET_DIR, 'session_*'))
    
    if not session_dirs:
        raise ValueError(f"Nenhuma pasta de sessão encontrada em {DATASET_DIR}")
    
    print(f"Encontradas {len(session_dirs)} sessões de dados")
    
    # Coleta possíveis imagens de todas as sessões
    image_candidates = []
    
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
                # Tenta dois possíveis caminhos para a imagem
                image_path = os.path.join(session_dir, row['image_path'])
                alt_image_path = os.path.join(session_dir, 'images', os.path.basename(row['image_path']))
                
                # Adiciona à lista de candidatos se a imagem existir
                if os.path.exists(image_path):
                    image_candidates.append((image_path, float(row['steering'])))
                elif os.path.exists(alt_image_path):
                    image_candidates.append((alt_image_path, float(row['steering'])))
                
        except Exception as e:
            print(f"Erro ao processar sessão {session_name}: {str(e)}")
    
    # Verifica se temos imagens candidatas suficientes
    if len(image_candidates) < num_samples:
        print(f"Aviso: Apenas {len(image_candidates)} imagens encontradas, menor que {num_samples} solicitadas")
        num_samples = len(image_candidates)
    
    # Seleciona amostras aleatórias
    selected_candidates = random.sample(image_candidates, num_samples)
    
    # Carrega as imagens selecionadas
    for image_path, steering in selected_candidates:
        try:
            image = mpimg.imread(image_path)
            all_images.append((image, image_path, steering))
            print(f"Carregada imagem: {os.path.basename(image_path)}, ângulo: {steering:.6f}")
        except Exception as e:
            print(f"Erro ao carregar imagem {image_path}: {str(e)}")
    
    return all_images

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

def apply_all_augmentations(image, steering_angle):
    """
    Aplica sequencialmente todas as técnicas de aumento de dados para demonstração.
    Retorna um dicionário com todas as versões da imagem.
    """
    result = {
        "Original": (image.copy(), steering_angle)
    }
    
    # Zoom
    zoomed_image = zoom(image.copy())
    result["Zoom"] = (zoomed_image, steering_angle)
    
    # Pan
    panned_image = pan(image.copy())
    result["Pan"] = (panned_image, steering_angle)
    
    # Brilho
    bright_image = img_random_brightness(image.copy())
    result["Brilho"] = (bright_image, steering_angle)
    
    # Flip
    flipped_image, flipped_angle = img_random_flip(image.copy(), steering_angle)
    result["Flip"] = (flipped_image, flipped_angle)
    
    # Combinado: zoom + brilho
    combined1 = img_random_brightness(zoom(image.copy()))
    result["Zoom+Brilho"] = (combined1, steering_angle)
    
    # Combinado: pan + flip
    combined2_img, combined2_angle = img_random_flip(pan(image.copy()), steering_angle)
    result["Pan+Flip"] = (combined2_img, combined2_angle)
    
    # Combinado: todas as técnicas
    temp_img = zoom(image.copy())
    temp_img = pan(temp_img)
    temp_img = img_random_brightness(temp_img)
    combined_all_img, combined_all_angle = img_random_flip(temp_img, steering_angle)
    result["Todas"] = (combined_all_img, combined_all_angle)
    
    return result

# =========================================================================
# Funções de visualização
# =========================================================================

def visualize_augmentations(images_data):
    """
    Visualiza as imagens originais e suas versões aumentadas.
    images_data: lista de tuplas (imagem, caminho, ângulo)
    """
    num_samples = len(images_data)
    
    for i, (image, image_path, steering) in enumerate(images_data):
        # Aplica todas as técnicas de aumento
        aug_images = apply_all_augmentations(image, steering)
        
        # Configura o subplot
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Aumento de Dados - Amostra {i+1}', fontsize=16)
        fig.tight_layout(pad=3.0)
        
        # Mostra cada versão da imagem
        techniques = list(aug_images.keys())
        for j, technique in enumerate(techniques):
            img, angle = aug_images[technique]
            row, col = j // 4, j % 4
            axs[row, col].imshow(img)
            axs[row, col].set_title(f'{technique}\nÂngulo: {angle:.4f}')
            axs[row, col].axis('off')
        
        # Adiciona informações do arquivo
        plt.figtext(0.5, 0.01, f'Arquivo: {os.path.basename(image_path)}', 
                  ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()

# =========================================================================
# Função principal
# =========================================================================

def main():
    try:
        print("Demo de Técnicas de Aumento de Dados")
        print("------------------------------------")
        
        # Carrega imagens aleatórias
        images_data = load_random_images(NUM_SAMPLES)
        
        if not images_data:
            print("Nenhuma imagem encontrada para demonstração.")
            return
            
        # Visualiza as técnicas de aumento
        print(f"\nVisualizando {len(images_data)} amostras com diferentes técnicas de aumento...")
        visualize_augmentations(images_data)
        
        print("\nDemo concluída com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a demonstração: {str(e)}")

# Executa a função principal quando o script é executado diretamente
if __name__ == "__main__":
    main()
