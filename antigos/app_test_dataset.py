import os
import numpy as np
import cv2
import glob
import random
import pandas as pd

DATASET_DIR = 'dataset' 
 
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

def main():
    """Função principal que orquestra todo o fluxo de trabalho."""
    try:
        print("Carregando dados de todas as sessões...")
        image_paths, steering_angles = load_sessions_data()
        
        if len(image_paths) == 0:
            print("Nenhum dado foi carregado. Verifique a estrutura das pastas e arquivos.")
            return
        
        print(f"Total de imagens carregadas: {len(image_paths)}")
        
   
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")


if __name__ == "__main__":
    main()

