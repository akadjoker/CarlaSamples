import os
import numpy as np
import cv2
import glob
import pandas as pd
import datetime
import shutil

DATASET_DIR = 'dataset'

def load_sessions_data():
 
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
            print(f"Aviso: Arquivo CSV não encontrado em {session_dir}, skeep sessão")
            continue
        
        print(f"Carregando dados da sessão: {session_name}")
        
        # Carrega o CSV
        try:
            df = pd.read_csv(csv_path)
            
            # Verifica se o CSV tem as colunas esperadas
            if 'image_path' not in df.columns or 'steering' not in df.columns:
                print(f"Aviso: Formato de CSV inválido em {session_name}, skeep sessão")
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

def create_augmented_session():
 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_session_dir = f"{DATASET_DIR}/session_{timestamp}_augmented"
    new_images_dir = f"{new_session_dir}/images"
    
    os.makedirs(new_images_dir, exist_ok=True)
    
 
    new_csv_path = f"{new_session_dir}/steering_data.csv"
    new_csv = open(new_csv_path, "w")
    new_csv.write("image_path,steering\n")
    
 
    image_paths, steering_angles = load_sessions_data()
    
    if len(image_paths) == 0:
        print("Nenhum dado foi carregado. Verifica a estrutura das pastas e arquivos.")
        return
    
    print(f"Total de imagens originais: {len(image_paths)}")
    print(f"Nova sessão com imagens espelhadas: {new_session_dir}")
    
    # Processa cada imagem
    processed_count = 0
    for img_path, steering in zip(image_paths, steering_angles):
        try:
            # Lê a imagem
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erro ao ler imagem: {img_path}")
                continue
            
            # Espelha horizontalmente
            flipped_img = cv2.flip(img, 1)  # 1 para espelhar horizontalmente
            
            # Inverte o valor do steering
            inverted_steering = -steering
            
            # Salva a nova imagem
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            new_img_filename = f"flipped_frame_{timestamp}.jpg"
            new_img_path = os.path.join(new_images_dir, new_img_filename)
            
            cv2.imwrite(new_img_path, flipped_img)
            
            # Escreve no CSV
            new_csv.write(f"images/{new_img_filename},{inverted_steering:.6f}\n")
            new_csv.flush()
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processadas {processed_count} imagens", end="\r")
                
        except Exception as e:
            print(f"Erro ao processar imagem {img_path}: {str(e)}")
    
    new_csv.close()
    print(f"\nProcesso concluído! {processed_count} imagens geradas e salvas em {new_session_dir}")
    return new_session_dir

def main():
 
    try:
        print("Verificando diretório de dataset...")
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
            print(f"Diretório {DATASET_DIR} criado!")
        
        # Cria a sessão aumentada
        augmented_session = create_augmented_session()
        
        if augmented_session:
            print(f"Sessão aumentada criada com sucesso: {augmented_session}")
            print("Esta sessão contém as imagens espelhadas horizontalmente com valores de steering invertidos.")
 
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")

if __name__ == "__main__":
    main()
