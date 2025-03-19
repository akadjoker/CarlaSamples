import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
 

def find_all_sessions(base_dir="dataset"):
    # Procuramos por pastas que começam com "session_" dentro da pasta dataset
    session_dirs = glob.glob(os.path.join(base_dir, "session_*"))
    
    valid_sessions = []
    for session_dir in session_dirs:
        # Verificamos se existe o arquivo CSV de dataset e a pasta de imagens
        csv_path = os.path.join(session_dir, "steering_data.csv")
        images_dir = os.path.join(session_dir, "images")
        
        if os.path.exists(csv_path) and os.path.isdir(images_dir):
            valid_sessions.append(session_dir)
    
    print(f"Encontradas {len(valid_sessions)} sessões válidas com datasets:")
    for session in valid_sessions:
        print(f" - {os.path.basename(session)}")
    
    return valid_sessions

def load_all_sessions_data(session_dirs):
    combined_data = pd.DataFrame()
    
    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        csv_path = os.path.join(session_dir, "steering_data.csv")
        
        try:

            session_data = pd.read_csv(csv_path)
            

            required_cols = ['image_path', 'steering']
            if not all(col in session_data.columns for col in required_cols):
                print(f"Aviso: Sessão {session_name} não tem as colunas necessárias, saltamos...")
                continue
            

            session_data['session'] = session_name
            

            session_data['full_image_path'] = session_data['image_path'].apply(
                lambda x: os.path.join(session_dir, x if not x.startswith('/') else x[1:])
            )
            

            valid_rows = []
            for idx, row in session_data.iterrows():
                if os.path.exists(row['full_image_path']):
                    valid_rows.append(True)
                else:
                    print(f"Imagem não encontrada: {row['full_image_path']}")
                    valid_rows.append(False)
            
            session_data = session_data[valid_rows]
            

            combined_data = pd.concat([combined_data, session_data], ignore_index=True)
            
            print(f"Sessão {session_name}: {len(session_data)} amostras válidas")
            
        except Exception as e:
            print(f"Erro ao carregar sessão {session_name}: {e}")
    
    print(f"Total de amostras combinadas: {len(combined_data)}")
    return combined_data



def preprocess_image(img_path):
    """
    Pré-processa a imagem aplicando corte ANTES do resize
    """
    img = cv2.imread(img_path)
    if img is None:
        raise Exception(f"Falha ao carregar imagem: {img_path}")
    return img


def preProcess(img):
    img = img[120:480,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    return img


 
 

def augmentImage(img, steering):

    if np.random.rand() < 0.5:#pan
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:#zoom
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:#brightness
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:#flip
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering

def data_generator(data, batch_size=32, target_size=(200, 66), augment=True):

    num_samples = len(data)
    while True:
        # Embaralhar os dados a cada época
        shuffled_data = data.sample(frac=1)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = shuffled_data.iloc[offset:offset+batch_size]
            
            images = []
            steerings = []
            
            for _, row in batch_samples.iterrows():
                # Carregar e pré-processar imagem
                img = preprocess_image(row['full_image_path'], target_size)
                steering = row['steering']
                
                # Aumento de dados 
                if augment:
                   img, steering = augmentImage(img, steering)

                    
                
                img = preProcess(img)
                images.append(img)
                steerings.append(steering)
            
            X = np.array(images)
            y = np.array(steerings)
            
            yield X, y








def create_model(input_shape=(66, 200, 3)):
    initializer = tf.keras.initializers.HeNormal()
    
    model = Sequential([
        # Normalização de entrada
        tf.keras.layers.Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape),
        

        
        # Camadas convolucionais com BatchNormalization
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        
        Conv2D(64, (3, 3), activation='elu', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        
        Conv2D(64, (3, 3), activation='elu', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        
        # Flatten
        Flatten(),
        
        # Camadas densas com mais Dropout
        Dense(100, activation='elu', kernel_initializer=initializer),
        Dropout(0.5),
        
        Dense(50, activation='elu', kernel_initializer=initializer),
        Dropout(0.5),
        
        Dense(10, activation='elu', kernel_initializer=initializer),
        Dropout(0.3),
        
        # Saída: ângulo de direção
        Dense(1)
    ])
    
    # Compilar com otimizador mais avançado e taxa de aprendizado adaptativa
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )
    
    # Compilar com loss function mais específica
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Podemos tambem testar: tf.keras.losses.Huber(delta=1.0)
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    model.summary()
    return model

def visualize_sample_images(data, num_samples=5):
    """
    Visualiza algumas imagens de amostra do dataset
    """
    if len(data) < num_samples:
        num_samples = len(data)
        
    plt.figure(figsize=(15, 10))
    samples = data.sample(n=num_samples)
    
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = row['full_image_path']
        steering = row['steering']
        session = row['session']
        
        # Carregar imagem
        img = cv2.imread(img_path)
        if img is None:
            print(f"Aviso: Não é possível carregar a imagem {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"Steering: {steering:.2f}\nSession: {session}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()




def balance_data(data,display=True):
    nBin = 31
    samplesPerBin =  300
    hist, bins = np.histogram(data['steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    if display:
        hist, _ = np.histogram(data['steering'], (nBin))
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.savefig('balance.png')    
        #plt.show()
    return data

def train_model():
    # Encontrar todas as sessões disponíveis
    sessions = find_all_sessions()
    
    if not sessions:
        print("Nenhuma sessão válida encontrada. Verifica o diretório dataset.")
        return
    
    # Carregar dados de todas as sessões
    combined_data = load_all_sessions_data(sessions)
    
    if len(combined_data) < 10:
        print("Dados insuficientes para treinar. É necessário pelo menos 10 amostras.")
        return
    
    combined_data = balance_data(combined_data)
    
    # Visualizar algumas imagens de amostra
    visualize_sample_images(combined_data)
    
    # Analisar a distribuição dos ângulos de direção
    plt.figure(figsize=(10, 6))
    plt.hist(combined_data['steering'], bins=50)
    plt.title('Distribuição dos Ângulos de Direção')
    plt.xlabel('Ângulo de Direção')
    plt.ylabel('Frequência')
    plt.savefig('steering_distribution.png')
    plt.show()
    
    # Dividir em treino/validação
    train_data, val_data = train_test_split(combined_data, test_size=0.2, random_state=42)
    print(f"Dados de treino: {len(train_data)}, Dados de validação: {len(val_data)}")
    
    # Parâmetros de treino
    batch_size = 32
    epochs = 20
    target_size = (66, 200)  # altura, largura
    
    # Criar geradores
    train_generator = data_generator(train_data, batch_size, target_size, augment=True)
    val_generator = data_generator(val_data, batch_size, target_size, augment=False)
    
    # Criar modelo
    model = create_model(input_shape=target_size + (3,))
    
    # Callbacks
    callbacks = [
        # Salvar o melhor modelo
        tf.keras.callbacks.ModelCheckpoint(
            'best_steering_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Parar se não houver melhoria
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # Reduzir a taxa de aprendizado quando o treino estagna
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Treinar o modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_data) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_data) // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Salvar o modelo final
    model.save('steering_model_final.keras')
    
    # Plotar história do treino
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    #plt.show()
    
    print("Treino concluído! Modelos guardados como 'best_steering_model.keras' e 'steering_model_final.keras'")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()
