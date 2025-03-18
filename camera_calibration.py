import numpy as np
import cv2
import glob
import pickle

def calibrate_camera():
    """
    Calibra a câmera usando um conjunto de imagens de calibração
    Retorna a matriz da câmera e os coeficientes de distorção
    """
    # Arrays para armazenar pontos de objeto e pontos de imagem de todas as imagens
    objpoints = [] # pontos 3D no espaço real
    imgpoints = [] # pontos 2D no plano da imagem

    # Prepara os pontos do objeto (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordenadas

    # Ler imagens de calibração
    images = glob.glob('camera_cal/calibration*.jpg')

    # Percorrer todas as imagens de calibração
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Encontrar os cantos do tabuleiro de xadrez
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # Se encontrados, adicionar pontos de objeto e pontos de imagem
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    if len(imgpoints) > 0:
        # Calibrar a câmera
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        return mtx, dist
    else:
        raise Exception("Não foi possível calibrar a câmera. Certifique-se de que as imagens de calibração estão no diretório 'camera_cal'.")

def undistort(img, mtx, dist):
    """
    Corrige a distorção da imagem usando a matriz da câmera e coeficientes de distorção
    """
    return cv2.undistort(img, mtx, dist, None, mtx)

def load_or_calibrate_camera():
    """
    Tenta carregar a calibração salva ou calibra a câmera novamente
    """
    try:
        with open('camera_cal/calibration.p', 'rb') as f:
            calibration = pickle.load(f)
            mtx = calibration['mtx']
            dist = calibration['dist']
            print("Calibração da câmera carregada com sucesso.")
            return mtx, dist
    except:
        try:
            print("Calibrando câmera...")
            mtx, dist = calibrate_camera()
            
            # Salvar calibração para uso futuro
            calibration = {'mtx': mtx, 'dist': dist}
            with open('camera_cal/calibration.p', 'wb') as f:
                pickle.dump(calibration, f)
            print("Calibração da câmera concluída e salva.")
            return mtx, dist
        except Exception as e:
            print(f"Erro ao calibrar a câmera: {e}")
            raise
