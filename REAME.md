# Descrição dos Programas CARLA

## 1. app_camera.py
Este programa cria uma simulação utilizando o ambiente CARLA para veículos autónomos. Configura uma cena com um Tesla Model 3 que pode ser controlado pelo utilizador através do teclado. O programa apresenta duas vistas de câmara simultaneamente utilizando Pygame: uma câmara principal em terceira pessoa e uma câmara frontal para deteção de faixas de rodagem. Os controlos incluem aceleração, travagem e direção, com informações sobre velocidade e FPS exibidas no ecrã. O programa carrega um mapa otimizado (Town04_Opt ou similar) e configura o ambiente para melhor desempenho.

## 2. app_datacollect.py
Este programa é focado na recolha de dados para treino de modelos de condução autónoma no CARLA. Permite gravar frames da câmara e os valores correspondentes de direção (steering) num conjunto de dados estruturado. Os dados são organizados em sessões com timestamps e guardados em ficheiros CSV. O programa oferece controlo manual do veículo e visualização em tempo real através do OpenCV. Inclui funcionalidades como ativação/desativação da recolha de dados, piloto automático opcional e visualização de informações de desempenho.

## 3. app_simples.py
Este programa implementa uma versão mais simples de simulação CARLA com foco na deteção de faixas de rodagem. Utiliza OpenCV para processar as imagens da câmara e detetar linhas de estrada usando técnicas como transformação para escala de cinzentos, limiarização, deteção de contornos (Canny) e transformada de Hough para linhas. O programa inclui uma implementação de deteção de faixas usando o método de janela deslizante (sliding window). A interface permite controlo manual do veículo e visualização em tempo real das linhas detetadas.

## 4. app_simples_slide.py
Esta é uma versão estendida do app_simples.py, adicionando controlos deslizantes (trackbars) interativos para ajustar parâmetros de deteção de linhas em tempo real. O utilizador pode modificar valores como limiar de binarização, parâmetros de deteção de contornos Canny, configurações da transformada de Hough e inclinação máxima de linhas. O programa exibe várias visualizações simultaneamente, incluindo a imagem original, o resultado da deteção, os contornos e as linhas identificadas, facilitando a calibração dos algoritmos de deteção.

## 5. app_view_augmentation.py
Este programa não faz parte da simulação CARLA, mas é uma ferramenta para visualizar técnicas de aumento de dados (data augmentation) aplicadas a imagens recolhidas em sessões anteriores. Carrega imagens aleatórias do conjunto de dados gerado pelo app_datacollect.py e aplica várias transformações como zoom, pan (deslocamento), alteração de brilho e espelhamento horizontal. O programa exibe as imagens originais e aumentadas lado a lado utilizando Matplotlib, mostrando como estas técnicas podem ser usadas para expandir conjuntos de dados para treino de redes neuronais para condução autónoma.


## 6. app_training_torch.py
Este programa implementa o treino de um modelo de condução autónoma utilizando PyTorch. O script carrega os dados recolhidos pelo app_datacollect.py, aplicando técnicas de balanceamento para evitar enviesamentos causados por ângulos de direção sobre-representados. Utiliza a arquitetura NVIDIA para condução autónoma e incorpora técnicas de aumento de dados (data augmentation) como zoom, pan, alteração de brilho e espelhamento para melhorar a generalização do modelo. O treino é dividido em conjuntos de treino e validação, com avaliação em tempo real do desempenho do modelo. O programa inclui visualizações para histogramas de distribuição de ângulos de direção e permite guardar o modelo treinado para utilização posterior.

## 7. app_torch_detail.py
Esta aplicação foca-se na avaliação detalhada de um modelo PyTorch previamente treinado. Fornece análises estatísticas completas sobre o desempenho do modelo, incluindo métricas como erro quadrático médio (MSE), erro absoluto médio (MAE) e coeficiente de determinação (R²). O programa gera visualizações avançadas como gráficos de dispersão de predições versus valores reais, distribuição de erros, comportamento do modelo ao longo do tempo e visualizações dos piores casos de predição. Adicionalmente, inclui a capacidade de visualizar mapas de ativação de características das camadas convolucionais, permitindo entender quais partes da imagem influenciam mais as decisões do modelo. Gera um relatório HTML completo com todas as análises e visualizações.

## 8. app_test_dataset.py
Este é um script simples para validar a integridade e disponibilidade dos dados de treino. O programa percorre todas as sessões de recolha de dados armazenadas na pasta "dataset", verifica a existência e validade dos ficheiros CSV com os ângulos de direção e confirma que as imagens referenciadas existem no sistema de ficheiros. Fornece estatísticas sobre o número de sessões encontradas e a quantidade de imagens disponíveis para treino. É útil como ferramenta de diagnóstico para garantir que todos os dados necessários estão disponíveis antes de iniciar o processo de treino.

## 9. app_tensor_detail.py
Similar ao app_torch_detail.py, mas projetado para modelos treinados com TensorFlow/Keras. Realiza avaliação detalhada do desempenho de um modelo de direção autónoma, gerando métricas e visualizações para compreender a sua eficácia. Inclui capacidades específicas do TensorFlow como Grad-CAM (Gradient-weighted Class Activation Mapping) para visualizar quais regiões da imagem são mais influentes nas predições do modelo. Também gera um relatório HTML abrangente com todas as análises realizadas. O programa é otimizado para funcionar mesmo em ambientes sem GPU, forçando a execução em CPU quando necessário.

## 10. app_training_tensor.py
Este programa implementa o treino de um modelo de condução autónoma utilizando TensorFlow/Keras. Similar ao app_training_torch.py em funcionalidade, mas utilizando a biblioteca Keras. Carrega dados de múltiplas sessões, aplica balanceamento de dados e técnicas de aumento de dados para melhorar a generalização. Implementa a arquitetura NVIDIA para condução autónoma com camadas convolucionais e densas utilizando ativação ELU (Exponential Linear Unit). O programa é configurado para funcionar em CPU, tornando-o acessível em ambientes sem aceleração GPU. Inclui visualizações do processo de treino, como gráficos de perda (loss) e distribuição de ângulos de direção, e guarda o modelo final para uso posterior.

## Carla em Modo Simples
/CarlaUE4.sh -RenderOffScreen -quality-level=Low
