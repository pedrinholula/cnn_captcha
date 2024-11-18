# Quebra de captcha usando IA

Este projeto aborda a resolução automatizada de Captchas alfanuméricos com 5 digitos, utilizando uma abordagem a partir de IA como redes neurais CRNN (Convolutional Recurrent Neural Networks). Sabe-se que  captchas são amplamente utilizados para verificar atividades humanas online, sendo alguns tipos extremamente desafiadores para sistemas automatizados. Foram-se tentadas 3 abordagens diferentes, sendo escolhida a utilização de CRNN através de um pipeline que inclui desde o pré-processamento das imagens até a avaliação detalhada do modelo, com foco em eficiência e precisão. O trabalho explora limitações da abordagem e aponta direções para aprimoramentos futuros.

## Introdução


### Contexto e motivação
Os Captchas são amplamente utilizados como barreira de segurança. Eles visam distinguir usuários humanos de bots, protegendo sistemas contra acessos automatizados maliciosos. Porém, com o avanço da inteligência artificial, métodos automáticos têm se tornado cada vez mais eficazes, representando uma oportunidade para aplicar novas técnicas de reconhecimento visual.

## Trabalhos relacionados
A resolução automática de Captchas é  amplamente explorada principalmente após o avanço das redes neurais profundas. Métodos convencionais utilizavam técnicas baseadas em extração manual de caracteres e modelos de classificação como SVMs. No entanto, esses métodos são pouco eficazes em captchas com distorções ou ruídos complexos.

Recentemente, abordagens bque se baseiam em deep learning, como redes convolucionais para extração de características e redes recorrentes para reconhecimento de sequências se mostram melhores que os métodos anteriores. O uso de CRNNs, que combinam essas duas formas, é uma boa solução. Além disso, o Mecanismo de Atenção permite que o modelo foque em partes relevantes da imagem, melhorando a precisão em caracteres sobrepostos ou distorcidos.

Este projeto se inspira nesses avanços, implementando uma solução prática baseada nas arquiteturas CRNN e Atenção, utilizando TensorFlow e técnicas modernas de pré-processamento com o Pillow e o OpenCV.

## Metodologia
O pipeline deste projeto foi dividido em três etapas principais:
1. Pré-processamento
2. Arquitetura do modelo
3. Treinamento e validação
4. Disponibilização do modelo via API

### Pré-processamento

O dataset contém 1.070 imagens de Captchas alfanuméricos, cada um com 5 caracteres, porém apenas 19 dimensões. Percebeu-se que alguns caracteres como `0,1,i,o` não estavam presentes. Além disso, apenas caracteres minúsculos foram detectados. Para melhorar a qualidade dos dados de entrada, foram aplicadas as seguintes etapas:

- Redimensionamento: As imagens foram redimensionadas para (200x50) pixels para manter consistência.
- Conversão para Escala de Cinza: Simplificou o modelo ao reduzir a dimensionalidade.
- Threshold: Aplicado para binarização das imagens, aumentando o contraste entre caracteres e o fundo.
- Testes com Dilatação e Fechamento: Ambos os métodos foram avaliados para reduzir ruídos e melhorar a segmentação dos caracteres. O Fechamento foi escolhido, pois apresentou melhores resultados na preservação da forma dos caracteres.
- Normalização: Os valores de pixel foram escalados para o intervalo [0, 1].
- Codificação dos Rótulos: Os rótulos foram convertidos para uma representação one-hot encoding.

### Arquitetura do modelo
A solução utiliza uma combinação de Redes Neurais Convolucionais e Recorrentes com Mecanismo de Atenção.

Estrutura:

- Camadas convolucionais: Extraem características das imagens.
- Camadas Conv2D com ativação ReLU e MaxPooling2D para reduzir as dimensões.
- Redimensionamento: Recursos extraídos são reorganizados para serem processados como sequências temporais.
- Camadas Recorrentes (BiLSTM): Capturam dependências temporais entre caracteres.
- Mecanismo de Atenção: Permite ao modelo focar em regiões relevantes da sequência e melhorar o reconhecimento.
- Camada de Saída: Produz a probabilidade de cada caractere em cada posição.

Além disso, como tentativa inicial, foi testado o uso do PyTesseract, que apresentou limitações em devido ao ruído presente nas imagens. Outra tentativa foi o uso de LLMs multimodal, como o Gemini que apresentou igualmente uma baixa acurácia - ambos abaixo de 0,4.

### Treinamento e validação

- Divisão de Dados: Os dados foram divididos aleatoriamente, reservando 80% para treinamento e 20% para validação. Essa abordagem simplificada permitiu uma avaliação eficiente do modelo. Foi testado o 5 K-fold com leave one out, mas para modelos complexos, a divisão é ineficiente.
- Configuração de hiperparâmetros:
	- Batch Size: 32
	- Épocas: 50
	- Otimização: Adam com taxa de aprendizado inicial de 0.001.
	- Função de Perda: categorical_crossentropy para classificação multi-classe.
- Treinamento: Implementado em TensorFlow, monitorando a métrica accuracy e ajustando hiperparâmetros conforme necessário.

### Disponibilização via API
Para o uso do modelo foi desenvolvida uma API utilizando FastAPI. A API foi projetada para receber imagens no formato multipart/form-data, no campo file. O fluxo de processamento é o seguinte:

1. A imagem é enviada como upload no campo file;
2. O servidor realiza o pré-processamento internamente, aplicando as etapas descritas anteriormente;
3. A imagem processada é passada para o modelo treinado;
4. O modelo retorna a sequência de caracteres reconhecidos, que é enviada como resposta ao cliente;

## Testes e resultados

Realizamos testes com dois tipos de extratificação, com diferentes pré-processamentos e número de épocas. As métricas de avaliação consideraram a acurácia média e o desvio padrão.
Variações no tamanho do kernel para dilatação e fechamento foram testadas, com os seguintes tamanhos: 1x1, 1x3, 3x3, 3x5, 5x3 e 5x5. Os melhores resultados foram obtidos com os kernels 3x3 e 3x5, sendo o fechamento 3x3 o escolhido para a arquitetura final.

### Resultados com 5-fold e 10 Épocas:
	- Pré-processamento com Dilatação:
		- Acurácia média: 41.05%
		- Desvio padrão: 1.74%
	- Pré-processamento com Fechamento:
		- Acurácia média: 40.54%
		- Desvio padrão: 3.94%
### Resultados com 0.2 Stratification:
	- 10 Épocas:
		- Fechamento: Acurácia média: 39.16%
		- Dilatação: Acurácia média: 33.08%
	- 30 Épocas:
		- Fechamento: Acurácia média: 70.31%
		- Dilatação: Acurácia média: 53.55%
	- 50 Épocas:
		- Fechamento: Acurácia média: 59.51%
		- Dilatação: Acurácia média: 71.03%

### Comparação com Outros Modelos:
- PyTesseract (856 testes): Acurácia média: 37.5%
- Gemini-1.5-flash (100 testes): Acurácia média: 41.82%

## Discussão e aprendizados
Os experimentos mostraram que o modelo CRNN com pré-processamento e a escolha de kernels específicos para dilatação ou fechamento proporcionaram um desempenho superior em comparação com outras abordagens.
### Aprendizados
- Pré-processamento e escolha de kernel: O uso do kernel 3x3, tanto para dilatação quanto para fechamento, apresentou bons resultados o que confirma a importância de pre-processamento e de ajustar bem esses parâmetros para melhorar a segmentação dos caracteres. Esse resultado é consistente com a literatura, onde kernels de tamanho médio (como 3x3) são frequentemente os mais eficazes e recomendados para preservar as características essenciais das imagens de captcha, ao mesmo tempo em que eliminam ruídos excessivos.

- Impacto das épocas: A acurácia do modelo melhorou significativamente com o aumento das épocas, até 30, alcançando uma média de 70.31% com o pre-processamento de fechamento. Esse resultado indica que o modelo se beneficiou de mais iterações para aprender os padrões nas imagens de captcha. Ainda assim é bom observar que uma  queda no desempenho após o valor. Isso pode mostrar overfitting, especialmente com o pré-processamento de fechamento.

- Comparação com outros modelos: Os modelos tradicionais, como PyTesseract e a LLM, apresentaram desempenhos bem inferiores, com acurácias abaixo de 42%. Isso reforça a eficácia do modelo CRNN, que é mais robusto ao lidar com as complexidades dos captchas, como distorções e ruídos. Isso não quer dizer que algumas abordagens a LLM multimodal não possa se beneficiar, ainda mais com o avanço da área.

### Desafios
- Ruídos e distorções: Apesar das melhorias com o pré-processamento, ainda houve alguns casos de falhas no reconhecimento, principalmente em captchas com distorções complexas ou sobreposição de caracteres. Isso indica que há espaço para aprimorar o modelo.

- Overfitting: A performance que aumentou inicialmente e depois apresentou uma queda sugerindo que o modelo pode estar se superajustando ao treinamento. Técnicas adicionais, como regularização ou aumento de dados, podem ser úteis para acabar com esse problema.

- Limitada experiência com DevOps: Um ponto importante a ser destacado é que a não disponibilização da API usando ferramentas de container, como o Docker, foi devido à falta de experiência com práticas de DevOps. Isso limitou a escalabilidade e a portabilidade da solução.

### Melhorias

- Otimização de API e modelos: A API ainda pode ser aprimorada com a inclusão de testes automatizados, validações de entrada e saída de dados e melhorias nas em segurança, CI/CD, dentre outros. Além disso, a otimização do modelo, considerando ajustes finos de hiperparâmetros, pode gerar melhores resultados de acurácia e desempenho.

- Aumento de dados: O uso de técnicas de aumento de dados, como distorções aleatórias nas imagens de treino, pode ajudar o modelo a generalizar melhor para novos casos de captcha.

- Exploração de modelos híbridos: Integrar a abordagem CRNN com outras técnicas, como GANs, para gerar captchas mais desafiadores para o treinamento, pode melhorar a precisão do modelo.

- DevOps: Para aprimorar a entrega e a escalabilidade da solução, seria importante investir em containers (como Docker), o que permitiria a fácil implementação e execução do modelo em diferentes ambientes, mesmo sem experiência avançada em DevOps

## Conclusão

O teste foi divertido de fazer e me fez relembrar de vários conceitos. No geral estou satisfeito, ainda que eu saiba que muita coisa pode melhorar.

### Referencias

Algumas referencias:

- https://cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf
- https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
- https://github.com/Suganth10/Captcha-Recognition-using-CNN
- https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras/blob/master/CRNN%20Model.ipynb
- https://nanonets.com/blog/ocr-with-tesseract/

Stack Overflow (sempre)
ChatGPT e Gemini me a ajudando a debugar e escrever!