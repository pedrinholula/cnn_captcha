# README
Esse é o Readme com o básico para começar.

Documentação mais detalhada em DOC.MD

## Inicializando
Baixar as dependencias no Pipfile
### Treinando o modelo
Da pasta principal, executar `./.src/train.py`
### Iniciar a API
Executar `cd api && fastapi dev main.py`

## Estrutura de pastas
```
captcha-ocr/
│
├── api/
	├── models/				# Modelo princial
	├── routes/				# Rotas da API
│   ├── main.py				# Entrypoint da api
│   └── utils.py			# Funções secundárias
|
├── data/
│   ├── samples/			# Imagens originais do dataset
│   └── tmp/          		# Pasta temporária
│
├── notebooks/
│   ├── cnn.ipynb			# Testes com as CNNs
│   └── llm.ipynb			# testes com LLM
│
├── src/ 					# Funções principais do modelo
 	└── train.py			# Arquivo principal para execução do pipeline completo
│
│
├── models/
│   └── production.h5		# Modelo em produção
│
├── Pipfile			        # Dependências do projeto
└── README.md               # Documentação do projeto
```