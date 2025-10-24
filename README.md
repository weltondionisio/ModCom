# ModCom

The model wants to edit files outside of your workspace (c:\Users\Welton Dionisio\Desktop\MC\README.md). Do you want to allow this?

Contents:

# Modelos Computacionais - Análise de Dados Biológicos

Este repositório contém um Jupyter Notebook (`modcom.ipynb`) utilizado na disciplina de Modelos Computacionais aplicados a dados biológicos, com foco em análise de dados e aprendizado de máquina.

## Estrutura do Notebook

O notebook está organizado em três seções principais:

### 1. Análise de Preços de Casas
- Carregamento de dados usando KaggleHub
- Pré-processamento de dados
  - Remoção de variáveis não relevantes
  - Tratamento de dados ausentes
  - Codificação de variáveis categóricas
- Implementação de modelos:
  - Regressão Linear
  - Random Forest
  - XGBoost
  - Rede Neural
- Visualizações:
  - Gráficos de dispersão
  - Distribuição de resíduos
  - Importância das variáveis

### 2. Análise do Dataset Iris
- Redução de dimensionalidade (PCA)
- Clusterização
  - K-Means (k=2 e k=3)
  - DBSCAN
- Visualizações:
  - Variância explicada por componentes principais
  - Pontuação de silhueta
  - Gráfico de cotovelo
  - Comparação de agrupamentos

### 3. Rede Neural Linear
- Implementação de rede neural usando TensorFlow
- Otimização de hiperparâmetros com Keras Tuner
- Avaliação de modelos:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - R² Score
- Visualizações do treinamento:
  - Curvas de loss
  - Previsões vs valores reais

## Bibliotecas Utilizadas
- pandas: Manipulação de dados
- numpy: Computação numérica
- matplotlib & seaborn: Visualização de dados
- scikit-learn: Algoritmos de machine learning
- tensorflow: Deep learning
- xgboost: Gradient boosting
- keras-tuner: Otimização de hiperparâmetros

## Requisitos
```python
kagglehub[pandas-datasets]
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
xgboost
keras-tuner
Como Executar
Clone este repositório
Instale as dependências necessárias
Abra o notebook modcom.ipynb em um ambiente Jupyter
Execute as células sequencialmente
Visualizações Geradas
O notebook gera várias visualizações salvas em alta resolução:

gráficos.tiff: Resultados da regressão linear
gráficos_xgboost.tiff: Resultados do modelo XGBoost
PCA.png: Análise de componentes principais
pontuação_de_silhueta.tiff: Análise de clusters
cotovelo.tiff: Método do cotovelo para clustering
Modelos Salvos
O melhor modelo treinado é salvo como best_house_price_model.h5
Notas de Aula
Este notebook foi criado como parte da aula do dia 22/10/2025, focando em aplicações práticas de modelos computacionais em dados biológicos.

Allow
Skip
