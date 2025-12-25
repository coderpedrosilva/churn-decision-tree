import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# Caminhos do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_synthetic.csv")

# Carregar dados
df = pd.read_csv(DATA_PATH)

# Separar variáveis explicativas e variável alvo
X = df.drop("churn", axis=1)
y = df["churn"]

# Encoding das variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Divisão em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Recriar o modelo de Árvore de Decisão
modelo = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

modelo.fit(X_treino, y_treino)

# 1️⃣ Importância das variáveis
importancia_variaveis = pd.DataFrame({
    "variavel": X.columns,
    "importancia": modelo.feature_importances_
}).sort_values(by="importancia", ascending=False)

print("\n Importância das Variáveis:")
print(importancia_variaveis)

# 2️⃣ Regras aprendidas pela árvore (formato texto)
print("\n Regras aprendidas pela Árvore de Decisão:\n")
regras_arvore = export_text(modelo, feature_names=list(X.columns))
print(regras_arvore)
