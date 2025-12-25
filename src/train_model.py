import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Caminho até a raiz do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_synthetic.csv")

# 1️⃣ Carregar dados
df = pd.read_csv(DATA_PATH)

# 2️⃣ Separar variáveis explicativas e variável alvo
X = df.drop("evasao", axis=1)
y = df["evasao"]

# 3️⃣ Encoding das variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# 4️⃣ Separar dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5️⃣ Criar modelo de Árvore de Decisão
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

# 6️⃣ Treinar o modelo (aprendizado acontece aqui)
model.fit(X_train, y_train)

# 7️⃣ Gerar previsões
y_pred = model.predict(X_test)

# 8️⃣ Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Acurácia do modelo:", round(accuracy, 4))

print("\nMatriz de Confusão:")
print(cm)

print("\nRelatório de Classificação:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Cliente Ativo", "Cliente em Evasão"]
))
