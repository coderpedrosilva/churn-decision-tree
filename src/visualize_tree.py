import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Caminhos do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_synthetic.csv")
IMG_DIR = os.path.join(BASE_DIR, "images")

# Garantir que a pasta de imagens exista
os.makedirs(IMG_DIR, exist_ok=True)

# Carregar dados
df = pd.read_csv(DATA_PATH)

# Separar variáveis explicativas e alvo
X = df.drop("churn", axis=1)
y = df["churn"]

# Encoding das variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Divisão treino / teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Modelo de Árvore de Decisão
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Plot da árvore
plt.figure(figsize=(22, 12))
plot_tree(
    model,
    feature_names=X.columns,          # nomes das variáveis (já em PT)
    class_names=["Permanece", "Churn"],# rótulos em português
    filled=True,
    rounded=True,
    fontsize=9
)

# Salvar imagem
image_path = os.path.join(IMG_DIR, "decision_tree.png")
plt.savefig(image_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Árvore de decisão salva em: {image_path}")
