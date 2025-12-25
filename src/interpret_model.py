import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# Caminhos do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_synthetic.csv")

# Carregar dados
df = pd.read_csv(DATA_PATH)

X = df.drop("churn", axis=1)
y = df["churn"]

# Encoding
X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Recriar o modelo
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# 1️⃣ Importância das features
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nImportância das Features:")
print(feature_importance)

# 2️⃣ Regras da árvore (texto)
print("\nRegras aprendidas pela Árvore:\n")
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
