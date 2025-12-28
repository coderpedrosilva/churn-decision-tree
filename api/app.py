from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

# Carregar modelo treinado
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "api", "model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Churn Decision API")

# =========================
# Modelo de entrada
# =========================
class ClienteInput(BaseModel):
    tempo_contrato_meses: int
    valor_mensal: float
    uso_total_gb: float
    chamadas_suporte: int
    pagamentos_atrasados: int
    tipo_contrato: str  # "mensal" ou "anual"
    renovacao_automatica: int  # 0 ou 1
    desconto: int  # 0 ou 1

# =========================
# Endpoint de decisão
# =========================
@app.post("/predict")
def predict(cliente: ClienteInput):

    # Montar input no mesmo formato do treino
    tipo_mensal = 1 if cliente.tipo_contrato == "mensal" else 0

    X = np.array([[
        cliente.tempo_contrato_meses,
        cliente.valor_mensal,
        cliente.uso_total_gb,
        cliente.chamadas_suporte,
        cliente.pagamentos_atrasados,
        tipo_mensal,
        cliente.renovacao_automatica,
        cliente.desconto
    ]])

    prob = model.predict_proba(X)[0][1]

    return {
        "probabilidade_churn": round(float(prob), 2),
        "classe": "Cliente em Evasão" if prob >= 0.5 else "Cliente Ativo"
    }
