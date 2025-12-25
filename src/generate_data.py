import numpy as np
import pandas as pd

# Garantir reprodutibilidade
np.random.seed(42)

# Número de registros
N = 1500

# =========================
# Variáveis numéricas
# =========================

tempo_contrato_meses = np.random.randint(1, 60, N)
valor_mensal = np.random.uniform(50, 300, N).round(2)
uso_total_gb = np.random.uniform(10, 1000, N).round(1)
chamadas_suporte = np.random.poisson(2, N)
pagamentos_atrasados = np.random.poisson(1, N)

# =========================
# Variáveis categóricas / binárias
# =========================

tipo_contrato = np.random.choice(
    ["mensal", "anual"],
    size=N,
    p=[0.65, 0.35]
)

renovacao_automatica = np.random.choice([0, 1], size=N, p=[0.4, 0.6])
desconto = np.random.choice([0, 1], size=N, p=[0.5, 0.5])

# =========================
# Probabilidade base de churn
# =========================

probabilidade_churn = (
    0.35
    + (chamadas_suporte * 0.08)
    + (pagamentos_atrasados * 0.10)
    - (tempo_contrato_meses * 0.004)
    - (renovacao_automatica * 0.15)
    - (desconto * 0.10)
)

# Ajuste conforme tipo de contrato
probabilidade_churn += np.where(
    tipo_contrato == "mensal", 0.15, -0.20
)

# Garantir valores entre 0 e 1
probabilidade_churn = np.clip(probabilidade_churn, 0, 1)

# =========================
# Variável alvo (churn)
# =========================

churn = np.random.binomial(1, probabilidade_churn)

# =========================
# Criar DataFrame
# =========================

df = pd.DataFrame({
    "tempo_contrato_meses": tempo_contrato_meses,
    "valor_mensal": valor_mensal,
    "uso_total_gb": uso_total_gb,
    "chamadas_suporte": chamadas_suporte,
    "pagamentos_atrasados": pagamentos_atrasados,
    "tipo_contrato": tipo_contrato,
    "renovacao_automatica": renovacao_automatica,
    "desconto": desconto,
    "churn": churn
})

# =========================
# Salvar dataset
# =========================

df.to_csv("data/churn_synthetic.csv", index=False)

print("Dataset de churn gerado com sucesso!")
print(df.head())
