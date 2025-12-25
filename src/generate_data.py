import numpy as np
import pandas as pd

np.random.seed(42)

N = 1500

# Variáveis numéricas
tenure_months = np.random.randint(1, 60, N)
monthly_fee = np.random.uniform(50, 300, N).round(2)
total_usage_gb = np.random.uniform(10, 1000, N).round(1)
support_calls = np.random.poisson(2, N)
late_payments = np.random.poisson(1, N)

# Variáveis categóricas / binárias
contract_type = np.random.choice(
    ["monthly", "annual"],
    size=N,
    p=[0.65, 0.35]
)

auto_renew = np.random.choice([0, 1], size=N, p=[0.4, 0.6])
discount = np.random.choice([0, 1], size=N, p=[0.5, 0.5])

# Probabilidade base de churn
churn_prob = (
    0.35
    + (support_calls * 0.08)
    + (late_payments * 0.10)
    - (tenure_months * 0.004)
    - (auto_renew * 0.15)
    - (discount * 0.10)
)

# Ajustes por tipo de contrato
churn_prob += np.where(contract_type == "monthly", 0.15, -0.20)

# Limitar probabilidade entre 0 e 1
churn_prob = np.clip(churn_prob, 0, 1)

# Gerar churn (variável alvo)
churn = np.random.binomial(1, churn_prob)

# Criar DataFrame
df = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_fee": monthly_fee,
    "total_usage_gb": total_usage_gb,
    "support_calls": support_calls,
    "late_payments": late_payments,
    "contract_type": contract_type,
    "auto_renew": auto_renew,
    "discount": discount,
    "churn": churn
})

# Salvar CSV
df.to_csv("data/churn_synthetic.csv", index=False)

print("Dataset gerado com sucesso!")
print(df.head())
