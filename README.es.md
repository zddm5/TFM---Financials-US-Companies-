# Predicción de Quiebras Empresariales con Machine Learning

> Trabajo Fin de Máster (TFM) — Máster en Data Science, Big Data & Business Analytics, **Universidad Complutense de Madrid** (Febrero 2026)

Pipeline de machine learning que predice la insolvencia empresarial a partir de 18 indicadores financieros, comparando **XGBoost**, **Regresión Logística** y **Regresión Lineal** sobre un dataset de **78.682 empresas estadounidenses (1999–2018)**.

**🇬🇧 English version:** [README.md](./README.md)

---

## 📊 Resultados principales

| Modelo | ROC-AUC (Test) | Accuracy | Recall (Quiebras) |
|---|---|---|---|
| Regresión Lineal | 0,6293 | 0,0663 | 1,000 |
| Regresión Logística | 0,6679 | 0,3724 | 0,831 |
| **XGBoost (ganador)** | **0,8125** | **0,8201** | 0,6025 |

- **ROC-AUC de 0,8125** sobre 15.737 empresas de test — competitivo con baselines sofisticados de AutoML.
- Gestiona un **desbalance de clases 1:14** (solo el 6,6% de empresas son quiebras) mediante `scale_pos_weight`.
- Principales predictores de quiebra: **operating risk, net profit margin, quick ratio, interest coverage, financial flexibility**.

---

## 🧠 Qué hace el proyecto

1. **EDA** de 78.682 registros empresariales con 18 indicadores financieros normalizados en cinco categorías:
   - *Liquidez*: quick ratio, current ratio, cash flow ratio
   - *Rentabilidad*: ROA, ROE, net profit margin
   - *Endeudamiento*: debt ratio, debt-to-equity, interest coverage
   - *Eficiencia*: asset turnover, revenue growth, working capital ratio
   - *Riesgo*: management risk, industrial risk, operating risk
2. **Preprocesamiento**: imputación por mediana, split 80/20 estratificado, `StandardScaler` ajustado solo sobre train.
3. **Tres modelos entrenados en paralelo** — cada uno con un rol diferente (referencia, baseline interpretable, modelo principal).
4. **Evaluación** centrada en ROC-AUC (apropiada para clasificación desbalanceada) más matrices de confusión y análisis de coste (un falso negativo es mucho más caro que un falso positivo en riesgo de crédito).
5. **Interpretabilidad** vía feature importance y valores SHAP.
6. **Demo de productivización**: dos perfiles sintéticos de empresa (sana vs. en riesgo) para mostrar que el modelo se comporta razonablemente en los extremos.

---

## 🏗️ Estructura del repositorio

```
.
├── notebooks/
│   └── Corporate_Bankruptcy_Prediction_XGBoost.ipynb   # Análisis end-to-end
├── docs/
│   └── TFM_Final.pdf                                   # Memoria completa (español)
├── requirements.txt                                    # Dependencias Python
├── .gitignore
├── LICENSE
├── README.md                                           # Versión en inglés
└── README.es.md                                        # Este archivo (español)
```

---

## 🚀 Cómo ejecutarlo

### Requisitos
- Python 3.10+
- `pip` o `conda`

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/zddm5/TFM---Financials-US-Companies-.git
cd TFM---Financials-US-Companies-

# 2. Crear un entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Abrir el notebook
jupyter notebook notebooks/Corporate_Bankruptcy_Prediction_XGBoost.ipynb
```

---

## 🔬 Puntos clave de la metodología

### Por qué XGBoost como modelo principal
- Captura interacciones no lineales típicas de ratios financieros.
- Manejo nativo del desbalance de clases mediante `scale_pos_weight ≈ 14,07`.
- Alta interpretabilidad vía feature importance y SHAP.
- No requiere escalado de features.

### Configuración XGBoost
```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=14.07,
    random_state=42,
)
```

### Por qué ROC-AUC como métrica principal
Con un desbalance 1:14, la accuracy es engañosa y el F1 depende del umbral elegido. ROC-AUC resume la capacidad de discriminación para todos los umbrales. Además se analizó el coste de cada tipo de error: un **falso negativo (aprobar crédito a una empresa que luego quiebra) es mucho más caro que un falso positivo**, por lo que se prioriza la sensibilidad sobre la especificidad.

### Simulación de escenarios
Se construyeron dos perfiles sintéticos llevando los 18 indicadores a percentiles extremos:
- **Perfil sano** (alta rentabilidad y liquidez, baja deuda y riesgo) → probabilidad de supervivencia **99,99%**.
- **Perfil en riesgo** (opuesto) → probabilidad de supervivencia **98,77%**, es decir **1,23% de riesgo de quiebra** — la diferencia entre extremos es económicamente relevante para decisiones de crédito.

---

## ⚠️ Limitaciones

- El **desbalance (1:14)** limita el recall a ~60% con threshold por defecto — en la práctica, los equipos de crédito deberían ordenar empresas por probabilidad y actuar sobre el top-N más arriesgado.
- Modelo entrenado con datos de EE.UU. 1999–2018: se recomienda reentrenamiento cada 6–12 meses y no se garantiza la transferencia a otros mercados o periodos.
- El modelo **no** captura shocks macroeconómicos, factores cualitativos (cambios de dirección, reputación) ni ciclos sectoriales específicos.
- Sobreajuste moderado (AUC train 0,958 vs. test 0,813), aceptable para este problema pero a monitorizar.

---

## 🆚 Comparativa con alternativas

| Enfoque | Fortaleza | Debilidad vs. este trabajo |
|---|---|---|
| Z-Score de Altman (1968) | Simple, muy conocido | Supuestos lineales, menor granularidad |
| AutoML (auto-sklearn, H2O) | Mayor accuracy bruto posible | Menos control, menor interpretabilidad, más cómputo |
| **Este proyecto (XGBoost)** | Interpretable, consciente del coste, reproducible | Requiere reentrenamiento periódico |

---

## 💼 Aplicaciones prácticas

- **Entidades financieras**: scoring de crédito, monitorización de cartera, provisiones por pérdidas esperadas.
- **Inversores**: construcción de cartera ajustada al riesgo, sistemas de alerta temprana.
- **Reguladores**: monitorización de riesgo sistémico por industria.

---

## 📄 Memoria del TFM

La memoria completa del TFM (en español) está disponible aquí: [docs/TFM_Final.pdf](./docs/TFM_Final.pdf).

---

## 👤 Autor

**Diego José Zuniga López**
Máster en Data Science, Big Data & Business Analytics — Universidad Complutense de Madrid
📍 Madrid, España
🔗 [GitHub @zddm5](https://github.com/zddm5)

---

## 📜 Licencia

Este proyecto se publica bajo la Licencia MIT — ver [LICENSE](./LICENSE) para más detalles.
