# TD-HDARP — Ruteo de Profesionales de la Salud

Implementación del problema TD-HDARP (Time-Dependent Heterogeneous Dial-a-Ride Problem) para el Capstone ICS2122. Combina un modelo MILP exacto resuelto con Gurobi y una metaheurística ALNS adaptada de Pilati et al. (2025), con un módulo predictivo XGBoost para la matriz de tiempos de viaje dependiente de la hora.

## Setup

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar
pip install -e ".[dev]"

# Verificar Gurobi (requiere licencia académica activa)
python -c "import gurobipy as gp; m = gp.Model(); print('Gurobi OK')"

# Correr tests
pytest
```

## Estructura

- `src/data/` — Carga y representación de instancias.
- `src/milp/` — MILP en Gurobi (referencia y oráculo del Caso Base).
- `src/alns/` — Algoritmo ALNS con operadores destroy/repair y función de evaluación con penalizaciones dinámicas.
- `src/baselines/` — Heurística miope para comparación.
- `src/predictive/` — Modelo XGBoost para $\tau_{ij}(t)$.
- `src/reports/` — KPIs y visualizaciones.
- `scripts/` — CLIs para correr Caso Base e instancia completa.
- `tests/` — Tests unitarios e integración.
- `notebooks/` — Resultados ejecutables.

## Datos

Los datos están en `DATOS P5 - Ruteo de profesionales de la salud/`. El loader los lee desde ahí.

## Reproducibilidad

```bash
# MILP sobre Caso Base (oráculo)
python scripts/run_case_base.py --solver milp --time 43200

# ALNS sobre Caso Base (5 seeds)
python scripts/run_case_base.py --solver alns --seeds 5

# Heurística miope
python scripts/run_case_base.py --solver miope

# Comparación
python scripts/compare.py
```

## Referencias

- Pilati, F., Tronconi, R., & Doerner, K. F. (2025). Tailored ALNS to optimize real-world logistic services for dependent patients.
- Zhao, J., Poon, M., Zhang, Z., & Gu, R. (2022). Adaptive large neighborhood search for the time-dependent profitable dial-a-ride problem.
- Portell, L., & Lourenço, H. R. (2024). The rich heterogeneous dial-a-ride problem with trip time prediction.
- Detti, P., Papalini, F., & Zabalo Manrique de Lara, G. (2017). A multi-depot dial-a-ride problem with heterogeneous vehicles and compatibility constraints in healthcare.
- Ropke, S., & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows.
