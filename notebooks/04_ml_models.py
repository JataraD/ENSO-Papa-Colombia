# ============================================================
# 04_ml_models.py
# ML Pipeline — Predicción de Precios de Papa
# Regresión Lineal · Ridge · Random Forest · XGBoost
# Feature Engineering + Comparación automática de modelos
# Autor: Juan David Atará Delgado | Mayo 2026
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LinearRegression, Ridge, Lasso
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline        import Pipeline
import xgboost as xgb

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("=== ML PIPELINE — PREDICCIÓN PRECIO PAPA ===\n")

# ════════════════════════════════════════════════════════════
# 1. CARGA Y FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
print("── 1. Feature Engineering ──────────────────────────")

conn = sqlite3.connect("data/processed/enso_papa_db.sqlite")

df = pd.read_sql("""
    SELECT
        o.year, o.month, o.oni,
        o.phase,
        ROUND(AVG(p.precio_cop_kg), 0)      AS precio,
        ROUND(AVG(c.precipitacion_mm), 2)   AS lluvia,
        ROUND(AVG(c.temperatura_c), 3)      AS temperatura
    FROM oni_index o
    JOIN precios p ON o.year=p.year AND o.month=p.month
    JOIN clima c   ON o.year=c.year AND o.month=c.month
    WHERE p.variedad = 'Papa Parda Pastusa'
      AND p.mercado  = 'Corabastos Bogotá'
      AND c.region   = 'Cundinamarca'
    GROUP BY o.year, o.month
    ORDER BY o.year, o.month
""", conn)
conn.close()

df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
df = df.sort_values('date').reset_index(drop=True)

# ── Feature Engineering ──────────────────────────────────────

# 1. Lags del precio (precios pasados)
for lag in [1, 2, 3, 6, 12]:
    df[f'precio_lag{lag}'] = df['precio'].shift(lag)

# 2. Media móvil del precio
for window in [3, 6, 12]:
    df[f'precio_ma{window}'] = df['precio'].shift(1).rolling(window).mean()

# 3. Lags del ONI (el efecto climático tiene retardo)
for lag in [1, 2, 3]:
    df[f'oni_lag{lag}'] = df['oni'].shift(lag)

# 4. Media móvil del ONI
df['oni_ma3'] = df['oni'].rolling(3).mean()
df['oni_ma6'] = df['oni'].rolling(6).mean()

# 5. Features temporales
df['mes_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['trend']   = np.arange(len(df))  # tendencia lineal

# 6. Features climáticas con lag
df['lluvia_lag1']      = df['lluvia'].shift(1)
df['lluvia_lag2']      = df['lluvia'].shift(2)
df['temperatura_lag1'] = df['temperatura'].shift(1)

# 7. Variables dummy de fase ENSO
df['es_nino'] = (df['phase'] == 'El Niño').astype(int)
df['es_nina'] = (df['phase'] == 'La Niña').astype(int)

# 8. Variación porcentual del precio (target alternativo)
df['precio_variacion_pct'] = df['precio'].pct_change() * 100

# 9. ONI al cuadrado (efecto no lineal)
df['oni_squared'] = df['oni'] ** 2

# Eliminar NAs
df_clean = df.dropna().reset_index(drop=True)

print(f"  Dataset: {len(df_clean)} meses con features completas")
print(f"  Período: {df_clean['date'].min().strftime('%Y-%m')} "
      f"→ {df_clean['date'].max().strftime('%Y-%m')}")
print(f"  Features generadas: {len([c for c in df_clean.columns if c not in ['date','year','month','precio','phase']])}")

# ── Definir features ─────────────────────────────────────────
FEATURES = [
    'precio_lag1', 'precio_lag2', 'precio_lag3',
    'precio_lag6', 'precio_lag12',
    'precio_ma3', 'precio_ma6', 'precio_ma12',
    'oni', 'oni_lag1', 'oni_lag2', 'oni_lag3',
    'oni_ma3', 'oni_ma6', 'oni_squared',
    'mes_sin', 'mes_cos', 'trend',
    'lluvia_lag1', 'lluvia_lag2', 'temperatura_lag1',
    'es_nino', 'es_nina',
]

X = df_clean[FEATURES]
y = df_clean['precio']

# Train/test split temporal (últimos 24 meses = test)
SPLIT = len(df_clean) - 24
X_train, X_test = X.iloc[:SPLIT], X.iloc[SPLIT:]
y_train, y_test = y.iloc[:SPLIT], y.iloc[SPLIT:]
dates_test = df_clean['date'].iloc[SPLIT:]

print(f"\n  Train: {SPLIT} meses | Test: {len(X_test)} meses")

# ════════════════════════════════════════════════════════════
# 2. ENTRENAMIENTO DE MODELOS
# ════════════════════════════════════════════════════════════
print("\n── 2. Entrenamiento de Modelos ─────────────────────")

models = {
    'Regresión Lineal': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge (α=10)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=10))
    ]),
    'Lasso (α=10)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=10, max_iter=10000))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=200, max_depth=8,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        ))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42
        ))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb.XGBRegressor(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=0
        ))
    ]),
}

# TimeSeriesSplit para cross-validation
tscv = TimeSeriesSplit(n_splits=5)

results = {}
predictions = {}

for name, pipeline in models.items():
    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=tscv, scoring='neg_mean_absolute_error'
    )

    # Entrenar en train completo
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results[name] = {
        'MAE'     : round(mae, 1),
        'RMSE'    : round(rmse, 1),
        'R2'      : round(r2, 4),
        'MAPE_pct': round(mape, 2),
        'CV_MAE'  : round(-cv_scores.mean(), 1),
        'CV_std'  : round(cv_scores.std(), 1),
    }
    predictions[name] = y_pred

    print(f"  {name:<22} MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  "
          f"R²={r2:.3f}  MAPE={mape:.1f}%")

# ════════════════════════════════════════════════════════════
# 3. MEJOR MODELO + FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
print("\n── 3. Mejor modelo ─────────────────────────────────")

best_name = min(results, key=lambda x: results[x]['MAE'])
best_model = models[best_name]
print(f"  🏆 Mejor modelo: {best_name}")
print(f"     MAE  = ${results[best_name]['MAE']:,.0f} COP/kg")
print(f"     RMSE = ${results[best_name]['RMSE']:,.0f} COP/kg")
print(f"     R²   = {results[best_name]['R2']:.4f}")
print(f"     MAPE = {results[best_name]['MAPE_pct']:.1f}%")

# Feature importance del mejor modelo (si aplica)
feature_importance = None
inner_model = best_model.named_steps['model']

if hasattr(inner_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'feature'   : FEATURES,
        'importance': inner_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance = fi.head(15)
    print(f"\n  Top 5 features más importantes:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"    {row['feature']:<25}: {row['importance']:.4f}")
    feature_importance.to_csv(
        "data/processed/feature_importance.csv", index=False)

elif hasattr(inner_model, 'coef_'):
    scaler = best_model.named_steps['scaler']
    fi = pd.DataFrame({
        'feature'    : FEATURES,
        'coefficient': inner_model.coef_,
        'importance' : np.abs(inner_model.coef_)
    }).sort_values('importance', ascending=False)
    feature_importance = fi.head(15)
    feature_importance.to_csv(
        "data/processed/feature_importance.csv", index=False)

# ════════════════════════════════════════════════════════════
# 4. PREDICCIÓN A FUTURO (6 meses)
# ════════════════════════════════════════════════════════════
print("\n── 4. Predicción a 6 meses ─────────────────────────")

last_row    = df_clean.iloc[-1].copy()
last_prices = list(df_clean['precio'].tail(12).values)
last_oni    = list(df_clean['oni'].tail(6).values)
last_lluvia = list(df_clean['lluvia'].tail(3).values)
last_temp   = list(df_clean['temperatura'].tail(2).values)

future_preds = []

oni_scenarios = {
    'Neutro'       : 0.0,
    'El Niño Mod.' : 1.0,
    'El Niño Fuerte': 2.0,
    'La Niña Mod.' : -1.0,
    'La Niña Fuerte': -2.0,
}

for scenario_name, oni_val in oni_scenarios.items():
    scenario_prices = last_prices.copy()
    scenario_oni    = last_oni.copy()
    scenario_preds  = []

    for step in range(6):
        future_month = (df_clean['month'].iloc[-1] + step) % 12 + 1
        future_year  = df_clean['year'].iloc[-1] + \
                       (df_clean['month'].iloc[-1] + step) // 12
        future_trend = len(df_clean) + step

        row = {
            'precio_lag1'   : scenario_prices[-1],
            'precio_lag2'   : scenario_prices[-2],
            'precio_lag3'   : scenario_prices[-3],
            'precio_lag6'   : scenario_prices[-6],
            'precio_lag12'  : scenario_prices[-12],
            'precio_ma3'    : np.mean(scenario_prices[-3:]),
            'precio_ma6'    : np.mean(scenario_prices[-6:]),
            'precio_ma12'   : np.mean(scenario_prices[-12:]),
            'oni'           : oni_val,
            'oni_lag1'      : scenario_oni[-1],
            'oni_lag2'      : scenario_oni[-2],
            'oni_lag3'      : scenario_oni[-3],
            'oni_ma3'       : np.mean(scenario_oni[-3:]),
            'oni_ma6'       : np.mean(scenario_oni[-6:]),
            'oni_squared'   : oni_val ** 2,
            'mes_sin'       : np.sin(2 * np.pi * future_month / 12),
            'mes_cos'       : np.cos(2 * np.pi * future_month / 12),
            'trend'         : future_trend,
            'lluvia_lag1'   : last_lluvia[-1],
            'lluvia_lag2'   : last_lluvia[-2],
            'temperatura_lag1': last_temp[-1],
            'es_nino'       : 1 if oni_val >= 0.5 else 0,
            'es_nina'       : 1 if oni_val <= -0.5 else 0,
        }

        X_future = pd.DataFrame([row])[FEATURES]
        pred     = best_model.predict(X_future)[0]

        scenario_preds.append({
            'escenario'  : scenario_name,
            'mes'        : step + 1,
            'year'       : future_year,
            'month'      : future_month,
            'precio_pred': round(pred, 0),
            'oni_input'  : oni_val,
        })

        scenario_prices.append(pred)
        scenario_oni.append(oni_val)

    future_preds.extend(scenario_preds)

future_df = pd.DataFrame(future_preds)
future_df.to_csv("data/processed/predicciones_futuras.csv", index=False)

print("  Predicciones a 6 meses por escenario ENSO:")
pivot_future = future_df.pivot(
    index='escenario', columns='mes', values='precio_pred')
print(pivot_future.to_string())

# ════════════════════════════════════════════════════════════
# 5. GUARDAR MODELOS Y RESULTADOS
# ════════════════════════════════════════════════════════════
print("\n── 5. Guardando modelos ────────────────────────────")

# Guardar todos los modelos
for name, pipeline in models.items():
    fname = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('α', 'alpha')
    joblib.dump(pipeline, f"models/{fname}.pkl")

# Resultados comparación
results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ['Modelo','MAE','RMSE','R2','MAPE_pct','CV_MAE','CV_std']
results_df = results_df.sort_values('MAE')
results_df.to_csv("data/processed/model_comparison.csv", index=False)

# Predicciones en test set
test_preds_df = pd.DataFrame({
    'date'  : dates_test.values,
    'actual': y_test.values,
    **{name: predictions[name] for name in predictions}
})
test_preds_df.to_csv("data/processed/test_predictions.csv", index=False)

# Metadata del mejor modelo
metadata = {
    'best_model'    : best_name,
    'features'      : FEATURES,
    'train_size'    : SPLIT,
    'test_size'     : len(X_test),
    'metrics'       : results[best_name],
    'all_results'   : results,
}
with open("data/processed/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Modelos guardados en models/")
print(f"  Comparación guardada en data/processed/model_comparison.csv")

print(f"""
╔══════════════════════════════════════════════════════╗
║           RESUMEN DEL PIPELINE ML                    ║
╠══════════════════════════════════════════════════════╣
║  Modelos entrenados:    {len(models):<30}║
║  Features utilizadas:   {len(FEATURES):<30}║
║  Meses de entrenamiento:{SPLIT:<30}║
║  Meses de test:         {len(X_test):<30}║
║  Mejor modelo:          {best_name:<30}║
║  MAE del mejor:         ${results[best_name]['MAE']:<29,.0f}║
║  MAPE del mejor:        {str(results[best_name]['MAPE_pct'])+'%':<30}║
╚══════════════════════════════════════════════════════╝
""")
print("✅ Pipeline ML completado")