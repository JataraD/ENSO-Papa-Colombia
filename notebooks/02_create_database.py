# ============================================================
# 02_create_database.py
# Base de datos SQLite — Fenómeno ENSO vs Papa Colombia
# Autor: Juan David Atará Delgado | Mayo 2026
# ============================================================

import pandas as pd
import sqlite3
import os

print("=== CREANDO BASE DE DATOS ENSO-PAPA ===\n")

os.makedirs("data/processed", exist_ok=True)
conn = sqlite3.connect("data/processed/enso_papa_db.sqlite")

# ── Cargar CSVs ──────────────────────────────────────────────
oni      = pd.read_csv("data/raw/oni_index.csv")
clima    = pd.read_csv("data/raw/ideam_clima_regiones.csv")
precios  = pd.read_csv("data/raw/precios_papa.csv")

# ── Cargar tablas ────────────────────────────────────────────
oni.to_sql("oni_index",    conn, if_exists="replace", index=False)
clima.to_sql("clima",      conn, if_exists="replace", index=False)
precios.to_sql("precios",  conn, if_exists="replace", index=False)

# ── Índices ──────────────────────────────────────────────────
conn.executescript("""
    CREATE INDEX IF NOT EXISTS idx_oni_date    ON oni_index(date);
    CREATE INDEX IF NOT EXISTS idx_oni_phase   ON oni_index(phase);
    CREATE INDEX IF NOT EXISTS idx_clima_date  ON clima(date);
    CREATE INDEX IF NOT EXISTS idx_clima_reg   ON clima(region);
    CREATE INDEX IF NOT EXISTS idx_precio_date ON precios(date);
    CREATE INDEX IF NOT EXISTS idx_precio_var  ON precios(variedad);
""")
print("✅ Tablas e índices creados\n")

# ════════════════════════════════════════════════════════════
# QUERIES SQL — DE BÁSICO A AVANZADO
# ════════════════════════════════════════════════════════════

# ── Q1: ¿Cuántos meses de cada fase ENSO hubo? ──────────────
q1 = pd.read_sql("""
    SELECT
        phase                           AS fase_enso,
        COUNT(*)                        AS total_meses,
        ROUND(AVG(oni), 2)              AS oni_promedio,
        MIN(oni)                        AS oni_minimo,
        MAX(oni)                        AS oni_maximo
    FROM oni_index
    GROUP BY phase
    ORDER BY oni_promedio DESC
""", conn)
print("Q1 — Distribución de fases ENSO (2000–2025):")
print(q1.to_string(index=False))
q1.to_csv("data/processed/q1_fases_enso.csv", index=False)

# ── Q2: Precio promedio de papa por fase ENSO ───────────────
q2 = pd.read_sql("""
    SELECT
        p.enso_phase                            AS fase_enso,
        p.variedad,
        COUNT(*)                                AS observaciones,
        ROUND(AVG(p.precio_cop_kg), 0)          AS precio_promedio,
        ROUND(MIN(p.precio_cop_kg), 0)          AS precio_minimo,
        ROUND(MAX(p.precio_cop_kg), 0)          AS precio_maximo,
        ROUND(AVG(p.precio_cop_kg) -
            (SELECT AVG(precio_cop_kg)
             FROM precios), 0)                  AS diferencia_vs_promedio
    FROM precios p
    GROUP BY p.enso_phase, p.variedad
    ORDER BY p.variedad, precio_promedio DESC
""", conn)
print("\nQ2 — Precio promedio por fase ENSO y variedad:")
print(q2.to_string(index=False))
q2.to_csv("data/processed/q2_precio_por_fase.csv", index=False)

# ── Q3: Impacto climático por región y fase ENSO ────────────
q3 = pd.read_sql("""
    SELECT
        c.region,
        c.enso_phase                                AS fase_enso,
        COUNT(*)                                    AS meses,
        ROUND(AVG(c.temperatura_c), 2)              AS temp_promedio,
        ROUND(AVG(c.precipitacion_mm), 1)           AS precip_promedio,
        ROUND(AVG(c.temperatura_c) -
            (SELECT AVG(temperatura_c)
             FROM clima c2
             WHERE c2.region = c.region), 2)        AS anomalia_temp,
        ROUND(AVG(c.precipitacion_mm) -
            (SELECT AVG(precipitacion_mm)
             FROM clima c2
             WHERE c2.region = c.region), 1)        AS anomalia_precip
    FROM clima c
    GROUP BY c.region, c.enso_phase
    ORDER BY c.region, temp_promedio DESC
""", conn)
print("\nQ3 — Anomalías climáticas por región y fase ENSO:")
print(q3.to_string(index=False))
q3.to_csv("data/processed/q3_anomalias_climaticas.csv", index=False)

# ── Q4: Años extremos — El Niño fuerte y su efecto ──────────
q4 = pd.read_sql("""
    SELECT
        o.year,
        ROUND(AVG(o.oni), 2)                        AS oni_anual,
        o.phase                                     AS fase_dominante,
        ROUND(AVG(p.precio_cop_kg), 0)              AS precio_promedio_papa,
        ROUND(AVG(c.temperatura_c), 2)              AS temp_promedio,
        ROUND(AVG(c.precipitacion_mm), 1)           AS precip_promedio
    FROM oni_index o
    JOIN precios p  ON o.year = p.year AND o.month = p.month
    JOIN clima c    ON o.year = c.year AND o.month = c.month
    GROUP BY o.year
    ORDER BY ABS(oni_anual) DESC
    LIMIT 10
""", conn)
print("\nQ4 — Top 10 años más extremos (mayor ONI absoluto):")
print(q4.to_string(index=False))
q4.to_csv("data/processed/q4_anios_extremos.csv", index=False)

# ── Q5: Correlación trimestral precio-clima ─────────────────
q5 = pd.read_sql("""
    SELECT
        o.year,
        CASE
            WHEN o.month IN (1,2,3)   THEN 'Q1 (Ene-Mar)'
            WHEN o.month IN (4,5,6)   THEN 'Q2 (Abr-Jun)'
            WHEN o.month IN (7,8,9)   THEN 'Q3 (Jul-Sep)'
            ELSE                           'Q4 (Oct-Dic)'
        END                                         AS trimestre,
        ROUND(AVG(o.oni), 2)                        AS oni_promedio,
        o.phase                                     AS fase_enso,
        ROUND(AVG(p.precio_cop_kg), 0)              AS precio_papa,
        ROUND(AVG(c.precipitacion_mm), 1)           AS lluvia_mm,
        ROUND(AVG(c.temperatura_c), 2)              AS temperatura_c
    FROM oni_index o
    JOIN precios p ON o.year = p.year AND o.month = p.month
    JOIN clima c   ON o.year = c.year AND o.month = c.month
                   AND c.region = 'Cundinamarca'
                   AND p.variedad = 'Papa Parda Pastusa'
                   AND p.mercado = 'Corabastos Bogotá'
    GROUP BY o.year, trimestre
    ORDER BY o.year, o.month
""", conn)
q5.to_csv("data/processed/q5_serie_trimestral.csv", index=False)
print(f"\nQ5 — Serie trimestral: {len(q5)} trimestres generados")

# ── Q6: Window functions — variación interanual de precios ──
q6 = pd.read_sql("""
    WITH precios_anuales AS (
        SELECT
            year,
            ROUND(AVG(precio_cop_kg), 0)    AS precio_anual,
            enso_phase
        FROM precios
        WHERE variedad = 'Papa Parda Pastusa'
          AND mercado  = 'Corabastos Bogotá'
        GROUP BY year
    )
    SELECT
        year,
        precio_anual,
        enso_phase,
        LAG(precio_anual) OVER (ORDER BY year)  AS precio_anio_anterior,
        ROUND(
            (precio_anual - LAG(precio_anual) OVER (ORDER BY year))
            * 100.0
            / LAG(precio_anual) OVER (ORDER BY year)
        , 1)                                    AS variacion_pct,
        ROUND(AVG(precio_anual)
            OVER (ORDER BY year
                  ROWS BETWEEN 2 PRECEDING
                  AND CURRENT ROW)
        , 0)                                    AS media_movil_3a
    FROM precios_anuales
    ORDER BY year
""", conn)
print("\nQ6 — Variación interanual de precios (window functions):")
print(q6.to_string(index=False))
q6.to_csv("data/processed/q6_variacion_precios.csv", index=False)

# ── Q7: Episodios críticos — El Niño fuerte + precio alto ───
q7 = pd.read_sql("""
    SELECT
        o.year,
        o.month,
        o.oni,
        o.phase,
        ROUND(AVG(p.precio_cop_kg), 0)  AS precio_promedio,
        ROUND(AVG(c.precipitacion_mm), 1) AS lluvia_mm,
        ROUND(AVG(c.temperatura_c), 2)  AS temp_c,
        CASE
            WHEN o.oni >= 1.5
            THEN 'El Niño Fuerte'
            WHEN o.oni <= -1.5
            THEN 'La Niña Fuerte'
            ELSE 'Moderado'
        END                             AS intensidad
    FROM oni_index o
    JOIN precios p ON o.year = p.year AND o.month = p.month
    JOIN clima c   ON o.year = c.year AND o.month = c.month
    WHERE ABS(o.oni) >= 1.5
    GROUP BY o.year, o.month
    ORDER BY ABS(o.oni) DESC
    LIMIT 15
""", conn)
print("\nQ7 — Episodios ENSO extremos (|ONI| ≥ 1.5):")
print(q7.to_string(index=False))
q7.to_csv("data/processed/q7_episodios_extremos.csv", index=False)

conn.close()
print("\n✅ Base de datos creada: data/processed/enso_papa_db.sqlite")
print("✅ 7 queries exportadas en data/processed/")