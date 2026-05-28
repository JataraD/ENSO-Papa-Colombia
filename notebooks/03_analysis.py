# ============================================================
# 03_analysis.py
# Análisis estadístico — ENSO vs Precios Papa Colombia
# Correlaciones, regresiones, visualizaciones
# Autor: Juan David Atará Delgado | Mayo 2026
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import os

print("=== ANÁLISIS ESTADÍSTICO ENSO-PAPA ===\n")

os.makedirs("assets", exist_ok=True)
conn = sqlite3.connect("data/processed/enso_papa_db.sqlite")

# ── Cargar datos procesados ──────────────────────────────────
serie    = pd.read_csv("data/processed/q5_serie_trimestral.csv")
variacion = pd.read_csv("data/processed/q6_variacion_precios.csv")
extremos  = pd.read_csv("data/processed/q7_episodios_extremos.csv")
fases     = pd.read_csv("data/processed/q2_precio_por_fase.csv")
anomalias = pd.read_csv("data/processed/q3_anomalias_climaticas.csv")

# ── Serie anual completa ─────────────────────────────────────
anual = pd.read_sql("""
    SELECT
        o.year,
        ROUND(AVG(o.oni), 3)           AS oni,
        o.phase                        AS fase,
        ROUND(AVG(p.precio_cop_kg), 0) AS precio,
        ROUND(AVG(c.precipitacion_mm), 1) AS lluvia,
        ROUND(AVG(c.temperatura_c), 2) AS temperatura
    FROM oni_index o
    JOIN precios p ON o.year = p.year AND o.month = p.month
    JOIN clima c   ON o.year = c.year AND o.month = c.month
    WHERE p.variedad = 'Papa Parda Pastusa'
      AND p.mercado  = 'Corabastos Bogotá'
      AND c.region   = 'Cundinamarca'
    GROUP BY o.year
    ORDER BY o.year
""", conn)
conn.close()

# ════════════════════════════════════════════════════════════
# ANÁLISIS 1 — CORRELACIÓN ONI vs PRECIO
# ════════════════════════════════════════════════════════════
print("── Análisis 1: Correlación ONI vs Precio ───────────")

r, p_val = stats.pearsonr(anual['oni'], anual['precio'])
r_spear, p_spear = stats.spearmanr(anual['oni'], anual['precio'])

print(f"  Pearson r:    {r:.4f}  (p={p_val:.4f})")
print(f"  Spearman rho: {r_spear:.4f}  (p={p_spear:.4f})")
print(f"  Interpretación: {'Correlación significativa' if p_val < 0.05 else 'No significativa'}")

# Regresión lineal
slope, intercept, r_sq, p_reg, se = stats.linregress(anual['oni'], anual['precio'])
print(f"  Regresión: precio = {slope:.1f} × ONI + {intercept:.1f}")
print(f"  R²: {r_sq**2:.4f}")

# ════════════════════════════════════════════════════════════
# ANÁLISIS 2 — PRECIO PROMEDIO POR FASE
# ════════════════════════════════════════════════════════════
print("\n── Análisis 2: Precio por fase ENSO ────────────────")

precio_neutro = anual[anual['fase']=='Neutro']['precio'].mean()
precio_nino   = anual[anual['fase']=='El Niño']['precio'].mean()
precio_nina   = anual[anual['fase']=='La Niña']['precio'].mean()

print(f"  Precio promedio Neutro:   ${precio_neutro:,.0f} COP/kg")
print(f"  Precio promedio El Niño:  ${precio_nino:,.0f} COP/kg "
      f"({(precio_nino/precio_neutro-1)*100:+.1f}% vs neutro)")
print(f"  Precio promedio La Niña:  ${precio_nina:,.0f} COP/kg "
      f"({(precio_nina/precio_neutro-1)*100:+.1f}% vs neutro)")

# ANOVA
grupos = [
    anual[anual['fase']=='El Niño']['precio'].values,
    anual[anual['fase']=='La Niña']['precio'].values,
    anual[anual['fase']=='Neutro']['precio'].values,
]
grupos = [g for g in grupos if len(g) > 0]
if len(grupos) >= 2:
    f_stat, p_anova = stats.f_oneway(*grupos)
    print(f"  ANOVA F={f_stat:.3f}, p={p_anova:.4f} "
          f"({'Diferencias significativas' if p_anova < 0.05 else 'No significativo'})")

# ════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ════════════════════════════════════════════════════════════
DARK  = "#0E1117"
MID   = "#1A1A2E"
BLUE  = "#3498DB"
RED   = "#E74C3C"
GREEN = "#2ECC71"
GOLD  = "#F39C12"
WHITE = "#FFFFFF"
GRAY  = "#7F8C8D"

phase_colors = {'El Niño': RED, 'La Niña': BLUE, 'Neutro': GRAY}

# ── FIGURA 1: Serie temporal ONI + Precios ──────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                sharex=True, facecolor=DARK)
fig.suptitle("Fenómeno ENSO y Precios de Papa en Colombia (2000–2025)",
             color=WHITE, fontsize=14, fontweight='bold', y=0.98)

# Panel 1 — ONI
ax1.set_facecolor(MID)
ax1.fill_between(anual['year'], anual['oni'],
                  where=anual['oni'] >= 0.5,
                  color=RED, alpha=0.6, label='El Niño')
ax1.fill_between(anual['year'], anual['oni'],
                  where=anual['oni'] <= -0.5,
                  color=BLUE, alpha=0.6, label='La Niña')
ax1.fill_between(anual['year'], anual['oni'],
                  where=(anual['oni'] > -0.5) & (anual['oni'] < 0.5),
                  color=GRAY, alpha=0.3, label='Neutro')
ax1.plot(anual['year'], anual['oni'],
         color=WHITE, linewidth=1.5, zorder=3)
ax1.axhline(y=0.5,  color=RED,  linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=-0.5, color=BLUE, linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=0,    color=WHITE, linestyle='-',  alpha=0.2, linewidth=0.5)
ax1.set_ylabel("Índice ONI (°C)", color=WHITE)
ax1.tick_params(colors=WHITE)
ax1.legend(facecolor=MID, labelcolor=WHITE, fontsize=9, loc='upper right')
ax1.set_title("Índice ONI — Oceanic Niño Index (NOAA)",
              color=GRAY, fontsize=10, loc='left')
for spine in ax1.spines.values():
    spine.set_color("#333")

# Anotar eventos clave
eventos = {
    2002: "El Niño\n2002-03", 2010: "La Niña\n2010-11",
    2015: "El Niño\n2015-16\n(Fuerte)", 2022: "La Niña\n2022-23"
}
for yr, label in eventos.items():
    if yr in anual['year'].values:
        oni_v = anual[anual['year']==yr]['oni'].values[0]
        ax1.annotate(label, xy=(yr, oni_v),
                    xytext=(yr, oni_v + (0.8 if oni_v > 0 else -0.8)),
                    color=WHITE, fontsize=7, ha='center',
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8))

# Panel 2 — Precios
ax2.set_facecolor(MID)
colors_by_phase = [phase_colors.get(f, GRAY) for f in anual['fase']]
ax2.bar(anual['year'], anual['precio'],
        color=colors_by_phase, alpha=0.8, width=0.7)
ax2.plot(anual['year'],
         anual['precio'].rolling(3, center=True).mean(),
         color=GOLD, linewidth=2.5, label='Media móvil 3 años', zorder=3)
ax2.set_ylabel("Precio COP/kg", color=WHITE)
ax2.set_xlabel("Año", color=WHITE)
ax2.tick_params(colors=WHITE)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
patches = [mpatches.Patch(color=c, label=f)
           for f, c in phase_colors.items()]
ax2.legend(handles=patches + [plt.Line2D([0],[0], color=GOLD,
           linewidth=2, label='Media móvil 3a')],
           facecolor=MID, labelcolor=WHITE, fontsize=9)
ax2.set_title("Precio Papa Parda Pastusa — Corabastos Bogotá",
              color=GRAY, fontsize=10, loc='left')
for spine in ax2.spines.values():
    spine.set_color("#333")

plt.tight_layout()
plt.savefig("assets/01_serie_temporal.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("\n✅ Figura 1 guardada: serie temporal")

# ── FIGURA 2: Scatter ONI vs Precio + Regresión ─────────────
fig, ax = plt.subplots(figsize=(10, 7), facecolor=DARK)
ax.set_facecolor(MID)

for fase, color in phase_colors.items():
    mask = anual['fase'] == fase
    ax.scatter(anual[mask]['oni'], anual[mask]['precio'],
               color=color, s=100, alpha=0.85,
               label=fase, zorder=3, edgecolors=WHITE, linewidths=0.5)
    for _, row in anual[mask].iterrows():
        ax.annotate(str(int(row['year'])),
                   (row['oni'], row['precio']),
                   textcoords="offset points", xytext=(5, 4),
                   fontsize=7, color=WHITE, alpha=0.7)

# Línea de regresión
x_line = np.linspace(anual['oni'].min(), anual['oni'].max(), 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color=GOLD, linewidth=2,
        linestyle='--', label=f'Regresión (r={r:.2f})', zorder=4)

ax.axvline(x=0.5,  color=RED,  linestyle=':', alpha=0.5)
ax.axvline(x=-0.5, color=BLUE, linestyle=':', alpha=0.5)
ax.axvline(x=0,    color=WHITE, linestyle='-', alpha=0.15)

ax.set_xlabel("Índice ONI (°C)", color=WHITE, fontsize=12)
ax.set_ylabel("Precio Papa (COP/kg)", color=WHITE, fontsize=12)
ax.set_title("Correlación ONI vs Precio de Papa\nColombia 2000–2025",
             color=WHITE, fontsize=13, fontweight='bold')
ax.tick_params(colors=WHITE)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.legend(facecolor=MID, labelcolor=WHITE, fontsize=10)
ax.text(0.05, 0.95,
        f"Pearson r = {r:.3f}\nSpearman ρ = {r_spear:.3f}\n"
        f"R² = {r_sq**2:.3f}\np = {p_val:.4f}",
        transform=ax.transAxes, color=WHITE, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor=DARK, alpha=0.8))
for spine in ax.spines.values():
    spine.set_color("#333")

plt.tight_layout()
plt.savefig("assets/02_correlacion_oni_precio.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("✅ Figura 2 guardada: correlación ONI-precio")

# ── FIGURA 3: Box plot precios por fase ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK)

# Box plot por fase ENSO
data_box = [anual[anual['fase']==f]['precio'].values
            for f in ['El Niño', 'Neutro', 'La Niña']]
bp = axes[0].boxplot(data_box, patch_artist=True,
                      labels=['El Niño', 'Neutro', 'La Niña'])
for patch, color in zip(bp['boxes'], [RED, GRAY, BLUE]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers','caps','medians','fliers']:
    for item in bp[element]:
        item.set_color(WHITE)
axes[0].set_facecolor(MID)
axes[0].tick_params(colors=WHITE)
axes[0].set_ylabel("Precio COP/kg", color=WHITE)
axes[0].set_title("Distribución de Precios por\nFase ENSO",
                   color=WHITE, fontsize=11, fontweight='bold')
axes[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
for spine in axes[0].spines.values():
    spine.set_color("#333")

# Anomalías climáticas por región
reg_data = anomalias[anomalias['fase_enso'] != 'Neutro'].copy()
x = np.arange(len(reg_data['region'].unique()))
width = 0.35
regions = reg_data['region'].unique()

nino_data = reg_data[reg_data['fase_enso']=='El Niño']['anomalia_precip'].values
nina_data = reg_data[reg_data['fase_enso']=='La Niña']['anomalia_precip'].values

if len(nino_data) == len(regions) and len(nina_data) == len(regions):
    b1 = axes[1].bar(x - width/2, nino_data, width,
                     label='El Niño', color=RED, alpha=0.8)
    b2 = axes[1].bar(x + width/2, nina_data, width,
                     label='La Niña', color=BLUE, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regions, color=WHITE)
    axes[1].axhline(y=0, color=WHITE, linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Anomalía Precipitación (mm)", color=WHITE)
    axes[1].set_title("Anomalía de Lluvia por Región\ny Fase ENSO",
                       color=WHITE, fontsize=11, fontweight='bold')
    axes[1].legend(facecolor=MID, labelcolor=WHITE)
    axes[1].set_facecolor(MID)
    axes[1].tick_params(colors=WHITE)
    for spine in axes[1].spines.values():
        spine.set_color("#333")

plt.tight_layout()
plt.savefig("assets/03_distribucion_fases.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("✅ Figura 3 guardada: distribución por fases")

# ── FIGURA 4: Heatmap — ONI vs Precio por año y mes ─────────
pivot_data = pd.read_sql("""
    SELECT
        o.year,
        o.month,
        ROUND(AVG(p.precio_cop_kg), 0) AS precio
    FROM oni_index o
    JOIN precios p ON o.year = p.year AND o.month = p.month
    WHERE p.variedad = 'Papa Parda Pastusa'
      AND p.mercado  = 'Corabastos Bogotá'
    GROUP BY o.year, o.month
""", sqlite3.connect("data/processed/enso_papa_db.sqlite"))

pivot = pivot_data.pivot(index='month', columns='year', values='precio')

fig, ax = plt.subplots(figsize=(18, 6), facecolor=DARK)
ax.set_facecolor(MID)

im = ax.imshow(pivot.values, cmap='RdYlGn_r',
               aspect='auto', interpolation='nearest')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha='right',
                    color=WHITE, fontsize=8)
ax.set_yticks(range(12))
ax.set_yticklabels(['Ene','Feb','Mar','Abr','May','Jun',
                     'Jul','Ago','Sep','Oct','Nov','Dic'],
                    color=WHITE)
ax.set_title("Heatmap — Precio Papa (COP/kg) por Año y Mes\n"
             "Corabastos Bogotá | Papa Parda Pastusa",
             color=WHITE, fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.ax.yaxis.set_tick_params(color=WHITE)
cbar.set_label("COP/kg", color=WHITE)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)

plt.tight_layout()
plt.savefig("assets/04_heatmap_precios.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("✅ Figura 4 guardada: heatmap precios")

# ── RESUMEN ESTADÍSTICO ──────────────────────────────────────
import json
resultados = {
    "pearson_r"            : round(float(r), 4),
    "pearson_p"            : round(float(p_val), 4),
    "spearman_rho"         : round(float(r_spear), 4),
    "r_squared"            : round(float(r_sq**2), 4),
    "precio_promedio_nino" : round(float(precio_nino), 0),
    "precio_promedio_nina" : round(float(precio_nina), 0),
    "precio_promedio_neutro": round(float(precio_neutro), 0),
    "diferencia_nino_pct"  : round(float((precio_nino/precio_neutro-1)*100), 1),
    "diferencia_nina_pct"  : round(float((precio_nina/precio_neutro-1)*100), 1),
}
with open("data/processed/resultados_estadisticos.json", "w") as f:
    json.dump(resultados, f, indent=2)

print(f"""
╔══════════════════════════════════════════╗
║      RESULTADOS ESTADÍSTICOS CLAVE       ║
╠══════════════════════════════════════════╣
║  Pearson r:        {r:<24.4f}║
║  Spearman rho:     {r_spear:<24.4f}║
║  R²:               {r_sq**2:<24.4f}║
║  El Niño vs neutro: {(precio_nino/precio_neutro-1)*100:+.1f}%{'':<17}║
║  La Niña vs neutro: {(precio_nina/precio_neutro-1)*100:+.1f}%{'':<17}║
╚══════════════════════════════════════════╝
""")
print("✅ Análisis completado — assets/ y data/processed/ actualizados")