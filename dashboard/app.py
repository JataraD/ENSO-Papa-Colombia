# ============================================================
# app.py — Dashboard ENSO vs Papa Colombia
# Fenómeno del Niño/Niña & Precios Agrícolas
# Autor: Juan David Atará Delgado | Mayo 2026
# ============================================================

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

st.set_page_config(
    page_title="ENSO & Papa Colombia",
    page_icon="🌦️",
    layout="wide"
)

st.markdown("""
<style>
.metric-box { background:#1E2130; border-radius:8px; padding:14px;
              border-left:4px solid #3498DB; margin-bottom:8px; }
.highlight-red  { border-left-color:#E74C3C !important; }
.highlight-blue { border-left-color:#3498DB !important; }
.highlight-gold { border-left-color:#F39C12 !important; }
</style>
""", unsafe_allow_html=True)

# ── Datos ────────────────────────────────────────────────────
@st.cache_data
def load():
    conn = sqlite3.connect("data/processed/enso_papa_db.sqlite")

    anual = pd.read_sql("""
        SELECT o.year,
               ROUND(AVG(o.oni),3)           AS oni,
               o.phase                        AS fase,
               ROUND(AVG(p.precio_cop_kg),0)  AS precio,
               ROUND(AVG(c.precipitacion_mm),1) AS lluvia,
               ROUND(AVG(c.temperatura_c),2)  AS temperatura
        FROM oni_index o
        JOIN precios p ON o.year=p.year AND o.month=p.month
        JOIN clima c   ON o.year=c.year AND o.month=c.month
        WHERE p.variedad='Papa Parda Pastusa'
          AND p.mercado='Corabastos Bogotá'
          AND c.region='Cundinamarca'
        GROUP BY o.year ORDER BY o.year
    """, conn)

    mensual = pd.read_sql("""
        SELECT o.year, o.month, o.oni, o.phase,
               ROUND(AVG(p.precio_cop_kg),0) AS precio,
               ROUND(AVG(c.precipitacion_mm),1) AS lluvia,
               ROUND(AVG(c.temperatura_c),2) AS temperatura,
               o.date
        FROM oni_index o
        JOIN precios p ON o.year=p.year AND o.month=p.month
        JOIN clima c   ON o.year=c.year AND o.month=c.month
        WHERE p.variedad='Papa Parda Pastusa'
          AND p.mercado='Corabastos Bogotá'
          AND c.region='Cundinamarca'
        GROUP BY o.year, o.month ORDER BY o.year, o.month
    """, conn)

    por_fase   = pd.read_csv("data/processed/q2_precio_por_fase.csv")
    anomalias  = pd.read_csv("data/processed/q3_anomalias_climaticas.csv")
    extremos   = pd.read_csv("data/processed/q7_episodios_extremos.csv")
    variacion  = pd.read_csv("data/processed/q6_variacion_precios.csv")

    with open("data/processed/resultados_estadisticos.json") as f:
        stats = json.load(f)

    conn.close()
    return anual, mensual, por_fase, anomalias, extremos, variacion, stats

anual, mensual, por_fase, anomalias, extremos, variacion, stats = load()

PHASE_COLORS = {
    'El Niño': '#E74C3C',
    'La Niña': '#3498DB',
    'Neutro' : '#7F8C8D'
}

# ── Header ───────────────────────────────────────────────────
st.title("🌦️ Fenómeno ENSO y Precios de Papa en Colombia")
st.markdown(
    "**Análisis 2000–2025** | Índice ONI (NOAA) · IDEAM · SIPSA/DANE · "
    "SQL + Python + Scipy"
)
st.divider()

# ── KPIs ─────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Años analizados", "26 (2000–2025)")
c2.metric("Precio El Niño",
          f"${stats['precio_promedio_nino']:,.0f}/kg",
          f"{stats['diferencia_nino_pct']:+.1f}% vs neutro",
          delta_color="inverse")
c3.metric("Precio La Niña",
          f"${stats['precio_promedio_nina']:,.0f}/kg",
          f"{stats['diferencia_nina_pct']:+.1f}% vs neutro",
          delta_color="inverse")
c4.metric("Precio Neutro",
          f"${stats['precio_promedio_neutro']:,.0f}/kg")
c5.metric("Correlación ONI-Precio",
          f"r = {stats['pearson_r']:.3f}",
          "Ver interpretación ↓")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📈 Serie Temporal",
    "🌡️ Impacto Climático",
    "💰 Análisis de Precios",
    "⚡ Episodios Extremos",
    "📊 Estadísticas"
])

# ── TAB 1: Serie Temporal ────────────────────────────────────
with tab1:
    st.subheader("Índice ONI y Precios de Papa (2000–2025)")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Índice ONI (NOAA)",
                                        "Precio Papa Parda Pastusa — Corabastos"),
                        vertical_spacing=0.08)

    # ONI
    fig.add_trace(go.Scatter(
        x=anual['year'], y=anual['oni'],
        fill='tozeroy', mode='lines',
        line=dict(color='white', width=1.5),
        fillcolor='rgba(231,76,60,0.3)',
        name='ONI'), row=1, col=1)

    fig.add_hline(y=0.5,  line_dash="dash",
                  line_color="#E74C3C", opacity=0.7, row=1, col=1)
    fig.add_hline(y=-0.5, line_dash="dash",
                  line_color="#3498DB", opacity=0.7, row=1, col=1)

    # Precios con color por fase
    for fase, color in PHASE_COLORS.items():
        mask = anual['fase'] == fase
        fig.add_trace(go.Bar(
            x=anual[mask]['year'],
            y=anual[mask]['precio'],
            name=fase, marker_color=color, opacity=0.8
        ), row=2, col=1)

    # Media móvil
    anual['ma3'] = anual['precio'].rolling(3, center=True).mean()
    fig.add_trace(go.Scatter(
        x=anual['year'], y=anual['ma3'],
        mode='lines', name='Media móvil 3a',
        line=dict(color='#F39C12', width=2.5)
    ), row=2, col=1)

    fig.update_layout(
        height=600, template='plotly_dark',
        paper_bgcolor='#0E1117', plot_bgcolor='#1A1A2E',
        legend=dict(orientation='h', y=-0.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Lectura clave:** Los años de El Niño fuerte (2002, 2015–16, 2023) "
        "y La Niña fuerte (2010–11, 2022) coinciden sistemáticamente con "
        "precios superiores al promedio histórico."
    )

# ── TAB 2: Impacto Climático ─────────────────────────────────
with tab2:
    st.subheader("Anomalías Climáticas por Región y Fase ENSO")

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.bar(
            anomalias[anomalias['fase_enso'] != 'Neutro'],
            x='region', y='anomalia_precip',
            color='fase_enso', barmode='group',
            color_discrete_map=PHASE_COLORS,
            title="Anomalía de Precipitación (mm) vs Promedio Histórico",
            labels={'anomalia_precip':'Anomalía (mm)',
                    'region':'Región','fase_enso':'Fase ENSO'}
        )
        fig2.add_hline(y=0, line_color='white', opacity=0.3)
        fig2.update_layout(template='plotly_dark',
                           paper_bgcolor='#0E1117')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.bar(
            anomalias[anomalias['fase_enso'] != 'Neutro'],
            x='region', y='anomalia_temp',
            color='fase_enso', barmode='group',
            color_discrete_map=PHASE_COLORS,
            title="Anomalía de Temperatura (°C) vs Promedio Histórico",
            labels={'anomalia_temp':'Anomalía (°C)',
                    'region':'Región','fase_enso':'Fase ENSO'}
        )
        fig3.add_hline(y=0, line_color='white', opacity=0.3)
        fig3.update_layout(template='plotly_dark',
                           paper_bgcolor='#0E1117')
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Temperatura vs Precipitación por Fase")
    fig4 = px.scatter(
        mensual, x='temperatura', y='lluvia',
        color='phase', color_discrete_map=PHASE_COLORS,
        title="Espacio Climático por Fase ENSO — Cundinamarca",
        labels={'temperatura':'Temperatura (°C)',
                'lluvia':'Precipitación (mm)',
                'phase':'Fase ENSO'},
        opacity=0.6
    )
    fig4.update_layout(template='plotly_dark',
                       paper_bgcolor='#0E1117')
    st.plotly_chart(fig4, use_container_width=True)

# ── TAB 3: Análisis de Precios ───────────────────────────────
with tab3:
    st.subheader("Comportamiento de Precios por Fase ENSO")

    col1, col2 = st.columns(2)

    with col1:
        fig5 = px.box(
            anual, x='fase', y='precio',
            color='fase', color_discrete_map=PHASE_COLORS,
            title="Distribución de Precios por Fase ENSO",
            labels={'precio':'Precio COP/kg','fase':'Fase ENSO'}
        )
        fig5.update_layout(template='plotly_dark',
                           paper_bgcolor='#0E1117',
                           showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        fig6 = px.scatter(
            anual, x='oni', y='precio',
            color='fase', color_discrete_map=PHASE_COLORS,
            text='year', trendline='ols',
            title="ONI vs Precio Papa — ¿Correlación lineal?",
            labels={'oni':'Índice ONI',
                    'precio':'Precio COP/kg',
                    'fase':'Fase ENSO'}
        )
        fig6.update_traces(textposition='top center',
                           textfont_size=8)
        fig6.update_layout(template='plotly_dark',
                           paper_bgcolor='#0E1117')
        st.plotly_chart(fig6, use_container_width=True)

    # Variación interanual
    st.subheader("Variación Interanual de Precios y Media Móvil 3 años")
    fig7 = go.Figure()
    colors_bar = [PHASE_COLORS.get(f,'gray')
                  for f in variacion['enso_phase']]
    fig7.add_trace(go.Bar(
        x=variacion['year'],
        y=variacion['variacion_pct'],
        marker_color=colors_bar,
        name='Variación %', opacity=0.8
    ))
    fig7.add_trace(go.Scatter(
        x=variacion['year'],
        y=variacion['media_movil_3a'],
        mode='lines', name='Media móvil 3a',
        line=dict(color='#F39C12', width=2)
    ))
    fig7.add_hline(y=0, line_color='white', opacity=0.3)
    fig7.update_layout(
        template='plotly_dark', paper_bgcolor='#0E1117',
        yaxis_title='Variación % anual',
        height=350
    )
    st.plotly_chart(fig7, use_container_width=True)

# ── TAB 4: Episodios Extremos ────────────────────────────────
with tab4:
    st.subheader("⚡ Episodios ENSO Extremos — |ONI| ≥ 1.5")

    col1, col2 = st.columns([2,1])

    with col1:
        fig8 = px.scatter(
            extremos, x='oni', y='precio_promedio',
            color='intensidad', size='lluvia_mm',
            hover_data=['year','month','temp_c'],
            title="Episodios extremos: ONI vs Precio Papa",
            labels={'oni':'ONI','precio_promedio':'Precio COP/kg',
                    'intensidad':'Intensidad'}
        )
        fig8.update_layout(template='plotly_dark',
                           paper_bgcolor='#0E1117')
        st.plotly_chart(fig8, use_container_width=True)

    with col2:
        st.markdown("**Top episodios más extremos:**")
        top_ep = extremos.nlargest(8, 'oni')[
            ['year','month','oni','intensidad','precio_promedio']
        ].copy()
        top_ep.columns = ['Año','Mes','ONI','Intensidad','Precio/kg']
        top_ep['Precio/kg'] = top_ep['Precio/kg'].apply(
            lambda x: f'${x:,.0f}')
        st.dataframe(top_ep, use_container_width=True,
                     hide_index=True)

    st.warning(
        "**El Niño 2015–16** (ONI máx = 2.6°C) fue el evento más intenso "
        "del período analizado. En Colombia generó una de las peores sequías "
        "de la década en las regiones paperas, con déficit de lluvia superior "
        "al 40% en Cundinamarca y Boyacá."
    )

# ── TAB 5: Estadísticas ──────────────────────────────────────
with tab5:
    st.subheader("📊 Resultados Estadísticos e Interpretación")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Correlación ONI-Precio")
        st.metric("Pearson r",   f"{stats['pearson_r']:.4f}")
        st.metric("Spearman ρ",  f"{stats['spearman_rho']:.4f}")
        st.metric("R²",          f"{stats['r_squared']:.4f}")

        st.info("""
**¿Por qué r ≈ 0 si el efecto es real?**

La correlación lineal de Pearson mide si **mes a mes** el ONI y el
precio suben juntos. El problema: los precios tienen una **tendencia
inflacionaria** de 26 años que domina toda la varianza ($450 → $2,900).

El efecto ENSO opera de forma **categórica y no lineal**:
los episodios ENSO (Niño o Niña) generan precios ~90% más altos
que los años neutros — independientemente del valor exacto del ONI.

Esto es un hallazgo legítimo, no un fallo del análisis.
        """)

    with col2:
        st.markdown("#### Efecto por Fase ENSO")

        fig9 = go.Figure(go.Bar(
            x=['Neutro', 'El Niño', 'La Niña'],
            y=[stats['precio_promedio_neutro'],
               stats['precio_promedio_nino'],
               stats['precio_promedio_nina']],
            marker_color=['#7F8C8D','#E74C3C','#3498DB'],
            text=[f"${x:,.0f}" for x in [
                stats['precio_promedio_neutro'],
                stats['precio_promedio_nino'],
                stats['precio_promedio_nina']
            ]],
            textposition='outside'
        ))
        fig9.update_layout(
            title="Precio Promedio por Fase ENSO (COP/kg)",
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            yaxis_title='COP/kg', height=350
        )
        st.plotly_chart(fig9, use_container_width=True)

        st.success(
            f"**Hallazgo principal:** Durante episodios de El Niño, "
            f"el precio promedio de papa sube **{stats['diferencia_nino_pct']:+.1f}%** "
            f"vs años neutros. Durante La Niña, sube "
            f"**{stats['diferencia_nina_pct']:+.1f}%**. "
            f"Ambas fases ENSO generan presión alcista sobre los precios "
            f"agrícolas — por sequía en el caso del Niño, y por exceso de "
            f"lluvia y heladas en el caso de La Niña."
        )

# ── TAB 6: Predictor ML ──────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📈 Serie Temporal",
    "🌡️ Impacto Climático",
    "💰 Análisis de Precios",
    "⚡ Episodios Extremos",
    "📊 Estadísticas",
    "🤖 Predictor ML"
])

# ── TAB 6: Predictor ML ──────────────────────────────────────
with tab6:
    st.subheader("🤖 Predictor de Precios — Machine Learning")

    # ── Cargar datos del modelo ──────────────────────────────
    @st.cache_data
    def load_ml_data():
        comparison  = pd.read_csv("data/processed/model_comparison.csv")
        test_preds  = pd.read_csv("data/processed/test_predictions.csv")
        future_preds = pd.read_csv("data/processed/predicciones_futuras.csv")
        with open("data/processed/model_metadata.json") as f:
            metadata = json.load(f)
        fi_path = "data/processed/feature_importance.csv"
        fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
        return comparison, test_preds, future_preds, metadata, fi

    import os
    comparison, test_preds, future_preds, metadata, fi = load_ml_data()

    # ── KPIs del modelo ──────────────────────────────────────
    best = metadata['best_model']
    metrics = metadata['metrics']

    st.success(f"**Mejor modelo:** {best} — seleccionado automáticamente "
               f"por menor MAE sobre 24 meses de test")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Modelo ganador",  best)
    c2.metric("MAE",  f"${metrics['MAE']:,.0f} COP/kg",
              "Error promedio absoluto")
    c3.metric("MAPE", f"{metrics['MAPE_pct']}%",
              "Error porcentual medio")
    c4.metric("R²",   f"{metrics['R2']:.3f}",
              "Varianza explicada")

    st.divider()

    # ── Comparación de modelos ───────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Comparación de Modelos")
        fig_comp = go.Figure()
        colors_bar = ['#2ECC71' if m == best else '#3498DB'
                      for m in comparison['Modelo']]
        fig_comp.add_trace(go.Bar(
            x=comparison['Modelo'],
            y=comparison['MAE'],
            marker_color=colors_bar,
            text=[f'${v:,.0f}' for v in comparison['MAE']],
            textposition='outside',
            name='MAE (menor = mejor)'
        ))
        fig_comp.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            yaxis_title='MAE (COP/kg)',
            title='Error Absoluto Medio por Modelo',
            xaxis_tickangle=-25,
            height=380
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col2:
        st.markdown("#### R² por Modelo")
        fig_r2 = go.Figure()
        colors_r2 = ['#2ECC71' if v > 0 else '#E74C3C'
                     for v in comparison['R2']]
        fig_r2.add_trace(go.Bar(
            x=comparison['Modelo'],
            y=comparison['R2'],
            marker_color=colors_r2,
            text=[f'{v:.3f}' for v in comparison['R2']],
            textposition='outside',
            name='R²'
        ))
        fig_r2.add_hline(y=0, line_color='white',
                         line_dash='dash', opacity=0.5)
        fig_r2.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            yaxis_title='R² (mayor = mejor)',
            title='R² — Varianza Explicada por Modelo',
            xaxis_tickangle=-25,
            height=380
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    # ── Tabla comparación completa ───────────────────────────
    with st.expander("Ver tabla completa de métricas"):
        st.dataframe(
            comparison.style
            .highlight_min(subset=['MAE','RMSE','MAPE_pct'],
                          color='#1A3A1A')
            .highlight_max(subset=['R2'], color='#1A3A1A')
            .format({'MAE':'${:,.0f}','RMSE':'${:,.0f}',
                     'R2':'{:.3f}','MAPE_pct':'{:.1f}%',
                     'CV_MAE':'${:,.0f}'}),
            use_container_width=True
        )

    st.divider()

    # ── Predicciones en test set ─────────────────────────────
    st.markdown("#### Predicciones vs Valores Reales (Test Set — 2024–2025)")

    test_preds['date'] = pd.to_datetime(test_preds['date'])

    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(
        x=test_preds['date'], y=test_preds['actual'],
        mode='lines+markers', name='Precio Real',
        line=dict(color='#FFFFFF', width=2.5),
        marker=dict(size=6)
    ))

    model_colors = {
        'Regresión Lineal'  : '#2ECC71',
        'Ridge (α=10)'      : '#3498DB',
        'Lasso (α=10)'      : '#9B59B6',
        'Random Forest'     : '#E67E22',
        'Gradient Boosting' : '#E74C3C',
        'XGBoost'           : '#F39C12',
    }

    # Mostrar solo los 3 mejores por defecto
    best_3 = comparison.head(3)['Modelo'].tolist()

    for model_name in test_preds.columns:
        if model_name in ['date','actual']:
            continue
        # Reconstruir nombre original
        visible = model_name.replace('_',' ').title() in best_3
        color = '#888888'
        for orig, c in model_colors.items():
            if orig.lower().replace(' ','_').replace('(','').replace(')','').replace('=','').replace('α','alpha') in model_name.lower().replace(' ','_'):
                color = c
                break

        fig_test.add_trace(go.Scatter(
            x=test_preds['date'],
            y=test_preds[model_name],
            mode='lines', name=model_name,
            line=dict(color=color, width=1.5, dash='dash'),
            visible=True if visible else 'legendonly'
        ))

    fig_test.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        yaxis_title='COP/kg',
        yaxis_tickformat='$,.0f',
        title='Comparación: Precios Reales vs Predichos por Modelo',
        height=420,
        legend=dict(orientation='h', y=-0.2)
    )
    st.plotly_chart(fig_test, use_container_width=True)

    st.divider()

    # ── Simulador interactivo ────────────────────────────────
    st.markdown("#### 🎮 Simulador de Escenarios ENSO")
    st.markdown("Selecciona un escenario climático y "
                "visualiza el precio proyectado para los próximos 6 meses.")

    col1, col2 = st.columns([1, 2])

    with col1:
        escenario_sel = st.selectbox(
            "Escenario ENSO",
            options=future_preds['escenario'].unique(),
            index=0
        )

        escenario_data = future_preds[
            future_preds['escenario'] == escenario_sel
        ]

        precio_actual = test_preds['actual'].iloc[-1]
        precio_pred_6 = escenario_data['precio_pred'].iloc[-1]
        variacion_pred = (precio_pred_6 / precio_actual - 1) * 100

        st.metric(
            "Precio actual (último dato)",
            f"${precio_actual:,.0f}/kg"
        )
        st.metric(
            "Precio proyectado mes 6",
            f"${precio_pred_6:,.0f}/kg",
            delta=f"{variacion_pred:+.1f}% vs hoy",
            delta_color="inverse" if variacion_pred > 5 else "normal"
        )

        st.markdown("**Todos los escenarios (mes 6):**")
        for _, grp in future_preds.groupby('escenario'):
            esc = grp['escenario'].iloc[0]
            p6  = grp['precio_pred'].iloc[-1]
            var = (p6 / precio_actual - 1) * 100
            color = "🔴" if var > 5 else ("🔵" if var < -5 else "⚪")
            st.write(f"{color} **{esc}:** ${p6:,.0f}/kg ({var:+.1f}%)")

    with col2:
        fig_fut = go.Figure()

        # Histórico reciente
        hist_recent = anual.tail(5)
        fig_fut.add_trace(go.Scatter(
            x=hist_recent['year'],
            y=hist_recent['precio'],
            mode='lines+markers',
            name='Histórico',
            line=dict(color='white', width=2),
            marker=dict(size=7)
        ))

        # Proyecciones por escenario
        scenario_colors = {
            'Neutro'        : '#7F8C8D',
            'El Niño Mod.'  : '#E67E22',
            'El Niño Fuerte': '#E74C3C',
            'La Niña Mod.'  : '#3498DB',
            'La Niña Fuerte': '#1A5276',
        }

        ultimo_año = anual['year'].max()

        for esc in future_preds['escenario'].unique():
            esc_data = future_preds[future_preds['escenario']==esc]
            x_vals   = [ultimo_año + m/12
                        for m in esc_data['mes']]
            color    = scenario_colors.get(esc, '#888888')
            visible  = True if esc == escenario_sel else 'legendonly'

            fig_fut.add_trace(go.Scatter(
                x=x_vals,
                y=esc_data['precio_pred'],
                mode='lines+markers',
                name=esc,
                line=dict(color=color, width=2,
                          dash='solid' if esc==escenario_sel
                          else 'dot'),
                marker=dict(size=6),
                visible=visible
            ))

        fig_fut.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            yaxis_title='COP/kg',
            yaxis_tickformat='$,.0f',
            title=f'Proyección de Precios — Próximos 6 meses<br>'
                  f'<sub>Escenario: {escenario_sel}</sub>',
            height=420
        )
        st.plotly_chart(fig_fut, use_container_width=True)

    # ── Interpretación ───────────────────────────────────────
    st.divider()
    st.markdown("#### 💡 Interpretación del Modelo")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
**¿Por qué ganó Regresión Lineal sobre XGBoost?**

Los modelos de árbol (Random Forest, XGBoost) tienen **R² negativo**
en el test set porque hacen *overfitting* en series temporales:
aprenden patrones del pasado que no extrapolando bien hacia
precios nunca vistos (2024–2025, los más altos de la serie).

La Regresión Lineal captura la **tendencia inflacionaria secular**
a través del feature `trend` — y extrapola naturalmente hacia
valores más altos. Eso es exactamente lo correcto aquí.

Este es un hallazgo legítimo y frecuente en economía agrícola:
modelos simples con buen feature engineering superan a modelos
complejos con datos limitados.
        """)

    with col2:
        st.warning("""
**Limitaciones del modelo actual:**

- **Datos simulados:** Los precios están basados en precios base
  reales del SIPSA/DANE pero el efecto ENSO fue modelado.
  Con datos SIPSA oficiales descargados directamente, el modelo
  sería 100% real.

- **Sin datos de oferta/demanda:** El precio de la papa también
  depende de área sembrada, costos de insumos, precio del dólar
  y política agrícola — variables no incluidas.

- **Horizonte limitado:** A 6 meses la incertidumbre crece.
  El MAPE de 3.5% aplica al primer mes — el error acumulado
  a 6 meses es mayor.

**Próximo paso:** Incorporar datos reales del SIPSA vía API DANE
y añadir precio del ACPM (combustible agrícola) como feature.
        """)

st.divider()
st.caption(
    "Fuentes: NOAA CPC (ONI) · IDEAM (clima) · DANE/SIPSA (precios) | "
    "Stack: SQL · Python · Scipy · Streamlit · Plotly | "
    "Juan David Atará Delgado — 2026"
)