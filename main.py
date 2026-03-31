import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import acf

# Configurarea paginii - trebuie să fie primul apel Streamlit
st.set_page_config(
    page_title="Impactul digitalizării asupra pieței muncii",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizat
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        padding: 2rem 0;
        border-bottom: 3px solid #3b82f6;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f8fafc;
        border-left: 5px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Încărcare date
file_path = Path(__file__).parent / "date" / "BAZADATEnoua.xlsx"

if not file_path.exists():
    st.error(f"Fișierul {file_path.name} nu a fost găsit în folderul 'date'.")
    st.stop()

try:
    df = pd.read_excel(file_path, sheet_name="PANEL")
except Exception as e:
    st.error(f"Eroare la încărcarea fișierului: {e}")
    st.stop()

# Curățare coloane numerice
numeric_columns = [
    "Employment rate%",
    "Internet_Use %",
    "ICT_goods %",
    "HighManu % eports",
    "GDP_CAPITA $",
    "GDP_annual growth %",
    "fixedborad per 100 subscribers",
    "R&D % of gdp",
    "youth employ %"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["An"] = pd.to_numeric(df["An"], errors="coerce")

# Header principal
st.markdown(
    '<h1 class="main-header">💻 Impactul digitalizării asupra pieței muncii</h1>',
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.markdown("### 📚 Despre această aplicație")
st.sidebar.info("""
Această aplicație a fost realizată în cadrul lucrării de licență și are ca scop
analiza **impactului digitalizării asupra pieței muncii**.

Prin intermediul acestei aplicații sunt evidențiate:
- transformările produse de digitalizare
- efectele automatizării asupra locurilor de muncă
- schimbările în competențele cerute pe piața muncii
- relația dintre indicatorii digitali și cei economici
- interpretarea vizuală a datelor prin grafice și tabele
""")

# Conținut principal
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📘 Despre lucrare")

    st.markdown("""
    Această aplicație a fost realizată pentru lucrarea de licență cu tema  
    **„Impactul digitalizării asupra pieței muncii”** și are rolul de a analiza
    relația dintre indicatorii digitalizării și evoluția ocupării forței de muncă
    în **27 de țări**, în perioada **2007–2022**.
    """)

    st.markdown("### 🎯 Obiectivele aplicației")

    features = [
        "Analiza descriptivă a datelor: explorarea indicatorilor economici și digitali pentru fiecare țară și an",
        "Compararea țărilor: identificarea diferențelor dintre state în funcție de nivelul digitalizării și al ocupării forței de muncă",
        "Analiza în dinamică: observarea evoluției indicatorilor în perioada 2007–2022",
        "Evidențierea relațiilor dintre variabile: compararea ocupării forței de muncă cu utilizarea internetului, PIB-ul pe cap de locuitor, cercetarea-dezvoltarea și alți indicatori",
        "Vizualizarea interactivă a datelor: utilizarea de grafice și tabele pentru interpretarea mai ușoară a rezultatelor",
        "Suport pentru cercetarea din lucrarea de licență: aplicația facilitează prezentarea și interpretarea rezultatelor empirice"
    ]

    for feature in features:
        st.markdown(f'<div class="feature-box">{feature}</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Despre baza de date")

    advantages = [
        "🌍 **27 de țări** incluse în analiză",
        "📅 **Perioada 2007–2022**",
        "📑 **Date de tip panel** (țară-an)",
        "💼 **Indicatori ai pieței muncii**",
        "💻 **Indicatori ai digitalizării**",
        "📈 **Indicatori economici complementari**"
    ]

    for advantage in advantages:
        st.markdown(advantage)

    st.markdown("### 🧾 Indicatori analizați")
    st.markdown("""
    - **Employment rate**
    - **Internet use**
    - **ICT goods**
    - **High-tech manufacturing exports**
    - **GDP per capita**
    - **GDP annual growth**
    - **Fixed broadband subscriptions**
    - **R&D expenditure**
    - **Youth employment / unemployment**
    """)

# Secțiunea despre date
st.markdown("---")
st.markdown("## 📁 Descrierea datelor utilizate")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    **🌍 Sursa datelor**
    - Baza de date: **World Bank Open Data**
    - Indicatori economici și sociali standardizați
    - Comparabilitate internațională ridicată
    - Utilizați frecvent în analize econometrice
    """)

with c2:
    st.markdown("""
    **📊 Structura bazei de date**
    - Set de date de tip **panel**
    - Observații pentru **27 de țări**
    - Perioada analizată: **2007–2022**
    - Fiecare observație reprezintă o combinație țară–an
    """)

with c3:
    st.markdown("""
    **📈 Variabile analizate**
    - indicatori ai pieței muncii
    - indicatori ai digitalizării
    - indicatori economici
    - indicatori ai inovării și conectivității
    """)

st.info("""
Datele au fost prelucrate și integrate într-o bază de date unificată, pentru a permite
analiza comparativă și evaluarea impactului digitalizării asupra pieței muncii.
""")

tari_analiza = [
    "Austria", "Belgium", "Bulgaria", "Cyprus", "Czechia",
    "Germany", "Denmark", "Spain", "Estonia", "Finland",
    "France", "Greece", "Croatia", "Hungary", "Ireland",
    "Italy", "Lithuania", "Luxembourg", "Latvia", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania",
    "Slovakia", "Slovenia", "Sweden"
]

st.markdown("### 🌍 Țările incluse în analiză:")

cols = st.columns(3)
for i, tara in enumerate(tari_analiza):
    cols[i % 3].write(f"• {tara}")

countries_used = pd.DataFrame({
    "Tara": [
        "Austria", "Belgium", "Bulgaria", "Cyprus", "Czechia",
        "Germany", "Denmark", "Spain", "Estonia", "Finland",
        "France", "Greece", "Croatia", "Hungary", "Ireland",
        "Italy", "Lithuania", "Luxembourg", "Latvia", "Malta",
        "Netherlands", "Poland", "Portugal", "Romania",
        "Slovakia", "Slovenia", "Sweden"
    ],
    "iso_alpha": [
        "AUT", "BEL", "BGR", "CYP", "CZE",
        "DEU", "DNK", "ESP", "EST", "FIN",
        "FRA", "GRC", "HRV", "HUN", "IRL",
        "ITA", "LTU", "LUX", "LVA", "MLT",
        "NLD", "POL", "PRT", "ROU",
        "SVK", "SVN", "SWE"
    ],
    "Inclusa in analiza": [1] * 27
})

fig_countries = px.choropleth(
    countries_used,
    locations="iso_alpha",
    color="Inclusa in analiza",
    hover_name="Tara",
    locationmode="ISO-3",
    scope="europe",
    color_continuous_scale=["#dbeafe", "#2563eb"],
    title="Cele 27 de țări europene incluse în analiză"
)

fig_countries.update_geos(
    fitbounds="locations",
    showcountries=True,
    showcoastlines=True,
    showland=True,
    landcolor="#f8fafc"
)

fig_countries.update_layout(
    coloraxis_showscale=False,
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)

st.plotly_chart(fig_countries, use_container_width=True, key="plot_harta_tari_analiza")

# Informații bază de date
with st.expander("📋 Informații despre baza de date", expanded=False):
    st.write(f"**Fișier:** {file_path.name} (sheet: PANEL)")
    st.write(f"📊 Număr rânduri: {df.shape[0]}")
    st.write(f"📊 Număr coloane: {df.shape[1]}")
    st.write(f"🌍 Număr țări: {df['Tara'].nunique()}")
    st.write(f"📅 Interval ani: {int(df['An'].min())} - {int(df['An'].max())}")

    if st.checkbox("Previzualizează primele 5 rânduri", key="checkbox_preview_db"):
        st.dataframe(df.head(), use_container_width=True)

# Analiza datelor
st.markdown("---")
st.markdown("## 📊 Analiza datelor")

c1, c2, c3 = st.columns(3)
c1.metric("Număr observații", df.shape[0])
c2.metric("Număr variabile", df.shape[1])
c3.metric("Număr țări", df["Tara"].nunique())

st.subheader("Filtrare date")

tari_disponibile = sorted(df["Tara"].dropna().unique())
ani = sorted(df["An"].dropna().astype(int).unique())

f1, f2 = st.columns(2)

with f1:
    tara_selectata = st.selectbox("Selectează țara", tari_disponibile, key="tara_selectata_main")

with f2:
    interval_ani = st.slider(
        "Selectează intervalul de ani",
        min_value=min(ani),
        max_value=max(ani),
        value=(min(ani), max(ani)),
        key="interval_ani_main"
    )

df_filtrat = df[
    (df["Tara"] == tara_selectata) &
    (df["An"] >= interval_ani[0]) &
    (df["An"] <= interval_ani[1])
]

st.subheader("Date filtrate")
st.dataframe(df_filtrat, use_container_width=True)

st.subheader("Statistici descriptive")
st.dataframe(df[numeric_columns].describe(), use_container_width=True)

st.subheader("Evoluția indicatorilor")

indicator = st.selectbox("Alege indicatorul", numeric_columns, key="indicator_main")

chart_data = df_filtrat[["An", indicator]].dropna().sort_values("An")
chart_data["An"] = chart_data["An"].astype(int).astype(str)

if not chart_data.empty:
    fig_main = px.line(
        chart_data,
        x="An",
        y=indicator,
        title=f"Evoluția indicatorului {indicator}"
    )

    fig_main.update_layout(
        xaxis_title="An",
        yaxis_title=indicator,
        xaxis=dict(type="category")
    )

    st.plotly_chart(fig_main, use_container_width=True, key="plot_evolutie_indicator_main")
else:
    st.warning("Nu există date disponibile pentru indicatorul selectat în intervalul ales.")

# Deschidere dashboard
if "show_app" not in st.session_state:
    st.session_state.show_app = False

st.markdown("---")

if not st.session_state.show_app:
    if st.button("🚀 Deschide analiza datelor", key="btn_show_dashboard"):
        st.session_state.show_app = True

if st.session_state.show_app:
    st.markdown("## 📊 Dashboard analiză")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Analiză indicatori",
        "🌍 Hartă Europa",
        "📊 Corelații",
        "🤖 Machine Learning",
        "🧩 Analiză Cluster"
    ])

    with tab1:
        st.subheader("Analiza indicatorilor")

        st.dataframe(df_filtrat, use_container_width=True)

        indicator_tab = st.selectbox(
            "Alege indicatorul",
            numeric_columns,
            key="indicator_tab"
        )

        chart_data = df_filtrat[["An", indicator_tab]].dropna().sort_values("An")
        chart_data["An"] = chart_data["An"].astype(int).astype(str)

        if not chart_data.empty:
            fig_tab1 = px.line(
                chart_data,
                x="An",
                y=indicator_tab,
                title=f"Evoluția indicatorului {indicator_tab}"
            )

            fig_tab1.update_layout(
                xaxis_title="An",
                yaxis_title=indicator_tab,
                xaxis=dict(type="category")
            )

            st.plotly_chart(fig_tab1, use_container_width=True, key="plot_tab1_indicator")
        else:
            st.warning("Nu există date disponibile pentru indicatorul selectat.")

    with tab2:
        st.subheader("🌍 Hartă Europa")

        year_selected = st.selectbox(
            "Selectează anul",
            sorted(df["An"].dropna().astype(int).unique()),
            key="year_map"
        )

        indicator_map = st.selectbox(
            "Indicator hartă",
            numeric_columns,
            key="indicator_map"
        )

        map_df = df[df["An"] == year_selected].copy()

        fig_tab2 = px.choropleth(
            map_df,
            locations="Tara",
            locationmode="country names",
            color=indicator_map,
            hover_name="Tara",
            title=f"{indicator_map} - {year_selected}",
            color_continuous_scale="Blues"
        )

        fig_tab2.update_geos(
            scope="europe",
            showcountries=True,
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )

        fig_tab2.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )

        st.plotly_chart(fig_tab2, use_container_width=True, key="plot_tab2_harta_europa")

    with tab3:
        st.subheader("Corelații între variabile")

        corr = df[numeric_columns].corr()
        st.dataframe(corr, use_container_width=True)

    with tab4:
        st.subheader("🤖 Machine Learning aplicat pe baza de date")

        st.markdown("""
        În această secțiune sunt testate modele de machine learning pentru a estima
        **rata ocupării forței de muncă** pe baza indicatorilor de digitalizare și a
        variabilelor economice disponibile în baza de date.
        """)

        st.markdown("""
        Modelul încearcă să explice **Employment rate%** pe baza principalilor indicatori
        ai digitalizării și ai contextului economic.
        """)

        st.markdown("### Modele disponibile")
        st.markdown("""
        - **Linear Regression** – pentru relații liniare între variabile
        - **Decision Tree Regressor** – pentru relații neliniare și reguli de decizie
        - **Random Forest Regressor** – pentru predicții mai robuste
        """)

        target = "Employment rate%"

        features_ml = [
            "Internet_Use %",
            "ICT_goods %",
            "HighManu % eports",
            "GDP_CAPITA $",
            "GDP_annual growth %",
            "fixedborad per 100 subscribers",
            "R&D % of gdp",
            "youth employ %"
        ]

        ml_df = df[features_ml + [target]].dropna()  #pastrez doar obs complete

        st.write(f"Număr observații folosite în model: {ml_df.shape[0]}")

        model_name = st.selectbox(
            "Alege modelul de machine learning",
            ["Linear Regression", "Decision Tree", "Random Forest"],
            key="ml_model"
        )

        X = ml_df[features_ml]
        y = ml_df[target]

        X_train, X_test, y_train, y_test = train_test_split(  # set de antrenare si set de test
            X, y, test_size=0.2, random_state=42
        )

        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                random_state=42
            )

        model.fit(X_train, y_train)   # antrenez modelul si obtin valori estimate
        y_pred = model.predict(X_test)

        # metrici de bază
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # MAPE
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

        # MASE
        y_test_reset = y_test.reset_index(drop=True)
        y_pred_series = pd.Series(y_pred).reset_index(drop=True)

        if len(y_test_reset) > 1:
            naive_errors = np.abs(y_test_reset[1:] - y_test_reset[:-1].values)
            model_errors = np.abs(y_test_reset[1:] - y_pred_series[1:])
            mase = model_errors.mean() / naive_errors.mean() if naive_errors.mean() != 0 else np.nan
        else:
            mase = np.nan

        # reziduuri și ACF
        residuals = y_test_reset - y_pred_series #verific corelatia
        nlags = min(10, max(1, len(residuals) - 1))
        acf_values = acf(residuals, nlags=nlags)

        st.markdown("### Performanța modelului")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        c3.metric("R²", f"{r2:.3f}")
        c4.metric("MAPE (%)", f"{mape:.2f}")
        c5.metric("MASE", f"{mase:.3f}" if pd.notna(mase) else "N/A")

        results_df = pd.DataFrame({
            "Valori reale": y_test_reset,
            "Valori estimate": y_pred_series,
            "Reziduuri": residuals
        })

        st.markdown("### Comparație între valori reale și estimate")
        st.dataframe(results_df.head(20), use_container_width=True)

        st.markdown("### Grafic valori reale vs. valori estimate")

        plot_df = results_df.reset_index(drop=True)
        plot_df["Observație"] = plot_df.index.astype(str)

        fig_pred = px.line(
            plot_df,
            x="Observație",
            y=["Valori reale", "Valori estimate"],
            title="Comparație între valorile reale și cele estimate"
        )

        fig_pred.update_layout(
            xaxis_title="Observație",
            yaxis_title="Employment rate%",
            xaxis=dict(type="category")
        )

        st.plotly_chart(fig_pred, use_container_width=True, key="plot_tab4_pred_vs_real")

        st.markdown("### 🔄 Autocorelația reziduurilor (ACF)")

        acf_df = pd.DataFrame({
            "Lag": range(len(acf_values)),
            "ACF": acf_values
        })

        fig_acf = px.bar(
            acf_df,
            x="Lag",
            y="ACF",
            title="ACF pentru reziduuri"
        )

        fig_acf.update_layout(
            xaxis_title="Lag",
            yaxis_title="ACF"
        )

        st.plotly_chart(fig_acf, use_container_width=True, key="plot_tab4_acf")

        if model_name in ["Decision Tree", "Random Forest"]:
            if hasattr(model, "feature_importances_"):
                importance_df = pd.DataFrame({
                    "Variabilă": features_ml,
                    "Importanță": model.feature_importances_
                }).sort_values("Importanță", ascending=False)

                st.markdown("### Importanța variabilelor")
                st.dataframe(importance_df, use_container_width=True)
                st.bar_chart(importance_df.set_index("Variabilă"))

    with tab5:
        st.subheader("🧩 Analiză Cluster (K-Means)")

        st.markdown("""
        În această secțiune este realizată o analiză de tip cluster pentru fiecare an selectat
        (**2007**, **2013**, **2022**), pentru a identifica grupuri de țări cu profiluri similare
        din perspectiva digitalizării.
        """)

        # Fișierele pentru fiecare an
        file_2007 = Path(__file__).parent / "date" / "2007.xlsx"
        file_2013 = Path(__file__).parent / "date" / "2013.xlsx"
        file_2022 = Path(__file__).parent / "date" / "2022.xlsx"

        data_files = {
            "2007": file_2007,
            "2013": file_2013,
            "2022": file_2022
        }

        an_selectat = st.selectbox(
            "Selectează anul pentru analiza cluster",
            ["2007", "2013", "2022"],
            key="cluster_year"
        )

        selected_file = data_files[an_selectat]

        if not selected_file.exists():
            st.error(f"Fișierul pentru anul {an_selectat} nu a fost găsit.")
        else:
            try:
                df_cluster = pd.read_excel(selected_file)
            except Exception as e:
                st.error(f"Eroare la încărcarea fișierului pentru anul {an_selectat}: {e}")
                st.stop()

            # Curățare nume coloane
            df_cluster.columns = (
                df_cluster.columns.astype(str)
                .str.strip()
                .str.replace("\n", " ", regex=False)
                .str.replace("\r", " ", regex=False)
            )

            st.markdown("### Previzualizare date")
            st.dataframe(df_cluster.head(), use_container_width=True)

            # Coloanele reale din fișierele tale
            features_cluster = [
                "frameworks",
                "ictregulatory",
                "internetuse",
                "highTech",
                "fixedbroad",
                "ictgoods"
            ]

            required_cols = ["tara"] + features_cluster
            missing_cols = [col for col in required_cols if col not in df_cluster.columns]

            if missing_cols:
                st.error(f"Lipsesc următoarele coloane din fișierul {an_selectat}: {missing_cols}")
            else:
                # Transformare numerică
                for col in features_cluster:
                    df_cluster[col] = pd.to_numeric(df_cluster[col], errors="coerce")

                # Păstrează doar coloanele necesare
                df_cluster = df_cluster[required_cols].dropna()

                st.write(f"Număr țări incluse în clusterizare: {df_cluster.shape[0]}")

                k = st.slider(
                    "Număr clustere (k)",
                    min_value=2,
                    max_value=6,
                    value=3,
                    key="cluster_k"
                )

                # Standardizare
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_cluster[features_cluster])

                # Model KMeans: min varianta interclasa si max varianta extraclasa
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                df_cluster["Cluster"] = kmeans.fit_predict(X_scaled) + 1

                st.markdown("### Rezultatele clusterizării")
                st.dataframe(df_cluster, use_container_width=True)

                st.markdown("### Distribuția țărilor pe clustere")
                cluster_counts = df_cluster["Cluster"].value_counts().sort_index().reset_index()
                cluster_counts.columns = ["Cluster", "Număr țări"]
                st.dataframe(cluster_counts, use_container_width=True)

                st.bar_chart(cluster_counts.set_index("Cluster"))

                fig_cluster = px.scatter(
                    df_cluster,
                    x="internetuse",
                    y="ictgoods",
                    color=df_cluster["Cluster"].astype(str),
                    hover_name="tara",
                    title=f"Clusterizarea țărilor pentru anul {an_selectat}",
                    labels={
                        "internetuse": "Internet use",
                        "ictgoods": "ICT goods",
                        "color": "Cluster"
                    }
                )

                fig_cluster.update_layout(
                    xaxis_title="Internet use",
                    yaxis_title="ICT goods",
                    legend_title="Cluster"
                )

                st.plotly_chart(fig_cluster, use_container_width=True, key="plot_tab5_cluster")

                st.info("""
                Analiza cluster evidențiază grupuri de țări cu caracteristici similare din perspectiva
                digitalizării. Clusterizarea este realizată pe baza indicatorilor: frameworks,
                ictregulatory, internetuse, highTech, fixedbroad și ictgoods.
                """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background-color: #f8fafc; border-radius: 10px; margin-top: 2rem;">
    <h4>🎓 Facultatea de Cibernetică, Statistică și Informatică Economică</h4>
    <p><strong>Lucrare de licență:</strong> Impactul digitalizării asupra pieței muncii</p>
    <p><strong>Anul:</strong> III | <strong>Specializarea:</strong> Cibernetică Economică</p>
    <p><em>Aplicație interactivă pentru analiza datelor de tip panel utilizând Python și Streamlit</em></p>
</div>
""", unsafe_allow_html=True)