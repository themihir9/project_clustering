import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# ================= PAGE CONFIG =================
st.set_page_config(layout="wide")

# CSS
st.markdown("""
<style>

.scrollable-summary {
    height: 310px;
    overflow-y: auto;
    border: 1px solid #ccc;
    padding: 10px;
    background-color: #f9f9f9;
}
</style>
""", unsafe_allow_html=True)

st.title("🌍 World Development Clustering Analysis")

# ================= LOAD DATA =================
try:
    df = pd.read_csv("World_development_mesurement.csv")
except:
    st.error("CSV file not found ❌")
    st.stop()

# Clean column names
df.columns = df.columns.str.strip()

# Select numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# ================= PREPROCESSING =================
imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(numeric_df)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# ================= MODEL =================
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(data_pca)

# ================= PCA DATA =================
pca_df = pd.DataFrame(data_pca, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = labels

# ================= SIDEBAR =================
st.sidebar.header("Determine Your Cluster")

user_input = {}

for col in numeric_df.columns:
    user_input[col] = st.sidebar.number_input(
        col,
        float(numeric_df[col].mean())
    )

input_df = pd.DataFrame([user_input])[numeric_df.columns]

# Prediction button (SIDEBAR)
if st.sidebar.button("Cluster Me"):

    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    input_pca = pca.transform(input_scaled)

    cluster_id = kmeans.predict(input_pca)[0]

    st.sidebar.success(f"Cluster: {cluster_id}")
    st.success(f"You belong to Cluster {cluster_id}")

# ================= LAYOUT =================
left_col, right_col = st.columns(2)

# ================= VISUALIZATION =================
with left_col:
    st.subheader("📊 Cluster Visualization")

    fig = px.scatter(
        pca_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        title="Clusters Visualized with PCA"
    )

    st.plotly_chart(fig, width='stretch')

# ================= SUMMARY =================
summary_df = numeric_df.copy()
summary_df['Cluster'] = labels
# RIGHT: Summary
with right_col:
    st.markdown('<h2 class="subheader">Cluster Summaries</h2>', unsafe_allow_html=True)

    summary_content = '<div class="scrollable-summary">'

    # Create cluster summary from dataset
    cluster_summary = summary_df.groupby('Cluster').mean()

    for cluster_id, row in cluster_summary.iterrows():
        summary_content += f"<p><strong>Cluster {cluster_id}</strong><br>"
        
        for col, val in row.items():
            summary_content += f"{col}: {round(val, 2)}<br>"
        
        summary_content += "</p><hr>"

    summary_content += "</div>"

    st.markdown(summary_content, unsafe_allow_html=True)

# ================= DATA TABLE =================
st.markdown("---")
st.subheader("📄 Dataset Preview")

df['Cluster'] = labels
st.dataframe(df.head(50))