import io
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Road Accident Intelligence", page_icon="🚦", layout="wide")

# Dark, compact dashboard styling inspired by the reference screenshots.
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #101116; color: #f2f3f7; }
[data-testid="stHeader"] { background: rgba(16,17,22,0.95); }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1600px; }
h1, h2, h3 { color: #f4f5fa !important; }
p, label, .stMarkdown { color: #d5d7e0; }
div[data-testid="stMetric"] {
  background: linear-gradient(145deg,#20212b,#191a22);
  border: 1px solid #393b4b; border-radius: 12px; padding: 13px 15px;
}
div[data-testid="stMetricLabel"] { color: #b8bac8 !important; font-size: .78rem; }
div[data-testid="stMetricValue"] { color: #f5f6fb !important; font-weight: 700; }
section[data-testid="stSidebar"] { background: #171820; border-right: 1px solid #2b2d39; }
div[data-testid="stTabs"] button { color: #d8d9e2; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #ff6b78; border-bottom-color: #ff6b78; }
div[data-testid="stAlert"] { background: #202330; }
div[data-testid="stPlotlyChart"] { background: #191a22; border: 1px solid #333543; border-radius: 12px; padding: 8px; }
div[data-testid="stPlotlyChart"] text { fill: #f0f2fa !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #e1e3ed !important; }
[data-testid="stCaptionContainer"] { color: #b9bdcc !important; }
hr { border-color: #343643; }
.small-muted { color:#9da1b4; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

BG = "#191a22"
PLOT_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="Arial, sans-serif", color="#f0f2fa", size=14),
    margin=dict(l=24, r=20, t=42, b=42),
    xaxis=dict(gridcolor="#30323f", zerolinecolor="#30323f",
               tickfont=dict(color="#e5e7f0", size=12), title_font=dict(color="#f0f2fa", size=13)),
    yaxis=dict(gridcolor="#30323f", zerolinecolor="#30323f",
               tickfont=dict(color="#e5e7f0", size=12), title_font=dict(color="#f0f2fa", size=13)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0f2fa", size=13))
)
COL_STATE = "State Name"
COL_CITY = "City Name"
COL_YEAR = "Year"
COL_MONTH = "Month"
COL_DAY = "Day of Week"
COL_TIME = "Time of Day"
COL_SEVERITY = "Accident Severity"
COL_WEATHER = "Weather Conditions"
COL_ROAD = "Road Type"

# Official administrative categories: 28 states and 8 union territories.
INDIAN_STATES = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
}
INDIAN_UTS = {
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir",
    "Ladakh", "Lakshadweep", "Puducherry"
}
STATE_ALIASES = {
    "orissa": "Odisha",
    "pondicherry": "Puducherry",
    "puducherry (pondicherry)": "Puducherry",
    "nct of delhi": "Delhi",
    "national capital territory of delhi": "Delhi",
    "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir (ut)": "Jammu and Kashmir",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "andaman and nicobar": "Andaman and Nicobar Islands",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "uttaranchal": "Uttarakhand",
    "chattisgarh": "Chhattisgarh",
}

def normalize_state_name(value):
    if pd.isna(value):
        return "Unknown"
    name = " ".join(str(value).strip().split())
    return STATE_ALIASES.get(name.casefold(), name)

FEATURES = [
    COL_WEATHER, COL_ROAD, "Road Condition", "Lighting Conditions",
    "Traffic Control Presence", "TimePeriod", "Season",
    "Number of Vehicles Involved", "Vehicle Type Involved",
    "Speed Limit (km/h)", "Driver Age", "Driver Gender",
    "Driver License Status", "Alcohol Involvement",
    "Accident Location Details", COL_STATE
]

def style_fig(fig, height=280):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_layout(
        title_font=dict(color="#f5f6fb", size=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0f2fa", size=13))
    )
    return fig

def count_frame(frame, column, label, limit=None):
    s = frame[column].fillna("Unknown").astype(str).value_counts()
    if limit:
        s = s.head(limit)
    return s.rename_axis(label).reset_index(name="Accidents")

def chart_bar(frame, column, label, limit=10, horizontal=False):
    data = count_frame(frame, column, label, limit)
    if horizontal:
        data = data.sort_values("Accidents", ascending=True)
        fig = px.bar(data, x="Accidents", y=label, orientation="h", color="Accidents",
                     color_continuous_scale=["#3949ab", "#ff6b78"])
    else:
        fig = px.bar(data, x=label, y="Accidents", color="Accidents",
                     color_continuous_scale=["#3949ab", "#ff6b78"])
        fig.update_xaxes(tickangle=-25)
    fig.update_coloraxes(showscale=False)
    return style_fig(fig)

st.title("🚦 Road Accident Intelligence")
st.markdown('<div class="small-muted">INDIA · ACCIDENT PATTERNS · SEVERITY MODELLING · ASSOCIATION MINING</div>', unsafe_allow_html=True)

# Load the project's committed datasets automatically. Visitors do not upload files.
DATA_DIR = Path(__file__).resolve().parent / "data"
def load_project_file(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return io.BytesIO(path.read_bytes())

data_file = load_project_file("cleaned_data.csv")
rules_file = load_project_file("association_rules.csv")
algo_file = load_project_file("algorithm_comparison.csv")
cluster_file = load_project_file("clustering_comparison.csv")
dbscan_file = load_project_file("dbscan_labels.csv")
hier_file = load_project_file("hier_labels.csv")

with st.sidebar:
    st.markdown("## ⚙️ Dashboard controls")
    st.success("Project datasets loaded automatically")
    st.caption("The analysis files are stored with the website. No CSV upload is required.")
    st.markdown("---")
    st.markdown("### About")
    st.caption("Public road accident analysis dashboard. Figures are based on the project's stored academic dataset.")

if data_file is None:
    st.error("The main dataset is missing. Please check that data/cleaned_data.csv is included in the deployment repository.")
    st.stop()

try:
    df = pd.read_csv(data_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()
if df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

# Normalize common spelling/legacy variants so one place is not counted twice.
if COL_STATE in df.columns:
    df[COL_STATE] = df[COL_STATE].apply(normalize_state_name)

# Derive only notebook-supported fields if they are absent.
if "TimePeriod" not in df and "Hour" in df:
    hour = pd.to_numeric(df["Hour"], errors="coerce")
    df["TimePeriod"] = pd.cut(hour, [-1,5,11,17,23],
                              labels=["Night","Morning","Afternoon","Evening"]).astype("object")
if "Season" not in df and COL_MONTH in df:
    month_num = pd.to_numeric(df[COL_MONTH], errors="coerce")
    if month_num.notna().sum() == 0:
        month_names = {m:i for i,m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"],1)}
        month_num = df[COL_MONTH].astype(str).str.strip().str.title().map(month_names)
    def season(m):
        if pd.isna(m): return "Unknown"
        m=int(m)
        if m in (12,1,2): return "Winter"
        if m in (3,4,5): return "Summer"
        if m in (6,7,8,9): return "Monsoon"
        return "Post-Monsoon"
    df["Season"] = month_num.apply(season)
if "row_id" not in df:
    df["row_id"] = np.arange(len(df))

# Filters
view = df.copy()
with st.sidebar:
    if COL_STATE in view:
        states = sorted(view[COL_STATE].dropna().astype(str).unique())
        chosen_states = st.multiselect("Filter states", states)
        if chosen_states:
            view = view[view[COL_STATE].astype(str).isin(chosen_states)]
    if COL_YEAR in view:
        years = sorted(pd.to_numeric(view[COL_YEAR], errors="coerce").dropna().astype(int).unique())
        chosen_years = st.multiselect("Filter years", years)
        if chosen_years:
            view = view[pd.to_numeric(view[COL_YEAR], errors="coerce").isin(chosen_years)]
    if COL_SEVERITY in view:
        severities = sorted(view[COL_SEVERITY].dropna().astype(str).unique())
        chosen_severity = st.multiselect("Filter severity", severities, default=severities)
        if chosen_severity:
            view = view[view[COL_SEVERITY].astype(str).isin(chosen_severity)]

# KPI row — states and union territories are counted separately.
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("RECORDED ACCIDENTS", f"{len(view):,}")
if COL_STATE in view:
    region_names = set(view[COL_STATE].dropna().astype(str).map(normalize_state_name))
    states_covered = len(region_names & INDIAN_STATES)
    uts_covered = len(region_names & INDIAN_UTS)
    k2.metric("STATES COVERED", f"{states_covered}/28")
    k3.metric("UTs COVERED", f"{uts_covered}/8")
else:
    k2.metric("STATES COVERED", "N/A")
    k3.metric("UTs COVERED", "N/A")
k4.metric("CITIES COVERED", f"{view[COL_CITY].nunique():,}" if COL_CITY in view else "N/A")
k5.metric("SEVERITY CLASSES", f"{view[COL_SEVERITY].nunique():,}" if COL_SEVERITY in view else "N/A")

if COL_STATE in view:
    observed = set(view[COL_STATE].dropna().astype(str).map(normalize_state_name))
    unrecognized = sorted(observed - INDIAN_STATES - INDIAN_UTS - {"Unknown"})
    if unrecognized:
        st.warning(
            "Some State Name values are not recognized as an Indian state or union territory: "
            + ", ".join(unrecognized)
            + ". They remain in charts, but are excluded from the state/UT KPI until corrected."
        )
st.markdown("---")

tabs = st.tabs(["Overview", "State & City Hotspots", "Severity Prediction", "Clustering", "Apriori vs FP-Growth", "Dataset"])
with tabs[0]:
    left,right = st.columns([1.05,1.4])
    with left:
        st.markdown("### Severity distribution")
        if COL_SEVERITY in view:
            pie = count_frame(view,COL_SEVERITY,"Severity")
            fig = px.pie(
                pie, names="Severity", values="Accidents", hole=.58,
                color_discrete_sequence=["#ff6b78","#f5b84b","#6c8cff","#5dd6c0","#ad8cff"]
            )
            fig.update_traces(
                textposition="inside", textinfo="percent",
                textfont=dict(color="#101116", size=14),
                hovertemplate="<b>%{label}</b><br>Accidents=%{value:,}<br>Share=%{percent}<extra></extra>"
            )
            st.plotly_chart(style_fig(fig,300),use_container_width=True)
        else: st.warning(f"Column missing: {COL_SEVERITY}")
    with right:
        st.markdown("### Recorded accidents by state")
        if COL_STATE in view:
            st.plotly_chart(chart_bar(view,COL_STATE,COL_STATE,10,True),use_container_width=True)
        else: st.warning(f"Column missing: {COL_STATE}")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("#### Road type")
        if COL_ROAD in view: st.plotly_chart(chart_bar(view,COL_ROAD,"Road type",8),use_container_width=True)
    with c2:
        st.markdown("#### Weather")
        if COL_WEATHER in view: st.plotly_chart(chart_bar(view,COL_WEATHER,"Weather",8),use_container_width=True)
    with c3:
        st.markdown("#### Time period")
        if "TimePeriod" in view: st.plotly_chart(chart_bar(view,"TimePeriod","Time period",8),use_container_width=True)
        elif COL_TIME in view: st.plotly_chart(chart_bar(view,COL_TIME,"Time of day",8),use_container_width=True)
    c4,c5 = st.columns(2)
    with c4:
        st.markdown("#### Seasonal distribution")
        if "Season" in view: st.plotly_chart(chart_bar(view,"Season","Season",8),use_container_width=True)
        else: st.info("Season column is not present in this uploaded CSV.")
    with c5:
        st.markdown("#### Severity by time period")
        tcol = "TimePeriod" if "TimePeriod" in view else COL_TIME
        if tcol in view and COL_SEVERITY in view:
            cross = view.groupby([tcol,COL_SEVERITY],dropna=False).size().reset_index(name="Accidents")
            fig = px.bar(cross,x=tcol,y="Accidents",color=COL_SEVERITY,barmode="group",
                         color_discrete_sequence=["#ff6b78","#6c8cff","#f5b84b","#5dd6c0"])
            st.plotly_chart(style_fig(fig,300),use_container_width=True)

with tabs[1]:
    st.markdown("### State-wise recorded accident counts")
    if COL_STATE in view:
        state_counts = view.groupby(COL_STATE,dropna=False).size().reset_index(name="Accident_Count").sort_values("Accident_Count",ascending=False)
        l,r = st.columns([1.2,1])
        with l: st.plotly_chart(chart_bar(view,COL_STATE,COL_STATE,15,True),use_container_width=True,key="state_hotspots_chart")
        with r:
            st.markdown("#### Top state totals")
            st.dataframe(state_counts.head(15),use_container_width=True,hide_index=True)
        st.download_button("Download state counts CSV",state_counts.to_csv(index=False).encode(), "state_hotspots.csv","text/csv")
    if COL_STATE in view and COL_CITY in view:
        city_frame = view[view[COL_CITY].fillna("Unknown").astype(str).str.lower()!="unknown"]
        city_counts = city_frame.groupby([COL_STATE,COL_CITY],dropna=False).size().reset_index(name="Accident_Count").sort_values("Accident_Count",ascending=False)
        st.markdown("### Top state-city combinations")
        st.dataframe(city_counts.head(25),use_container_width=True,hide_index=True)
        st.caption("These are counts of rows in the stored dataset. The notebook does not provide verified coordinates for each accident, so the dashboard does not invent precise accident points.")

    map_path = Path(__file__).resolve().parent / "geographical_hotspot_map.html"
    if map_path.exists():
        st.markdown("### Geographical hotspot map")
        st.caption("Interactive geographical map included with the project files.")
        import streamlit.components.v1 as components
        components.html(map_path.read_text(encoding="utf-8"), height=650, scrolling=True)

with tabs[2]:
    st.markdown("### Decision Tree severity prediction")
    missing = [c for c in FEATURES+[COL_SEVERITY] if c not in df.columns]
    if missing:
        st.error("The uploaded CSV is missing model columns: " + ", ".join(missing))
    else:
        @st.cache_resource(show_spinner="Training the Decision Tree model...")
        def train_model(csv_bytes):
            data = pd.read_csv(io.BytesIO(csv_bytes))
            X = data[FEATURES].copy()
            y = data[COL_SEVERITY].astype(str)
            cat = [c for c in FEATURES if X[c].dtype == "object" or c in [COL_STATE,COL_WEATHER,COL_ROAD,"Road Condition","Lighting Conditions","Traffic Control Presence","TimePeriod","Season","Vehicle Type Involved","Driver Gender","Driver License Status","Alcohol Involvement","Accident Location Details"]]
            num = [c for c in FEATURES if c not in cat]
            for c in cat: X[c] = X[c].fillna("Unknown").astype(str)
            for c in num:
                X[c] = pd.to_numeric(X[c],errors="coerce")
                X[c] = X[c].fillna(X[c].median())
            pre = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num","passthrough",num)])
            pipe = Pipeline([("prep",pre),("tree",DecisionTreeClassifier(max_depth=6,class_weight="balanced",random_state=42))])
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
            pipe.fit(Xtr,ytr)
            return pipe, accuracy_score(yte,pipe.predict(Xte)), list(pipe.named_steps["tree"].classes_)
        try:
            model,acc,classes=train_model(data_file.getvalue())
            a,b=st.columns(2)
            a.metric("Held-out test accuracy",f"{acc:.1%}")
            b.metric("Model", "Decision Tree · max depth 6")
            st.caption("Your notebook's own accuracy is measured separately; this website retrains the same model type using a preprocessing pipeline, so the result can differ. Treat predictions as educational, not as safety-critical advice.")
            formvals={}
            with st.form("predict_form"):
                c1,c2=st.columns(2)
                for i,feat in enumerate(FEATURES):
                    with (c1 if i%2==0 else c2):
                        if feat in classes:  # defensive; target is not expected to be a feature
                            continue
                        if feat in [COL_WEATHER,COL_ROAD,COL_STATE,"Road Condition","Lighting Conditions","Traffic Control Presence","TimePeriod","Season","Vehicle Type Involved","Driver Gender","Driver License Status","Alcohol Involvement","Accident Location Details"]:
                            opts=sorted(df[feat].fillna("Unknown").astype(str).unique().tolist())
                            if "Unknown" not in opts: opts.append("Unknown")
                            formvals[feat]=st.selectbox(feat,opts,key="input_"+feat)
                        else:
                            vals=pd.to_numeric(df[feat],errors="coerce").dropna()
                            formvals[feat]=st.number_input(feat,value=float(vals.median()) if len(vals) else 0.0,step=1.0,key="input_"+feat)
                submitted=st.form_submit_button("Predict severity",type="primary")
            if submitted:
                row=pd.DataFrame([formvals],columns=FEATURES)
                prediction=model.predict(row)[0]
                st.success(f"Predicted accident severity: {prediction}")
                probs=model.predict_proba(row)[0]
                p_df=pd.DataFrame({"Severity":model.named_steps["tree"].classes_,"Model score":probs}).sort_values("Model score",ascending=False)
                fig=px.bar(p_df,x="Severity",y="Model score",color="Model score",color_continuous_scale=["#3949ab","#ff6b78"])
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(style_fig(fig,260),use_container_width=True)
        except Exception as e:
            st.error(f"Model training failed: {e}")

with tabs[3]:
    st.markdown("### Clustering analysis")
    st.caption("Your clustering features are weather, road type, road condition, lighting conditions and time period. These are condition-pattern clusters, not geographic hotspots.")
    if cluster_file is not None:
        try:
            comp=pd.read_csv(cluster_file)
            st.dataframe(comp,use_container_width=True,hide_index=True)
            if "Method" in comp.columns and "Silhouette score" in comp.columns:
                st.plotly_chart(style_fig(px.bar(comp,x="Method",y="Silhouette score",color="Method",color_discrete_sequence=["#6c8cff","#ff6b78"]),280),use_container_width=True)
        except Exception as e: st.error(f"Cannot read clustering comparison: {e}")
    else:
        st.info("Upload `clustering_comparison.csv` in the sidebar to display the exact metrics from your notebook.")
    if dbscan_file is not None:
        try:
            labels=pd.read_csv(dbscan_file)
            cluster_col="cluster" if "cluster" in labels.columns else None
            if cluster_col:
                counts=labels[cluster_col].value_counts().rename_axis("Cluster").reset_index(name="Records")
                st.markdown("#### DBSCAN cluster sizes")
                st.plotly_chart(style_fig(px.bar(counts,x="Cluster",y="Records",color="Records",color_continuous_scale=["#3949ab","#ff6b78"]),260),use_container_width=True)
        except Exception as e: st.warning(f"Cannot read DBSCAN labels: {e}")
    if hier_file is not None:
        try:
            labels=pd.read_csv(hier_file)
            if "hier_cluster" in labels:
                counts=labels["hier_cluster"].value_counts().rename_axis("Hierarchical cluster").reset_index(name="Sample records")
                st.markdown("#### Hierarchical cluster sizes")
                st.plotly_chart(style_fig(px.bar(counts,x="Hierarchical cluster",y="Sample records",color="Sample records",color_continuous_scale=["#3949ab","#ff6b78"]),260),use_container_width=True)
        except Exception as e: st.warning(f"Cannot read hierarchical labels: {e}")

with tabs[4]:
    st.markdown("### Apriori vs FP-Growth")
    if algo_file is not None:
        try: algo=pd.read_csv(algo_file)
        except Exception: algo=pd.DataFrame()
    else: algo=pd.DataFrame()
    if not algo.empty:
        st.dataframe(algo,use_container_width=True,hide_index=True)
        if "Algorithm" in algo and "Runtime (seconds)" in algo:
            st.plotly_chart(style_fig(px.bar(algo,x="Algorithm",y="Runtime (seconds)",color="Algorithm",color_discrete_sequence=["#6c8cff","#ff6b78"]),280),use_container_width=True)
    else:
        st.info("Upload `algorithm_comparison.csv` to display the exact itemset counts and runtimes saved by your notebook.")
    if rules_file is not None:
        try:
            rules=pd.read_csv(rules_file)
            if "lift" in rules: rules=rules.sort_values("lift",ascending=False)
            st.markdown("#### Saved association rules")
            st.dataframe(rules.head(100),use_container_width=True,hide_index=True)
            st.download_button("Download displayed rules",rules.to_csv(index=False).encode(),"association_rules_filtered.csv","text/csv")
        except Exception as e: st.warning(f"Cannot read association rules: {e}")
    else:
        st.info("Upload `association_rules.csv` to explore your saved severity-related rules. Support, confidence and lift indicate co-occurrence patterns, not causation.")

with tabs[5]:
    st.markdown("### Dataset explorer")
    st.caption(f"{len(view):,} rows after filters · {len(view.columns)} columns")
    st.dataframe(view.head(1500),use_container_width=True,hide_index=True)
    st.download_button("Download filtered records CSV",view.to_csv(index=False).encode(),"filtered_accident_data.csv","text/csv")

st.markdown("---")
st.markdown('<div class="small-muted">ROAD ACCIDENT ANALYSIS PROJECT · Counts depend on the uploaded dataset. Model output is experimental and must not be used for emergency decisions.</div>',unsafe_allow_html=True)
