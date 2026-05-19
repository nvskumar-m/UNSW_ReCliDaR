import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Set up page configurations
st.set_page_config(
    page_title="ReCliDaR App v1.0.1",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENVIRONMENT-SAFE PARSING ENGINE ---
@st.cache_data(show_spinner=False)
def get_clean_epw_data(uploaded_file):
    import engine
    temp_filename = "temp_uploaded_weather.epw"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        epw_df, scaled_df = engine.get_epwData(temp_filename)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return epw_df, scaled_df
    except (ValueError, KeyError):
        from ladybug.epw import EPW
        from sklearn.preprocessing import StandardScaler
        
        try:
            epw = EPW(temp_filename)
            cliVars = [epw.dry_bulb_temperature.values, epw.dew_point_temperature.values,
                       epw.relative_humidity.values, epw.global_horizontal_radiation.values,
                       epw.direct_normal_radiation.values, epw.diffuse_horizontal_radiation.values,
                       epw.wind_speed.values, epw.wind_direction.values]
            cliNames = ['DBT','DPT','RH','GHR','DNR','DHR','WS','WD']
            
            temp_df = pd.DataFrame({name: var for name, var in zip(cliNames, cliVars)})
            temp_df['DateTime'] = pd.date_range(start="2018-01-01 00:00", end="2018-12-31 23:00", freq='h')
                    
            temp_df.index = temp_df['DateTime']
            temp_df = temp_df.drop('DateTime', axis=1)
            temp_df['Day'] = temp_df.index.dayofyear
            temp_df['Hour'] = temp_df.index.hour
            temp_df = temp_df.reset_index().iloc[:, 1:]
            
            dfs = [pd.pivot_table(temp_df, values=col, index='Day', columns='Hour', aggfunc=np.sum) for col in temp_df.columns[:8]]
            join_df = pd.concat(dfs, axis=1)
            join_df.columns = [f'{col}_{h}' for col in cliNames for h in range(24)]
            
            final_df = join_df.copy().reset_index().iloc[:, 1:]
            full_year_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='d')
                    
            final_df['Month'] = full_year_2025.month
            final_df['Day_of_Month'] = full_year_2025.day
            
            scaler = StandardScaler()
            scaled = scaler.fit_transform(final_df)
            
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return final_df, pd.DataFrame(scaled, columns=final_df.columns)
            
        except Exception as inner_e:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            st.error(f"Fallback parsing routine encountered a failure: {str(inner_e)}")
            return None, None


# --- CIRCULAR SEASONAL CONTINUITY CALCULATOR ---
def analyze_circular_continuity(month_counts, threshold_days=3):
    core_months = [m for m, count in month_counts.items() if count > threshold_days]
    extreme_months = [m for m, count in month_counts.items() if 0 < count <= threshold_days]
    
    if not core_months:
        return "Discontinuous / Insufficient Core Baseline Data", extreme_months

    if len(core_months) == 1:
        base_status = "Continuous Seasonal Continuity (Single Month Spanned)"
    else:
        core_months = sorted(core_months)
        gaps = []
        for i in range(len(core_months)):
            m1 = core_months[i]
            m2 = core_months[(i + 1) % len(core_months)]
            if m2 > m1:
                diff = m2 - m1
            else:
                diff = (m2 + 12) - m1
            if diff > 1:
                gaps.append(diff)
                
        if len(gaps) <= 1:
            base_status = "Continuous Seasonal Continuity Block"
        else:
            base_status = "Trans-seasonal / Discontinuous Block"
            
    if extreme_months:
        base_status += " (with Extreme Event Days)"
    return base_status, extreme_months


# --- OBJECTIVE v1.0.1 CONTINUITY REPORT GENERATOR ---
def generate_v101_report(rep_df, raw_counts_dict, methods):
    months_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
                  7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    lines = [
        "========================================================================",
        "        RECLIDAR CLIMATIC REPRESENTATION & CONTINUITY REPORT (v1.0.1)   ",
        "========================================================================\n"
    ]
    
    for m in methods:
        lines.append(f"## ALGORITHM: {m.upper()}")
        m_reps = rep_df[rep_df['Algorithm'] == m]
        
        for _, row in m_reps.iterrows():
            c_id = int(row['Cluster'])
            rep_m = months_map[int(row['Month'])]
            rep_d = int(row['Day'])
            weight = int(row['Total_Days'])
            
            col_name = f"{m}_{c_id}"
            m_counts = raw_counts_dict[col_name]
            
            continuity_status, extreme_indices = analyze_circular_continuity(m_counts, threshold_days=3)
            
            lines.append(f"  ▶ Cluster {c_id}: Represented by Date {rep_m}-{rep_d:02d}")
            lines.append(f"    - Total Days Covered      : {weight} days")
            lines.append(f"    - Seasonal Continuity     : {continuity_status}")
            
            if extreme_indices:
                ext_strings = [f"{months_map[idx]} ({m_counts[idx]} days)" for idx in extreme_indices]
                lines.append(f"    - Extreme Event Days      : Baseline continuity preserved. Outlier days recorded in: {', '.join(ext_strings)}.")
            else:
                lines.append(f"    - Extreme Event Days      : Zero outlier profile instances recorded outside core seasonal continuity blocks.")
            lines.append("")
        lines.append("-" * 72)
    return "\n".join(lines)


# --- RADAR VISUALIZATION FOR STREAMLIT (STRICT CLEAN TEXT OVERRIDE) ---
def generate_streamlit_radar(raw_counts_dict, methods):
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()
    angles += angles[:1] 
    
    fig, axs = plt.subplots(1, len(methods), figsize=(16, 6), subplot_kw=dict(polar=True))
    if len(methods) == 1:
        axs = [axs]
        
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, m in enumerate(methods):
        ax = axs[idx]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(month_labels, fontsize=9)
        
        ax.set_title(f"{m.upper()}\nSeasonal Continuity Profile", fontsize=11, pad=20, weight='bold')
        
        cluster_idx = 0
        for col_name, counts in raw_counts_dict.items():
            if "Limit" in col_name or "Extreme" in col_name:
                continue
                
            if col_name.startswith(m):
                c_num = col_name.split('_')[1]
                cluster_label = f"Cluster {c_num}"
                values = [counts[m_idx] for m_idx in range(1, 13)]
                values += values[:1] 
                
                color = color_list[cluster_idx % len(color_list)]
                ax.plot(angles, values, linewidth=2, color=color, linestyle='solid', label=cluster_label)
                ax.fill(angles, values, color=color, alpha=0.15)
                cluster_idx += 1
                
        # We assign an explicit clean string name directly here
        ax.plot(angles, [3] * len(angles), color='red', linewidth=1.2, linestyle='--', label='Extreme Event Limit')
        ax.set_rlabel_position(180)
        ax.tick_params(colors='#555555')
        ax.grid(True, linestyle=':')

    # Get the raw text arrays
    handles, labels = axs[0].get_legend_handles_labels()
    
    # HARD OVERRIDE: Forcibly strip out any variant text containing "(3 days)" or duplicates
    clean_labels = []
    clean_handles = []
    seen = set()
    
    for handle, label in zip(handles, labels):
        # Explicit clean string filter step
        fixed_label = label.replace(" (3 days)", "").strip()
        
        if fixed_label not in seen:
            seen.add(fixed_label)
            clean_labels.append(fixed_label)
            clean_handles.append(handle)
    
    fig.legend(
        handles=clean_handles,
        labels=clean_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05), 
        ncol=len(clean_labels),
        fontsize=10,
        frameon=True
    )
    plt.subplots_adjust(top=0.82, bottom=0.15, left=0.05, right=0.95, wspace=0.35)
    return fig


# --- BRANDING HEADER CONFIGURATION (image_b0e079.png Ingestion Engine) ---
image_path = "image_b0e079.png"
if os.path.exists(image_path):
    with open(image_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode()
    header_html = f"""
    <div style="margin-bottom: 5px; padding-top: 10px;">
        <div style="margin-bottom: 25px;">
            <img src="data:image/png;base64,{encoded_img}" style="width: 62px; height: 62px; border-radius: 8px; object-fit: contain; display: block;"/>
        </div>
        <div style="margin-bottom: 6px;">
            <h1 style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-weight: 700; font-size: 2.4rem; color: #FFFFFF; letter-spacing: -0.3px;">
                ReCliDaR: Representative Climate Days Recognizer
            </h1>
        </div>
        <div style="margin-bottom: 30px;">
            <p style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 1.05rem; color: #A0A0A0; font-weight: 400;">
                Department of Built Environment
            </p>
        </div>
    </div>
    """
else:
    header_html = """
    <div style="margin-bottom: 5px; padding-top: 10px;">
        <div style="background-color: #FFDC00; border-radius: 8px; padding: 12px; text-align: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-weight: 900; color: black; line-height: 1.0; width: 62px; height: 62px; display: flex; flex-direction: column; justify-content: center; align-items: center; margin-bottom: 25px;">
            <span style="font-size: 14px; letter-spacing: 0.2px;">UNSW</span>
            <span style="font-size: 7px; font-weight: 500; letter-spacing: 1.0px; margin-top: 2px;">SYDNEY</span>
        </div>
        <div style="margin-bottom: 6px;">
            <h1 style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-weight: 700; font-size: 2.4rem; color: #FFFFFF; letter-spacing: -0.3px;">
                ReCliDaR: Representative Climate Days Recognizer
            </h1>
        </div>
        <div style="margin-bottom: 30px;">
            <p style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 1.05rem; color: #A0A0A0; font-weight: 400;">
                Department of Built Environment
            </p>
        </div>
    </div>
    """

st.html(header_html)

# --- SIDEBAR CONFIGURATIONS ---
st.sidebar.header("⚙️ Model Architecture")
selected_methods = st.sidebar.multiselect(
    "Select Optimization Algorithms",
    options=["kMeans", "GMM", "HAC"],
    default=["kMeans", "GMM", "HAC"]
)

# --- CENTRALIZED INGESTION PANEL ---
st.html(
    """
    <p style="margin: 0 0 8px 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 0.95rem; color: #FFFFFF; font-weight: 500;">
        Choose an EPW file
    </p>
    """
)
uploaded_file = st.file_uploader("Choose an EPW file", type=["epw"], label_visibility="collapsed")


# --- DATA PROCESSING INTERACTION EXECUTION ---
if uploaded_file is not None:
    status_box = st.empty()
    
    status_box.info("⏳ Step 1/4: Ingesting climate file data arrays...")
    import engine
    epw_df, scaled_df = get_clean_epw_data(uploaded_file)
    
    if epw_df is not None and scaled_df is not None:
        status_box.info("⏳ Step 2/4: Computing Principal Component Analysis (PCA) dimensions...")
        X_pca = engine.run_PCA(scaled_df).to_numpy()
        
        status_box.info("⏳ Step 3/4: Running selected optimization and clustering algorithms...")
        results = {}
        if "kMeans" in selected_methods:
            km_labels, _ = engine.run_kMeans(scaled_df)
            results['kMeans'] = km_labels
            
        if "GMM" in selected_methods:
            best_n = min(range(2, 6), key=lambda n: GaussianMixture(n_components=n, random_state=42).fit(X_pca).bic(X_pca))
            results['GMM'] = GaussianMixture(n_components=best_n, random_state=42).fit(X_pca).predict(X_pca)
            
        if "HAC" in selected_methods:
            best_k = max(range(2, 6), key=lambda k: silhouette_score(X_pca, AgglomerativeClustering(n_clusters=k).fit_predict(X_pca)))
            results['HAC'] = AgglomerativeClustering(n_clusters=best_k).fit_predict(X_pca)
            
        status_box.info("⏳ Step 4/4: Constructing multi-panel profiles and metrics matrices...")
        rep_rows = []
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        dist_df = pd.DataFrame(index=range(1, 13))
        raw_counts_dict = {}
        
        for name, labels in results.items():
            current_df = epw_df.copy()
            current_df['cluster'] = labels
            
            for j in np.unique(labels):
                subset_scaled = scaled_df[labels == j]
                idx = engine.get_rep(subset_scaled)
                actual_idx = subset_scaled.index[idx]
                rep_day_data = epw_df.loc[actual_idx]
                
                rep_rows.append([name, j, int(rep_day_data["Month"]), int(rep_day_data["Day_of_Month"]), len(subset_scaled)])
                
                monthly_counts = [len(current_df[(current_df['cluster'] == j) & (current_df['Month'] == m)]) for m in range(1, 13)]
                col_name = f'{name}_{j}'
                dist_df[col_name] = monthly_counts
                raw_counts_dict[col_name] = {m: count for m, count in zip(range(1, 13), monthly_counts)}

        status_box.empty()
        
        if results:
            rep_df = pd.DataFrame(rep_rows, columns=['Algorithm', 'Cluster', 'Month', 'Day', 'Total_Days'])
            dist_df.index = month_names
            
            st.markdown("### Processed Engine Artifacts")
            tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "📄 Continuity Report", "💾 Export Data"])
            
            with tab1:
                st.subheader("Climatic Seasonal Continuity Profiles")
                radar_fig = generate_streamlit_radar(raw_counts_dict, list(results.keys()))
                st.pyplot(radar_fig)
            
            with tab2:
                st.subheader("Automated Text Summary Evaluation")
                report_text = generate_v101_report(rep_df, raw_counts_dict, list(results.keys()))
                st.code(report_text, language="text")
                
                st.download_button(
                    label="Download Text Report File",
                    data=report_text,
                    file_name="v101_continuity_report.txt",
                    mime="text/plain"
                )
                    
            with tab3:
                st.subheader("Downstream Artifact Extraction")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Representative Days Summary (`v101_representative_days.csv`)**")
                    st.dataframe(rep_df, use_container_width=True)
                    
                    csv_buffer1 = io.StringIO()
                    rep_df.to_csv(csv_buffer1, index=False)
                    st.download_button(
                        label="Download Representative Days CSV",
                        data=csv_buffer1.getvalue(),
                        file_name="v101_representative_days.csv",
                        mime="text/csv"
                    )
                    
                with col2:
                    st.write("**Monthly Breakdown Metrics (`v101_monthly_distribution.csv`)**")
                    st.dataframe(dist_df, use_container_width=True)
                    
                    csv_buffer2 = io.StringIO()
                    dist_df.to_csv(csv_buffer2)
                    st.download_button(
                        label="Download Monthly Distribution CSV",
                        data=csv_buffer2.getvalue(),
                        file_name="v101_monthly_distribution.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("Please select at least one optimization algorithm in the sidebar configuration section.")
else:
    st.html("<br>")
    st.info("💡 Please upload an EPW file above to begin pipeline operations and runtime metrics visualization.")
