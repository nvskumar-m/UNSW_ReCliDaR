import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ladybug.epw import EPW
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA

class ReCliDaR:
    """
    Comprehensive tool for Representative Climate Data Recognition.
    Supports k-Means, GMM, and HAC for large-scale .epw processing.
    """
    
    @staticmethod
    def process_epw(file_path):
        """Loads and scales EPW data."""
        epw = EPW(file_path)
        cliVars = [epw.dry_bulb_temperature.values, epw.dew_point_temperature.values,
                   epw.relative_humidity.values, epw.global_horizontal_radiation.values,
                   epw.direct_normal_radiation.values, epw.diffuse_horizontal_radiation.values,
                   epw.wind_speed.values, epw.wind_direction.values]
        cliNames = ['DBT','DPT','RH','GHR','DNR','DHR','WS','WD']
        
        temp_df = pd.DataFrame({name: var for name, var in zip(cliNames, cliVars)})
        temp_df['DateTime'] = pd.date_range(start="2018-01-01 00:00", end="2018-12-31 23:00", freq="h")
        temp_df.index = temp_df['DateTime']
        temp_df = temp_df.drop('DateTime', axis=1)
        temp_df['Day'], temp_df['Hour'] = temp_df.index.dayofyear, temp_df.index.hour
        
        dfs = [pd.pivot_table(temp_df, values=col, index='Day', columns='Hour', aggfunc=np.sum) for col in cliNames]
        join_df = pd.concat(dfs, axis=1)
        join_df.columns = [f'{col}_{i}' for col in cliNames for i in range(24)]
        
        final_df = join_df.copy().reset_index().iloc[:, 1:]
        full_year = pd.date_range(start='2025-01-01', end='2025-12-31', freq='d')
        final_df['Month'], final_df['Day_of_Month'] = full_year.month, full_year.day
        
        scaled = StandardScaler().fit_transform(final_df)
        return final_df, pd.DataFrame(scaled, columns=final_df.columns)

    @staticmethod
    def run_analysis(scaled_df, method='kMeans'):
        """Runs clustering using the selected algorithm (kMeans, GMM, or HAC)."""
        X = PCA(n_components=0.95, random_state=42).fit_transform(scaled_df.to_numpy())
        
        if method == 'kMeans':
            best_k = max(range(2, 6), key=lambda k: silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_))
            return KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X).labels_
        
        elif method == 'GMM':
            best_n = min(range(2, 6), key=lambda n: GaussianMixture(n_components=n, random_state=42).fit(X).bic(X))
            return GaussianMixture(n_components=best_n, random_state=42).fit(X).predict(X)
            
        elif method == 'HAC':
            best_k = max(range(2, 6), key=lambda k: silhouette_score(X, AgglomerativeClustering(n_clusters=k).fit_predict(X)))
            return AgglomerativeClustering(n_clusters=best_k).fit_predict(X)
            
        return None

    @staticmethod
    def get_representative_days(original_df, scaled_df, labels, method_name):
        """Output 1: Dataframe of Representative Days."""
        rep_days = []
        for j in np.unique(labels):
            subset_scaled = scaled_df[labels == j]
            D = pairwise_distances(subset_scaled.to_numpy(), metric='euclidean')
            actual_idx = subset_scaled.index[np.argmin(D.mean(axis=1))]
            row = original_df.loc[actual_idx]
            rep_days.append({
                'Algorithm': method_name,
                'Cluster': j,
                'Month': int(row["Month"]),
                'Day': int(row["Day_of_Month"]),
                'Total_Days': len(subset_scaled)
            })
        return pd.DataFrame(rep_days)

    @staticmethod
    def get_monthly_distribution(original_df, labels, method_name):
        """Output 2: Monthly Cluster counts."""
        dist_df = pd.DataFrame(index=range(1, 13))
        dist_df.index.name = 'Month'
        temp_df = original_df.copy()
        temp_df['cluster'] = labels
        for j in np.unique(labels):
            dist_df[f'{method_name}_{j}'] = [len(temp_df[(temp_df['cluster'] == j) & (temp_df['Month'] == m)]) for m in range(1, 13)]
        return dist_df

    # --- NEW AUTOMATED ANALYTICAL ENGINE EXTENSIONS ---

    @staticmethod
    def analyze_circular_continuity(month_counts, threshold_days=3):
        """Circular continuity math logic extracted from app.py."""
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

    @classmethod
    def generate_v101_report(cls, rep_df, raw_counts_dict, methods):
        """Generates the Objective v1.0.1 continuity text summary string."""
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
                
                continuity_status, extreme_indices = cls.analyze_circular_continuity(m_counts, threshold_days=3)
                
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

    @staticmethod
    def generate_radar_plots(raw_counts_dict, methods, save_path=None):
        """Generates Matplotlib Radar plots and returns the figure object (and optionally saves it)."""
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
                    
            ax.plot(angles, [3] * len(angles), color='red', linewidth=1.2, linestyle='--', label='Extreme Event Limit')
            ax.set_rlabel_position(180)
            ax.tick_params(colors='#555555')
            ax.grid(True, linestyle=':')

        handles, labels = axs[0].get_legend_handles_labels()
        
        clean_labels = []
        clean_handles = []
        seen = set()
        
        for handle, label in zip(handles, labels):
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
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            return fig
