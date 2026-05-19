# 🌍 ReCliDaR: Representative Climate Days Recognizer

**Product of University of New South Wales (UNSW Sydney), Australia**

ReCliDaR is a research tool designed to identify representative climate days from EnergyPlus Weather (.epw) files using unsupervised machine learning techniques. It enables building energy and urban microclimate modelers to reduce simulation time by leveraging weighted representative days, while still preserving the accuracy of performance assessments for urban built environments.

## Citation
If you use this software, please cite it using this DOI: [https://doi.org/10.5281/zenodo.20255919](https://doi.org/10.5281/zenodo.20255919)

---

## 🚀 Three Modes of Usage

### 🔹 1. Web Application (No Installation)
The fastest way to process individual files instantly in your browser.

- 🌐 URL: https://unsw-reclidar.streamlit.app/
- Best for: Instant individual file processing and cross-platform access

---

### 🔹 2. Windows Standalone Executable (Offline)
A portable desktop application that works without internet or Python.

- Access: Download ReCliDaR_tool.exe from the Releases section
- Best for: Private, offline research

---

### 🔹 3. Python Research Engine (Batch Processing)
A modular script (reclidar_engine.py) for large-scale processing.

- Best for: Automating analysis for 100+ files

---

## 💻 Developer Quick Start

### Install Dependencies

```bash
pip install ladybug-core pandas numpy scikit-learn matplotlib
```

### Example Usage

```python
import os
import glob
import pandas as pd
from reclidar_engine import ReCliDaR

# Configurations
EPW_FOLDER = r"Path to folder with input files"
OUTPUT_FOLDER = "Path to output folder"
METHODS = ["kMeans", "GMM", "HAC"]
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for epw_file in glob.glob(os.path.join(EPW_FOLDER, "*.epw")):
    loc_name = os.path.splitext(os.path.basename(epw_file))[0]
    print(f"Processing: {loc_name}...")
    
    # 1. Parse Data
    final_df, scaled_df = ReCliDaR.process_epw(epw_file)
    
    # 2. Iterate Models & Accumulate Results
    all_rep_dfs = []
    raw_counts_dict = {}
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    dist_df = pd.DataFrame(index=range(1, 13))
    
    for method in METHODS:
        labels = ReCliDaR.run_analysis(scaled_df, method=method)
        if labels is not None:
            # Get Representative Days
            rep_df = ReCliDaR.get_representative_days(final_df, scaled_df, labels, method)
            all_rep_dfs.append(rep_df)
            
            # Get Month breakdown Matrix
            m_dist = ReCliDaR.get_monthly_distribution(final_df, labels, method)
            for col in m_dist.columns:
                dist_df[col] = m_dist[col]
                raw_counts_dict[col] = m_dist[col].to_dict()
                
    # Combine outputs
    master_rep_df = pd.concat(all_rep_dfs, ignore_index=True)
    dist_df.index = month_names
    
    # 3. Save Non-UI Modular Exports to File System
    master_rep_df.to_csv(f"{OUTPUT_FOLDER}/{loc_name}_representative_days.csv", index=False)
    dist_df.to_csv(f"{OUTPUT_FOLDER}/{loc_name}_monthly_distribution.csv")
    
    # Generate and Save Text Report Document
    report_txt = ReCliDaR.generate_v101_report(master_rep_df, raw_counts_dict, METHODS)
    with open(f"{OUTPUT_FOLDER}/{loc_name}_continuity_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)
        
    # Generate and Save Headless Visualizations 
    ReCliDaR.generate_radar_plots(raw_counts_dict, METHODS, save_path=f"{OUTPUT_FOLDER}/{loc_name}_radar_profiles.png")

print("Batch complete!")
```

---

## 📜 License

MIT License

Copyright (c) 2026 ReCliDaR

Permission is hereby granted, free of charge, to any person obtaining a copy...
