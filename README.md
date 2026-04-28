# 🌍 ReCliDaR: Representative Climate Days Recognizer

**Product of University of New South Wales (UNSW Sydney), Australia**

ReCliDaR is a research tool designed to identify representative climate days from EnergyPlus Weather (.epw) files using unsupervised machine learning techniques. It enables building energy and urban microclimate modelers to reduce simulation time by leveraging weighted representative days, while still preserving the accuracy of performance assessments for urban built environments.

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
pip install ladybug-core pandas numpy scikit-learn
```

### Example Usage

```python
import pandas as pd
from reclidar_engine import ReCliDaR

epw_file = "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"

orig, scaled = ReCliDaR.process_epw(epw_file)

method = 'kMeans'
labels = ReCliDaR.run_analysis(scaled, method=method)

reps_df = ReCliDaR.get_representative_days(orig, scaled, labels, method)
dist_df = ReCliDaR.get_monthly_distribution(orig, labels, method)

reps_df.to_csv("representative_days.csv", index=False)
dist_df.to_csv("monthly_distribution.csv", index=False)

print("Analysis complete.")
```

---

## 📜 License

MIT License

Copyright (c) 2026 ReCliDaR

Permission is hereby granted, free of charge, to any person obtaining a copy...
