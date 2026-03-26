# Smart Energy Modeling Project

---
## 01 Notebook contents

The notebook files contain analysis and visualization of the input dataset.
- Merged 5 buildings → avg_energy (handled DST, missing hours, NaN)
-  Added weather features (temperature, humidity, pressure)
- Added time features (is_weekend, is_holiday)
- Clean DataFrame: 8760 rows, 8 columns, zero nulls
- Saved to data/processed/energy_features_2019.csv
- Decided: window=24, task=many-to-one regression