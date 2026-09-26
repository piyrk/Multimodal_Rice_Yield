# Paper Facts

The following statements are factual and safe to reuse.

- The study predicts rice yield in **kg/ha** for district-season-year observations in Andhra Pradesh, India.
- The validated multimodal dataset contains **239** samples after row-level quality checks.
- The canonical chronological split contains **172 training**, **26 validation**, and **41 held-out test** samples. Training years precede validation years, and validation years precede test years.
- The historical data range is **1998-99 through 2011-12**.
- Satellite inputs contain six Landsat-derived reflectance bands plus NDVI, represented as **7 × 128 × 128** float32 patches.
- Weather inputs contain **183 daily observations** with `rainfall_mm`, `temperature_c`, and `humidity_pct`.
- Soil inputs contain four features: nitrogen, pH, soil organic carbon, and clay.
- The final model uses unchanged SatelliteCNN, WeatherLSTM, and SoilMLP encoders, concatenated into a 320-dimensional representation and passed through a fusion MLP and regression head.
- The locked final configuration is learning rate **0.0005**, fusion dropout **0.0**, AdamW, MSELoss, batch size **16**, and **9 epochs**.
- Final training used the combined **198 train+validation samples**. Target normalization statistics were calculated from those 198 samples only.
- Three final checkpoints were trained with seeds **42, 123, and 2026** and evaluated on the same 41 test samples.
- The primary final results are the arithmetic mean ± sample standard deviation of the three seed-level test metrics: **MAE 591.175 ± 68.889 kg/ha**, **RMSE 758.353 ± 84.317 kg/ha**, and **R² 0.328446 ± 0.143360**.
- The per-seed primary results are: seed 42 MAE **512.187**, RMSE **660.994**, R² **0.493979**; seed 123 MAE **622.524**, RMSE **806.461**, R² **0.246747**; seed 2026 MAE **638.815**, RMSE **807.604**, R² **0.244611**.
- Secondary error analysis averaged the three seed predictions for each test sample. Under that different aggregation, mean absolute error was **556.416 kg/ha**, mean signed error was **+74.520 kg/ha**, and Pearson correlation was **0.6655**.
- The Streamlit application is an inference/demo system using supplied inputs. It is not a live data service and does not establish a scientifically validated future-season or 2027 forecast.

Metric definitions:

- **MAE** is the mean absolute difference between predicted and observed yield in kg/ha.
- **RMSE** is the square root of the mean squared prediction error in kg/ha.
- **R²** is `1 - SSE/SST` on the specified evaluation set.
- The reported ± values are sample standard deviations across the three final seeds, not confidence intervals.
