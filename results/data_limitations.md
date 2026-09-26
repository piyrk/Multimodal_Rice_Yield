# Data Limitations and Dataset Scope

## Multimodal Rice Yield Prediction Dataset  
**Study Region:** Andhra Pradesh, India  
**Crop:** Rice  
**Target Variable:** Yield (kg/ha)  
**Dataset Size:** 240 validated multimodal samples (from 273 initial candidates)  

---

## 1. SPATIAL SCOPE

### Satellite Representation
- **Modality:** Landsat 5 Collection 2 surface reflectance imagery
- **Spatial Resolution:** 30 m/pixel
- **Patch Size:** 128 × 128 pixels = 3.84 × 3.84 km patch
- **Centering:** Patch is centred on district centroid (from FAO/GAUL 2015 boundary)
- **Interpretation:** District-level spatial average, NOT field-level ground truth

### Critical Limitation
The satellite patch represents a fixed spatial window within each district, calculated from the district boundary centroid. It does NOT:
- Capture individual farm fields
- Represent entire district rice cultivation area
- Guarantee alignment with principal rice-growing regions within the district
- Account for heterogeneous land use within the patch

Therefore, satellite features should be interpreted as **district-level proxy indicators** of growing-season vegetation, not field-specific reflectance.

---

## 2. TEMPORAL SCOPE

### Yield Data
- **Temporal Coverage:** 1998–2011 (14 years)
- **Seasons:** Both Kharif (main rice season) and Rabi (secondary rice season)
- **Granularity:** District-season-year aggregated yield (kg/ha)
- **Source:** Indian Agricultural Statistics at a Glance

### Satellite Data
- **Sensor:** Landsat 5 TM (orbiting 1984–2013)
- **Season Windows:**
  - **Kharif:** June 1 – November 30 (monsoon cropping)
  - **Rabi:** December 1 – May 31 (post-monsoon cropping)
- **Availability:** Scenes with < 5% cloud cover after QA masking applied
- **Composite:** Seasonal median composite (reduces noise, fills cloud gaps)

### Weather Data
- **Source:** ERA5-Land ECMWF daily aggregates
- **Features:** Rainfall (mm), temperature (°C), humidity (%)
- **Resolution:** ~9 km gridded; district-level extraction via centroid coordinate
- **Coverage:** Same season windows as satellite (Kharif/Rabi)

### Soil Data
- **Source:** SoilGrids v2.0 (ISRIC)
- **Features:** Nitrogen (g/kg), pH, SOC – soil organic carbon (g/kg), Clay (%)
- **Depth:** 0–5 cm surface layer only
- **Temporal Note:** Soil data is static per district; no inter-annual variation captured

---

## 3. DATA QUALITY & COMPLETENESS

### Landsat Availability & Cloud Masking
- **Total candidates submitted for extraction:** 273
- **Extracted satellite patches:** 262
- **Valid patches (≥5% clear pixels):** 240
- **Excluded (cloud/water/no data):** 33 samples (12.1%)
  - **1998–2003 data:** Many patches heavily clouded or at image edges; Landsat 5 data quality varies in early mission years
  - **Rabi season:** More cloud/haze during post-monsoon months in coastal/southern districts
  - **Coastal districts:** Higher failure rate due to persistent water/haze during certain seasons

### Weather Availability
- **Total queries:** 273
- **Records retrieved:** 255–270 (weather retrieval had network issues for 2011-12 samples)
- **Completeness check:** All retrieved records were chronologically complete within season windows
- **Missing modality:** 3 samples failed weather retrieval due to network outages

### Soil Availability
- **Coverage:** 100% of 13 districts (static, one record per district)
- **Completeness:** All required features present for all districts

### Final Usable Dataset
| Modality | Extracted | Valid | Loss |
|---|---|---|---|
| Satellite | 262 | 240 | 8% |
| Weather | 270 | 241 | 1% |
| Soil | 273 | 273 | 0% |
| **Multimodal (all three)** | **270** | **240** | **12%** |

---

## 4. KNOWN BIASES & CONFOUNDS

### Satellite Patch Centering Bias
- Patch is centred on **district geometry centroid**, not on principal rice-growing area within district
- Larger districts may have rice cultivation concentrated in specific regions far from centroid
- **Implication:** Satellite features may not optimally represent rice-specific crop status

### Seasonal Timing Misalignment
- Landsat revisit cycle: 16 days
- Seasonal window (180 days) may miss critical growth phases if clouds persistent
- Median composite smooths variability; peak crop stress events may be dampened

### Rabi Season Challenges
- Post-monsoon haze and residual cloud cover in coastal/eastern AP districts
- Cooler temperatures in December–February reduce vegetation vigour signal
- Smaller Rabi cultivation area compared to Kharif; less land dedicated to rice

### Historical Data Quality
- **Landsat 5 mission lifecycle:** 1984–2013
  - Early years (1998–2003): Sensor degradation, inconsistent cloud masking in L2 product
  - Late years (2010–2013): Sensor near end-of-life; some scenes show scan-line corruption

### Soil Data Limitations
- Only 0–5 cm depth; does not capture subsurface properties (rooting zone N availability, moisture retention)
- Static per district; no inter-annual variability in soil conditions
- SoilGrids predictions are interpolated from sparse ground samples; coastal/remote areas have higher uncertainty

---

## 5. REPRESENTATIVENESS

### Geographic Coverage
- **Districts:** 13 of Andhra Pradesh (100% coverage)
- **Years:** 1998–2011 (14 years, with gaps in early data availability)
- **Seasons:** Kharif (primary) and Rabi (secondary)
- **Yield range:** 800–5200 kg/ha (varies by district, season, year)

### Not Representative Of
- Field-level heterogeneity (aggregated to district-season-year)
- Farmer-specific management practices
- Real-time operational crop monitoring
- Crop growth models (satellite phenology ≠ agronomic stage)

---

## 6. DATA ERRORS & HANDLING

### Missing Values
- **Satellite:** 0 NaN values (cloud-masked areas represented as invalid_fraction metric)
- **Weather:** 0 NaN values in core features (rainfall, temperature, humidity)
- **Soil:** 0 NaN values in core features
- **Yield:** All 240 samples have valid numeric yield values

### Duplicate Samples
- None found (unique sample_id constraint enforced)

### Outliers
- **Yield:** No automatic filtering applied
  - Lowest: ~800 kg/ha (drought/poor conditions)
  - Highest: ~5200 kg/ha (irrigated/optimal conditions)
  - All retained as valid agronomic variation

### Extraction Failures Logged
- 33 samples excluded due to satellite cloud cover
- 3 samples excluded due to weather API network failures
- Failure reason and metadata recorded in `results/extraction_log.csv`

---

## 7. RECOMMENDED USAGE NOTES

### For Model Development
1. **Do not interpret satellite features as field-level crop health.** They represent district-level seasonal medians.
2. **Account for temporal autocorrelation:** Adjacent years/seasons in same district are not independent samples.
3. **Consider Rabi season as secondary:** Lower cultivation, higher failure rates; smaller training set per district.
4. **Validate model spatially:** Test on district-held-out folds, not random shuffled splits.
5. **Account for Landsat 5 decay:** Model trained on mixed-quality satellite imagery from 1998–2013.

### For Research Reporting
1. State clearly: "Yield predictions at district-season-year granularity, not field-level."
2. Cite satellite spatial resolution and patch size: "128 × 128 pixels (3.84 × 3.84 km) centred on district centroid."
3. Report baseline: "240 valid multimodal samples from 273 candidates (87.9% usable)."
4. Mention Rabi season limitations: "Secondary season; higher cloud cover; smaller cultivation area."
5. Acknowledge soil static limitation: "District-level soil features; no inter-annual variation."

---

## 8. DATA PROVENANCE

### Sources
| Modality | Source | License | Version |
|---|---|---|---|
| Satellite | USGS Landsat 5 Collection 2 | Public Domain | LANDSAT/LT05/C02/T1_L2 |
| Weather | ECMWF ERA5-Land | CC4.0 | ecmwf/ERA5_LAND/DAILY_AGGR |
| Soil | ISRIC SoilGrids | CC4.0 | projects/soilgrids-isric/* |
| Yield | Indian Agri Stats | Public | 1998–2011 Ministry of Agriculture |
| Districts | FAO/GAUL | CC4.0 | FAO/GAUL/2015/level2 |

### Processing
- All extraction performed via Google Earth Engine Python API
- Cloud masking: QA_PIXEL band (Collection 2 L2 standard)
- Scaling: Landsat Collection 2 surface reflectance (0.0000275 × DN − 0.2)
- Projections: WGS84 (EPSG:4326)
- Units: kg/ha (yield), mm (rainfall), °C (temperature), % (humidity, cloud cover, soil clay)

---

## 9. VERSION & CHECKSUM

- **Dataset Version:** 1.0
- **Creation Date:** 2025-01-XX
- **Final Sample Count:** 240 (validated multimodal)
- **Extraction Log:** `results/extraction_log.csv`
- **Manifests:**
  - `data/processed/multimodal_manifest.csv`
  - `data/processed/satellite_manifest.csv`
  - `data/processed/weather_manifest.csv`
  - `data/processed/soil_features.csv`

---

## 10. CONTACT & ATTRIBUTION

**Project:** Predicting Crop Yield with Multimodal Deep Learning  
**Region:** Andhra Pradesh, India  
**Crop:** Rice  
**Degree:** B.Tech Capstone Project  
**Academic Year:** 2024–2025

---

*This document was generated automatically during multimodal dataset extraction.*  
*For updates or corrections, refer to the extraction scripts and manifests.*
