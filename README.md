# Replication package

**Agricultural Innovation Capacity, Creative Destruction, and Food Security under Climate Shocks: Evidence from Developing Countries**

[Author names] · [Year] · [Journal, once accepted]

This repository reproduces every table and figure in the paper, starting from public data downloaded directly from the original providers. No data are redistributed here; the scripts fetch them.

## Requirements

Python 3.10 or later and an internet connection. Install the packages with

```
pip install -r requirements.txt
```

Total run time is about 15–30 minutes on a standard laptop, most of it spent downloading data.

## How to run

```
cd code
python 01_build_base_panel.py
python 02_knowledge_stock_crop_calendar.py
python 03_climate_cckp.py
python 04_disasters_land.py
python 05_main_analysis.py
python 06_additional_tests.py
```

or simply `bash run_all.sh` from the repository root.

## What each script does

| Script | Purpose | Output |
|---|---|---|
| `01_build_base_panel.py` | Downloads GRAPE (public agricultural R&D), USDA ERS International Agricultural Productivity, FAOSTAT (food balances, food security indicators, crops, land use, production value) and World Bank data (WDI, WGI, historical income classification); builds the country-year panel | `data/panel_base.csv` |
| `02_knowledge_stock_crop_calendar.py` | Builds the R&D knowledge stock (Eq. 1) and the dominant-cereal growing season from Sacks et al. (2010) | `data/panel_v2.csv`, `data/crop_calendar_country.csv` |
| `03_climate_cckp.py` | Downloads monthly ERA5-based climate indices from the World Bank Climate Change Knowledge Portal and builds growing-season shocks | `data/panel_v3.csv` |
| `04_disasters_land.py` | Adds EM-DAT flood and drought impacts (via Our World in Data), population and land abundance | `data/panel_v4.csv` (analysis panel) |
| `05_main_analysis.py` | Tables 2–7 and 9, inputs for Figures 3–7, local projections, robustness, heterogeneity, Pesaran CD test, wild cluster bootstrap | `results/main_results.json`, `results/descriptives.csv` |
| `06_additional_tests.py` | Section 4.7 and Table 8: research scale versus agricultural size and land area | `results/additional_tests.json` |
| `econ.py` | Fixed-effects estimator, clustered, Driscoll–Kraay and Conley standard errors, wild cluster bootstrap, CD test | – |

## Data sources

| Data | Provider | Access |
|---|---|---|
| Public agricultural R&D (GRAPE v1.0.0) | van Dijk et al. (2025), Zenodo | https://zenodo.org/records/15507361 |
| Agricultural output, TFP and inputs | USDA Economic Research Service | https://www.ers.usda.gov/data-products/international-agricultural-productivity |
| Food balances, food security, crops, land, production value | FAOSTAT | https://www.fao.org/faostat |
| GDP, agricultural share, population, governance, income groups | World Bank | https://data.worldbank.org |
| Climate indices (ERA5, 0.25°) | World Bank Climate Change Knowledge Portal | https://climateknowledgeportal.worldbank.org |
| Crop calendar | Sacks et al. (2010), SAGE, University of Wisconsin | https://sage.nelson.wisc.edu/data-and-models/datasets/crop-calendar-dataset |
| Flood and drought impacts | EM-DAT (CRED), compiled by Our World in Data | https://ourworldindata.org/natural-disasters |

Data providers update their series from time to time. Results downloaded on a different date can differ slightly from the published numbers; the paper uses data accessed in October 2026.

## Notes

- Shocks are standardised against each country's 1981–2010 climate. Innovation capacity is measured over 1981–1990; outcomes over 1991–2022.
- For 15 countries with gaps in 1981–1990, baseline moderators use 1981–1995 averages.
- The change in cereal import dependency is winsorised at the 1st and 99th percentiles.

## Citation

If you use this code, please cite the paper: [full reference once published].

## License

Code: MIT License (see `LICENSE`). Data remain subject to the terms of their original providers.
