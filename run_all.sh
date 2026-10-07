#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/code"
for s in 01_build_base_panel.py 02_knowledge_stock_crop_calendar.py 03_climate_cckp.py 04_disasters_land.py 05_main_analysis.py 06_additional_tests.py; do
  echo "== $s"; python "$s"
done
