# Data

The training pipeline expects per-day CSV files organized by year:

```
data/raw/2022/<month>/*.csv
data/raw/2023/<month>/*.csv
```

Each CSV is a per-minute time series from a PV monitoring system with the following columns:

| Column | Description |
|---|---|
| `measured_on` | Timestamp |
| `ac_current__427`, `ac_power__423`, `ac_voltage__426` | AC-side electrical measurements |
| `dc_pos_current__425`, `dc_pos_voltage__424`, `dc_power__422` | DC-side electrical measurements (`dc_power__422` is the forecast target) |
| `ambient_temp__428`, `module_temp_1__429`, `module_temp_2__430`, `module_temp_3__431`, `inverter_temp__432`, `das_temp__433` | Temperature sensors |
| `poa_irradiance__421` | Plane-of-array irradiance |
| `das_battery_voltage__434` | Data acquisition system battery voltage |
| `system_id` | Site/system identifier |

Raw and processed data are not tracked in this repository (see `.gitignore`). Place your
own data under `data/raw/` following the structure above, or point `scripts/run_benchmark.py`
at a different location with `--data-dir`.

`scripts/run_benchmark.py` reads every CSV under `data/raw/<year>/`, sorts by `measured_on`,
drops rows with missing or non-numeric values, and builds sliding-window sequences (default
length 24) over the remaining features to predict the next `dc_power__422` reading.
