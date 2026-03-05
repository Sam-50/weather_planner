# 📋 Weather Planner for Field Technicians

_A Python‑based prototype that combines synthetic weather modelling, machine learning,
and simple optimization to help a utility field technician (e.g. KPLC) plan a
safe, efficient work‑day._

The system simulates historical weather data to train a risk classifier, pulls live
forecasts from Open‑Meteo, and builds a task schedule that maximises high‑priority
work while avoiding unsafe conditions and excessive travel.

---

## ⚙️ Features

- **Synthetic data generation** for reproducible model training
- **Multiclass weather‑risk classifier** (safe / risky / unsafe) using scikit‑learn
- **Real‑world forecast integration** via Open‑Meteo geocoding and hourly API
- **Deterministic fallback generator** when the API is unreachable or the date is
  out‑of‑range
- **Rule‑based safety constraints** (no outdoor work during “unsafe” hours)
- **Beam‑search scheduler** to optimise task order with travel time / priorities
- **Monte‑Carlo simulation** comparing AI schedule vs naive baseline
- Notebook demonstrating the entire end‑to‑end workflow

---

## 📁 Repository Structure

```
weather_planner/
├── README.md                 # ← you’re reading it!
├── all_cells.py              # core logic: data, models, forecasting & scheduling
└── New_Weather_Planner.ipynb # interactive walkthrough used for demos
```

---

## 🛠️ Prerequisites

- Python 3.10+ (tested on Ubuntu 24.04 dev container)
- Internet access for live forecasts (optional; fallback available)
- The following Python packages:

```text
pandas
numpy
scikit-learn
matplotlib
requests
```

Install via pip:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib requests
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Sam-50/weather_planner.git
cd weather_planner
```

### 2. Run the notebook (recommended)

```bash
jupyter lab New_Weather_Planner.ipynb
```

The notebook walks through:

1. generating a synthetic weather dataset,
2. training & evaluating the risk classifier,
3. fetching an Open‑Meteo forecast for Nakuru County,
4. computing a schedule and comparing it with a baseline,
5. visualising results and running Monte‑Carlo experiments.

### 3. Use the Python module

Import and call the functions in `all_cells.py` from your own script:

```python
from all_cells import simulate_weather_rows, get_nakuru_county_forecast, beam_search_schedule, Task

# 1. train or load a model (see notebook for details)
# 2. create task list
tasks = [
    Task("Transformer inspection", "Rural_Rongai", "High", 2, True),
    # …
]

# 3. fetch forecast for a given date
forecast = get_nakuru_county_forecast("2024-03-05", rf_model=my_trained_rf)

# 4. generate schedule
order, score, details = beam_search_schedule(tasks)
```

### 4. Scheduling API

The core scheduling functions expect a DataFrame with columns:

```
hour, location, rain_prob, wind_kph, temp_c,
humidity, pred_risk
```

– `get_nakuru_county_forecast` produces this format automatically.

---

## 🔧 Customisation

- **Locations** – edit `NAKURU_COUNTY_TOWNS` or supply your own
  coordinates in `MANUAL_COORDINATES`.
- **Season mapping** – the simple `"Wet"` default can be replaced with real
  climatology.
- **Task definitions** – change priorities, durations or add new ones.
- **Risk penalty weights** – adjust `RISK_PENALTY` or scheduling heuristics.
- **Beam width** – raise `beam_width` in `beam_search_schedule` for stronger
  (slower) optimisation.

---

## 👉 Example Output

After training the model and fetching a forecast, the scheduler
may print something like:

```
=== AI schedule ===
- Travel to Rural_Rongai (+20 min ≈ 1h).
- Scheduled 'Transformer inspection' at 9:00 for 2h | risks=['safe','safe'] | …
…
AI total score: 12.50 | travel=55 min
```

Monte‑Carlo experiments typically show higher weighted work delivered and
fewer disruptions compared with a naive, priority‑first baseline.

---

## 🧪 Testing & Evaluation

There are no automated tests yet. To verify behaviour manually:

1. Run the notebook and inspect printed reports and plots.
2. Alter `tasks` or `forecast` and observe how the schedule adapts.
3. Simulate API failure by disabling network or exceeding forecast range.

---

## 💡 Notes

- The model is trained on **simulated data only**; performance on real weather
  may vary.
- Forecasts are restricted to ~16 days ahead by Open‑Meteo; dates beyond that
  use a synthetic fallback.
- The code is intentionally educational and unoptimised for production.

---

## 🛠️ Future Work

- Persist and version models
- Support multiple technicians / multi‑day planning
- Integrate real GIS routing and road conditions
- Add CLI or web UI
- Write unit/integration tests

---

## 📄 License & Contributing

Feel free to fork and extend! No licence is currently declared – add one if you
plan to distribute.

Contributions are welcome; open a PR with bug fixes, features, or better
documentation.

---

> **Weather‑aware scheduling made simple** – a small prototype with
practical utility and plenty of room to grow ☀️🌧️⚡

