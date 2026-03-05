# Core libraries
import math
import itertools
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pd.set_option("display.max_columns", 200)

def simulate_weather_rows(n_rows: int = 5000) -> pd.DataFrame:
    """Simulate hourly weather samples for a small region with micro-variation.
    Features are inspired by typical forecast outputs.
    This function is used to generate data for *training the weather risk classification model*.
    """
    locations = ["Rongai_Town", "Rural_Rongai", "Open_Field", "Residential_Area"]
    hours = list(range(6, 20))  # 6:00–19:00
    seasons = ["Dry", "Wet"]

    rows = []
    for _ in range(n_rows):
        loc = random.choice(locations)
        hour = random.choice(hours)
        season = random.choices(seasons, weights=[0.55, 0.45])[0]

        # Baseline probabilities and correlations
        base_rain = 0.10 if season == "Dry" else 0.45
        # Afternoon convective showers in wet season
        hour_bump = 0.10 if (season == "Wet" and hour >= 13) else 0.0
        # Open field tends to be riskier (exposure + microclimate assumption)
        loc_bump = 0.10 if loc in ["Open_Field", "Rural_Rongai"] else 0.0

        rain_prob = np.clip(np.random.normal(base_rain + hour_bump + loc_bump, 0.18), 0, 1)
        wind_kph = np.clip(np.random.normal(12 + 18 * rain_prob + (5 if loc == "Open_Field" else 0), 8), 0, 80)
        thunder_prob = np.clip(np.random.normal(0.05 + 0.65 * rain_prob, 0.15), 0, 1)
        temp_c = np.clip(np.random.normal(27 - 5 * rain_prob + (1 if season == "Dry" else 0), 2.5), 16, 36)
        humidity = np.clip(np.random.normal(55 + 35 * rain_prob + (10 if season == "Wet" else 0), 10), 30, 98)
        visibility_km = np.clip(np.random.normal(12 - 7 * rain_prob, 2), 1, 15)

        # Risk label logic (safe/risky/unsafe) + a bit of randomness
        # Unsafe: high rain OR high thunder OR very low visibility
        unsafe_score = (rain_prob > 0.75) + (thunder_prob > 0.55) + (visibility_km < 4.0) + (wind_kph > 45)
        risky_score = (rain_prob > 0.45) + (thunder_prob > 0.25) + (visibility_km < 7.0) + (wind_kph > 30)

        # Resolve label
        if unsafe_score >= 2:
            risk = "unsafe"
        elif risky_score >= 2:
            risk = "risky"
        else:
            risk = "safe"

        # Inject mild label noise (real forecasts are imperfect)
        if random.random() < 0.04:
            risk = random.choice(["safe", "risky", "unsafe"])

        rows.append({
            "location": loc,
            "hour": hour,
            "season": season,
            "rain_prob": float(rain_prob),
            "wind_kph": float(wind_kph),
            "thunder_prob": float(thunder_prob),
            "temp_c": float(temp_c),
            "humidity": float(humidity),
            "visibility_km": float(visibility_km),
            "risk_label": risk
        })
    df = pd.DataFrame(rows)

    # Add some missing values to demonstrate preprocessing
    for col in ["wind_kph", "visibility_km", "humidity"]:
        mask = np.random.rand(len(df)) < 0.01
        df.loc[mask, col] = np.nan

    return df

training_weather_df = simulate_weather_rows(6000)
training_weather_df.to_csv('rongai_weather_data.csv', index=False)

# Ensure `weather_df` points to the training data for subsequent steps
weather_df = training_weather_df

X = weather_df.drop(columns=["risk_label"])
y = weather_df["risk_label"]

cat_cols = ["location", "season"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ],
    remainder="drop"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.22, random_state=RANDOM_SEED, stratify=y
)

X_train.shape, X_test.shape


fig = plt.figure(figsize=(7,4))
weather_df["risk_label"].value_counts().plot(kind="bar")
plt.title("Risk label distribution")
plt.xlabel("risk_label")
plt.ylabel("count")
plt.show()

weather_df[["rain_prob","thunder_prob","wind_kph","visibility_km","temp_c","humidity"]].describe().T


# Rain probability vs risk label
fig = plt.figure(figsize=(7,4))
for label in ["safe", "risky", "unsafe"]:
    subset = weather_df[weather_df["risk_label"] == label]["rain_prob"]
    plt.hist(subset, bins=30, alpha=0.5, label=label)
plt.title("Distribution of rain_prob by risk label")
plt.xlabel("rain_prob")
plt.ylabel("count")
plt.legend()
plt.show()

weather_df_fe = weather_df.copy()
weather_df_fe["is_afternoon"] = (weather_df_fe["hour"] >= 13).astype(int)
weather_df_fe["exposure_hint"] = weather_df_fe["location"].map({
    "Open_Field": 2,
    "Rural_Rongai": 1,
    "Rongai_Town": 0,
    "Residential_Area": 0,
}).astype(int)

X_fe = weather_df_fe.drop(columns=["risk_label"])
y_fe = weather_df_fe["risk_label"]

cat_cols_fe = ["location", "season"]
num_cols_fe = [c for c in X_fe.columns if c not in cat_cols_fe]

preprocess_fe = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]), num_cols_fe),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols_fe),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X_fe, y_fe, test_size=0.22, random_state=RANDOM_SEED, stratify=y_fe
)

X_train.shape

logreg = Pipeline(steps=[
    ("prep", preprocess_fe),
    ("model", LogisticRegression(max_iter=2000, n_jobs=None))
])

rf = Pipeline(steps=[
    ("prep", preprocess_fe),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("Done.")

def evaluate(model, name: str):
    y_pred = model.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=["safe","risky","unsafe"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["safe","risky","unsafe"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix – {name}")
    plt.show()

evaluate(logreg, "Logistic Regression (baseline)")
evaluate(rf, "Random Forest (final)")


# Get feature names from preprocessing
ohe = rf.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(cat_cols_fe).tolist()
feature_names = num_cols_fe + cat_feature_names

importances = rf.named_steps["model"].feature_importances_
fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

fi.head(15)


@dataclass(frozen=True)
class Task:
    name: str
    location: str
    priority: str  # High/Medium/Low
    duration_h: int
    is_outdoor: bool

PRIORITY_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}

tasks: List[Task] = [
    Task("Transformer inspection", "Rural_Rongai", "High", 2, True),
    Task("Pole repair", "Open_Field", "Medium", 3, True),
    Task("Meter replacement", "Residential_Area", "Low", 1, False),
    Task("Substation visual check", "Rongai_Town", "High", 1, False),
    Task("Line tightening", "Open_Field", "Medium", 2, True),
    Task("Customer safety audit", "Residential_Area", "Low", 1, False),
]

tasks


# Approximate travel times between locations (minutes). In real life you'd use GIS/routing.
locations = ["Rongai_Town","Rural_Rongai","Open_Field","Residential_Area"]
travel_minutes = {
    ("Rongai_Town","Rural_Rongai"): 20,
    ("Rongai_Town","Open_Field"): 30,
    ("Rongai_Town","Residential_Area"): 15,
    ("Rural_Rongai","Open_Field"): 25,
    ("Rural_Rongai","Residential_Area"): 25,
    ("Open_Field","Residential_Area"): 35,
}
def tmin(a,b):
    if a==b: return 0
    return travel_minutes.get((a,b), travel_minutes.get((b,a), 30))

# Build matrix
dist = pd.DataFrame([[tmin(a,b) for b in locations] for a in locations], index=locations, columns=locations)
dist


import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# REAL WEATHER DATA (NAKURU COUNTY) — Open‑Meteo
# =============================================================================

OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/1/forecast"

# Towns inside / strongly associated with Nakuru County.
# For towns that fail geocoding, we provide manual coordinates
NAKURU_COUNTY_TOWNS: List[str] = [
    "Nakuru, Kenya",
    "Naivasha, Kenya",
    "Gilgil, Kenya",
    "Molo, Kenya",  # This one fails geocoding
    "Rongai, Kenya",
    "Njoro, Kenya",
    "Bahati, Kenya",
    "Subukia, Kenya",
]

# Manual coordinates for towns that might fail geocoding
# Format: {town_name: (latitude, longitude)}
MANUAL_COORDINATES = {
    "Molo, Kenya": (-0.2488, 35.7324),  # The coordinates you provided
    # Add more if needed
    "Molo": (-0.2488, 35.7324),  # Also try without "Kenya"
}

def geocode_open_meteo(place: str, *, count: int = 1) -> Optional[Tuple[float, float, str]]:
    """
    Resolve a place name to coordinates using Open‑Meteo geocoding.
    Falls back to manual coordinates if available.
    Returns (lat, lon, resolved_name) or None if no result.
    """
    # First check if we have manual coordinates for this place
    if place in MANUAL_COORDINATES:
        lat, lon = MANUAL_COORDINATES[place]
        print(f"[INFO] Using manual coordinates for {place}: {lat}, {lon}")
        return (lat, lon, place.replace(", Kenya", ""))  # Clean the name
    
    # Also check without "Kenya" suffix
    place_without_country = place.replace(", Kenya", "")
    if place_without_country in MANUAL_COORDINATES:
        lat, lon = MANUAL_COORDINATES[place_without_country]
        print(f"[INFO] Using manual coordinates for {place}: {lat}, {lon}")
        return (lat, lon, place_without_country)
    
    # Try multiple variants of the place name
    variants = [
        place,  # Original
        place.replace(", Kenya", ""),  # Without country
        place.split(",")[0],  # Just the town name
        f"{place.split(',')[0]}, Nakuru, Kenya",  # Add county
    ]
    
    for variant in variants:
        try:
            params = {"name": variant, "count": count, "language": "en", "format": "json"}
            r = requests.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            results = data.get("results") or []
            if results:
                # Filter results to prefer Kenya locations
                kenya_results = [r for r in results if r.get('country', '').lower() == 'kenya']
                if kenya_results:
                    top = kenya_results[0]
                else:
                    top = results[0]
                
                return float(top["latitude"]), float(top["longitude"]), str(top.get("name", place))
        except:
            continue
    
    # If all geocoding attempts fail, try manual coordinates for common misspellings
    print(f"[WARN] Geocoding failed for: {place} — trying fuzzy match...")
    
    # Fuzzy match common Nakuru towns
    fuzzy_matches = {
        "molo": (-0.2488, 35.7324),
        "moloi": (-0.2488, 35.7324),  # Common misspelling
        "mollo": (-0.2488, 35.7324),  # Common misspelling
    }
    
    place_lower = place.lower()
    for key, (lat, lon) in fuzzy_matches.items():
        if key in place_lower:
            print(f"[INFO] Fuzzy matched {place} to {key} with coordinates: {lat}, {lon}")
            return (lat, lon, key.capitalize())
    
    return None

def build_location_coords(towns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Build a dict: location_name -> {latitude, longitude} using live geocoding.
    If a town fails geocoding, we skip it (and warn).
    """
    coords: Dict[str, Dict[str, float]] = {}
    
    # First, try to geocode all towns
    for t in towns:
        out = geocode_open_meteo(t)
        if out is None:
            print(f"[WARN] Geocoding failed for: {t} — skipping.")
            continue
        lat, lon, resolved = out
        # Use a clean, stable key for downstream modeling
        key = resolved.replace(" ", "_")
        coords[key] = {"latitude": lat, "longitude": lon}
    
    # If Molo is still missing, add it manually
    molo_found = any('Molo' in key for key in coords.keys())
    if not molo_found:
        print("[INFO] Adding Molo with manual coordinates")
        coords["Molo"] = {"latitude": -0.2488, "longitude": 35.7324}
    
    return coords

# Build location coordinates
LOCATION_COORDS = build_location_coords(NAKURU_COUNTY_TOWNS)
print("\nSuccessfully geocoded locations:")
for loc, coords in LOCATION_COORDS.items():
    print(f"  {loc}: {coords['latitude']}, {coords['longitude']}")

# Rest of your classes and functions remain the same...
class WeatherAPIClient:
    # ... (keep your existing WeatherAPIClient class)
    def __init__(self, base_url: str = OPEN_METEO_FORECAST_URL):
        self.base_url = base_url

    def get_hourly_forecast(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch hourly forecast. Units:
        - temperature_2m: °C
        - relative_humidity_2m: %
        - precipitation_probability: %
        - wind_speed_10m: m/s  (we convert to kph because our model expects kph)
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation_probability",
                "wind_speed_10m",
            ],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Africa/Nairobi",
        }

        try:
            r = requests.get(self.base_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            hourly = data.get("hourly")
            if not hourly:
                return pd.DataFrame()

            df = pd.DataFrame(hourly)
            df["time"] = pd.to_datetime(df["time"])
            df["hour"] = df["time"].dt.hour

            # Map Open‑Meteo field names -> model/training feature names
            df = df.rename(
                columns={
                    "precipitation_probability": "rain_prob",  # percent
                    "wind_speed_10m": "wind_kph",              # convert below
                    "temperature_2m": "temp_c",
                    "relative_humidity_2m": "humidity",
                }
            )

            # Convert wind from m/s -> kph
            df["wind_kph"] = df["wind_kph"] * 3.6

            # ---------------------------------------------------------------------
            # Derived / proxy features (Open‑Meteo doesn't provide them directly here)
            # ---------------------------------------------------------------------
            # thunder_prob: we approximate from rain probability.
            # visibility_km: we approximate from rain probability (wetter -> lower vis).
            # These are *proxies* used only because our training schema expects them.
            df["thunder_prob"] = (df["rain_prob"] / 100.0) * 0.30
            df["visibility_km"] = 12.0 - ((df["rain_prob"] / 100.0) * 7.0)

            # Ensure valid ranges
            df["visibility_km"] = df["visibility_km"].clip(lower=1.0, upper=15.0)
            df["thunder_prob"] = df["thunder_prob"].clip(lower=0.0, upper=1.0)

            # Convert rain_prob percent -> 0..1 if training expects fraction?
            # In this notebook's simulated data, rain_prob is 0..1. We'll match that:
            df["rain_prob"] = (df["rain_prob"] / 100.0).clip(0.0, 1.0)

            return df
            
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP error for coordinates ({latitude}, {longitude}): {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"[ERROR] Unexpected error for coordinates ({latitude}, {longitude}): {e}")
            return pd.DataFrame()

def simulate_fallback_forecast(
    location_coords: Dict[str, Dict[str, float]],
    date_label: str,
    work_hours: Tuple[int, int] = (8, 17),
) -> pd.DataFrame:
    """
    Fallback generator (only used if the API is unreachable or date is invalid).
    Creates a deterministic pseudo‑forecast for each location & work hour.
    """
    seed = int(date_label.replace("-", "")) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    rows = []
    for loc in location_coords.keys():
        for h in range(work_hours[0], work_hours[1]):
            # Simple diurnal + randomness pattern
            base = 0.15 + 0.10 * (h >= 13)
            rain_prob = np.clip(base + rng.normal(0, 0.08), 0, 1)
            wind_kph = np.clip(8 + rng.normal(0, 4), 0, 40)
            temp_c = np.clip(20 + rng.normal(0, 3), 10, 35)
            humidity = np.clip(55 + rng.normal(0, 15), 10, 100)
            thunder_prob = np.clip(rain_prob * 0.30, 0, 1)
            visibility_km = np.clip(12 - rain_prob * 7, 1, 15)

            rows.append(
                dict(
                    hour=h,
                    rain_prob=rain_prob,
                    wind_kph=wind_kph,
                    thunder_prob=thunder_prob,
                    temp_c=temp_c,
                    humidity=humidity,
                    visibility_km=visibility_km,
                    location=loc,
                )
            )

    df = pd.DataFrame(rows)
    return df

def get_nakuru_county_forecast(
    date_label: Optional[str] = None,
    rf_model=None  # Add parameter to pass the trained model
) -> pd.DataFrame:
    """
    Fetch Nakuru County hourly forecasts (work hours) from Open‑Meteo,
    then run the trained risk model to produce pred_risk per hour/location.

    Requirements:
    - rf_model: trained classifier must be passed to this function
    """
    if date_label is None:
        date_label = datetime.now().strftime("%Y-%m-%d")
    else:
        # Validate that the requested date is not too far in the future
        requested_date = datetime.strptime(date_label, "%Y-%m-%d")
        today = datetime.now()
        max_forecast_days = 16  # Open-Meteo typical max forecast range
        
        if requested_date > today + timedelta(days=max_forecast_days):
            print(f"[WARN] Date {date_label} is beyond Open-Meteo's forecast range (max {max_forecast_days} days). Using fallback data.")
            forecast_df = simulate_fallback_forecast(LOCATION_COORDS, date_label)
            forecast_df["season"] = "Wet"
            forecast_df["is_afternoon"] = (forecast_df["hour"] >= 13).astype(int)
            forecast_df["exposure_hint"] = 1
            
            # If we have a model, make predictions
            if rf_model is not None:
                model_features = [
                    "hour",
                    "rain_prob",
                    "wind_kph",
                    "thunder_prob",
                    "temp_c",
                    "humidity",
                    "visibility_km",
                    "is_afternoon",
                    "exposure_hint",
                    "location",
                    "season",
                ]
                X_forecast = forecast_df[model_features]
                forecast_df["pred_risk"] = rf_model.predict(X_forecast)
            
            keep_cols = ["hour", "location", "rain_prob", "wind_kph", "temp_c", "humidity", "pred_risk"]
            forecast_df = forecast_df[keep_cols].sort_values(["location", "hour"]).reset_index(drop=True)
            return forecast_df

    api = WeatherAPIClient()
    all_rows = []

    for loc_name, coords in LOCATION_COORDS.items():
        df_loc = api.get_hourly_forecast(
            latitude=coords["latitude"],
            longitude=coords["longitude"],
            start_date=date_label,
            end_date=date_label,
        )
        if df_loc.empty:
            print(f"[WARN] No data received for {loc_name}")
            continue

        # Keep typical field technician work hours (08:00–16:59)
        df_loc = df_loc[(df_loc["hour"] >= 8) & (df_loc["hour"] < 17)].copy()
        
        if df_loc.empty:
            print(f"[WARN] No work hours data for {loc_name}")
            continue
            
        df_loc["location"] = loc_name

        # Season is a categorical feature used during training.
        # For a production system, you'd map season by month or use historical climatology.
        df_loc["season"] = "Wet"  # simple default for Nakuru; adjust if desired

        # Feature engineering parity with training pipeline
        df_loc["is_afternoon"] = (df_loc["hour"] >= 13).astype(int)

        # exposure_hint is a rough proxy: urban centers more sheltered than open/rural.
        # With real GIS metadata, this would be computed more accurately.
        df_loc["exposure_hint"] = 1
        if "Nakuru" in loc_name or "Naivasha" in loc_name:
            df_loc["exposure_hint"] = 0
        if "Subukia" in loc_name or "Molo" in loc_name:
            df_loc["exposure_hint"] = 1

        all_rows.append(df_loc)

    if not all_rows:
        print("[WARN] Could not fetch Open‑Meteo data. Using deterministic fallback forecast.")
        forecast_df = simulate_fallback_forecast(LOCATION_COORDS, date_label)
        forecast_df["season"] = "Wet"
        forecast_df["is_afternoon"] = (forecast_df["hour"] >= 13).astype(int)
        forecast_df["exposure_hint"] = 1
    else:
        forecast_df = pd.concat(all_rows, ignore_index=True)

    # Align to the model's expected features (same order used in training)
    model_features = [
        "hour",
        "rain_prob",
        "wind_kph",
        "thunder_prob",
        "temp_c",
        "humidity",
        "visibility_km",
        "is_afternoon",
        "exposure_hint",
        "location",
        "season",
    ]
    
    # Ensure all required columns exist
    for col in model_features:
        if col not in forecast_df.columns:
            print(f"[WARN] Missing column {col}, adding with default value")
            if col in ["rain_prob", "thunder_prob"]:
                forecast_df[col] = 0.0
            elif col in ["wind_kph", "temp_c", "humidity", "visibility_km"]:
                forecast_df[col] = 0.0
            elif col in ["is_afternoon", "exposure_hint"]:
                forecast_df[col] = 0
            elif col == "season":
                forecast_df[col] = "Wet"
    
    X_forecast = forecast_df[model_features]

    # Predict risk using the trained Random Forest model
    if rf_model is not None:
        forecast_df["pred_risk"] = rf_model.predict(X_forecast)
    else:
        print("[WARN] No model provided. Adding placeholder pred_risk values.")
        forecast_df["pred_risk"] = 0.5  # Placeholder value

    # Keep the columns the scheduler needs
    keep_cols = ["hour", "location", "rain_prob", "wind_kph", "temp_c", "humidity", "pred_risk"]
    forecast_df = forecast_df[keep_cols].sort_values(["location", "hour"]).reset_index(drop=True)

    return forecast_df

# Example usage:
# First, make sure you have your trained model available
# For demonstration, let's create a dummy model if you don't have one
try:
    # Try to use your existing rf model
    rf
except NameError:
    print("[WARN] No 'rf' model found. Creating a dummy model for demonstration.")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    # You would normally train this model here

# Now fetch forecast for a valid date (use a past date since we're in 2026)
# Use a date that actually exists in Open-Meteo's database
valid_date = "2024-03-05"  # Use a past date instead of future date
print(f"\nFetching forecast for: {valid_date}")

forecast_today = get_nakuru_county_forecast(
    date_label=valid_date,  # Use a past date
    rf_model=rf  # Pass your trained model
)

if not forecast_today.empty:
    print("\nForecast data retrieved successfully:")
    print(forecast_today.head())
else:
    print("\nNo forecast data available.")


# Visualize rain probability over the workday per location (Open‑Meteo)
pivot = forecast_today.pivot_table(index="hour", columns="location", values="rain_prob")

fig = plt.figure(figsize=(10, 4))
for col in pivot.columns:
    plt.plot(pivot.index, pivot[col], label=col)

plt.title("Open‑Meteo rain probability by hour (Nakuru County)")
plt.xlabel("hour")
plt.ylabel("rain_prob (0–1)")
plt.legend(ncol=2, fontsize=8)
plt.show()

risk_pivot = forecast_today.pivot_table(index="hour", columns="location", values="pred_risk", aggfunc="first")
risk_pivot


RISK_PENALTY = {"safe": 0.0, "risky": 1.0, "unsafe": 5.0}  # unsafe mostly handled as hard constraint

def hourly_risk(location: str, hour: int) -> str:
    row = forecast_today[(forecast_today["location"] == location) & (forecast_today["hour"] == hour)]
    if row.empty:
        return "risky"  # conservative fallback
    return row.iloc[0]["pred_risk"]

def task_block_hours(start_hour: int, duration_h: int) -> List[int]:
    return list(range(start_hour, start_hour + duration_h))

def schedule_score(order: List[Task], start_hour=8, end_hour=17) -> Tuple[float, Dict]:
    current_hour = start_hour
    current_loc = "Rongai_Town"  # assume technician starts in town
    completed = []
    postponed = []
    total_score = 0.0
    total_travel_min = 0
    explanations = []

    for task in order:
        # travel time -> convert to hours (rounded up to next hour block for simplicity)
        travel = tmin(current_loc, task.location)
        travel_h = math.ceil(travel / 60)
        if current_hour + travel_h >= end_hour:
            postponed.append(task)
            explanations.append(f"Postponed '{task.name}' (no time after travel).")
            continue

        if travel_h > 0:
            total_travel_min += travel
            current_hour += travel_h
            explanations.append(f"Travel to {task.location} (+{travel} min ≈ {travel_h}h).")

        if current_hour + task.duration_h > end_hour:
            postponed.append(task)
            explanations.append(f"Postponed '{task.name}' (insufficient remaining hours).")
            continue

        hours = task_block_hours(current_hour, task.duration_h)
        risks = [hourly_risk(task.location, h) for h in hours]

        # Hard constraint: do not do outdoor work during unsafe hours
        if task.is_outdoor and any(r == "unsafe" for r in risks):
            postponed.append(task)
            explanations.append(
                f"Postponed '{task.name}' because forecast is UNSAFE at {task.location} during {hours}."
            )
            continue

        # Reward
        reward = PRIORITY_WEIGHT[task.priority] * task.duration_h

        # Risk penalty (for outdoor tasks primarily; indoor tasks are less affected)
        risk_cost = 0.0
        if task.is_outdoor:
            risk_cost = sum(RISK_PENALTY[r] for r in risks)

        # Moderate-risk preference: schedule high-priority earlier
        early_bonus = 0.0
        if task.priority == "High" and current_hour <= 11 and any(r == "risky" for r in risks):
            early_bonus = 0.5

        total_score += reward + early_bonus - risk_cost
        completed.append((task, current_hour, hours, risks, reward, risk_cost, early_bonus))
        explanations.append(
            f"Scheduled '{task.name}' at {current_hour}:00 for {task.duration_h}h | risks={risks} | "
            f"reward={reward:.1f} - risk_cost={risk_cost:.1f} + early_bonus={early_bonus:.1f}"
        )
        current_hour += task.duration_h
        current_loc = task.location

    # Travel penalty (soft)
    total_score -= total_travel_min / 60 * 0.5  # 0.5 points per travel hour
    details = {
        "completed": completed,
        "postponed": postponed,
        "travel_minutes": total_travel_min,
        "explanations": explanations
    }
    return total_score, details


def naive_order(tasks: List[Task]) -> List[Task]:
    return sorted(tasks, key=lambda t: (PRIORITY_WEIGHT[t.priority], t.duration_h), reverse=True)

def beam_search_schedule(tasks: List[Task], beam_width: int = 10) -> Tuple[List[Task], float, Dict]:
    beams = [([], tasks)]
    best = (None, -1e9, None)

    for _step in range(len(tasks)):
        new_beams = []
        for prefix, remaining in beams:
            for t in remaining:
                new_prefix = prefix + [t]
                new_remaining = [x for x in remaining if x != t]
                score, details = schedule_score(new_prefix)
                new_beams.append((new_prefix, new_remaining, score, details))

        # keep best partials by current score
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = [(p, r) for (p, r, s, d) in new_beams[:beam_width]]

        # update global best among complete schedules encountered
        for (p, r, s, d) in new_beams:
            if not r and s > best[1]:
                best = (p, s, d)

    return best

baseline = naive_order(tasks)
base_score, base_details = schedule_score(baseline)

ai_order, ai_score, ai_details = beam_search_schedule(tasks, beam_width=12)

base_score, ai_score, [t.name for t in baseline], [t.name for t in ai_order]

print("=== Baseline schedule ===")
for line in base_details["explanations"]:
    print("-", line)
print(f"Baseline total score: {base_score:.2f} | travel={base_details['travel_minutes']} min")
print()

print("=== AI schedule ===")
for line in ai_details["explanations"]:
    print("-", line)
print(f"AI total score: {ai_score:.2f} | travel={ai_details['travel_minutes']} min")

def simulate_execution(details: Dict, risky_fail_p=0.25, unsafe_fail_p=0.85) -> Dict:
    completed_tasks = 0
    weighted_work = 0
    disruptions = 0

    for (task, start_hour, hours, risks, reward, risk_cost, early_bonus) in details["completed"]:
        fail_p = 0.0
        if task.is_outdoor:
            if any(r == "unsafe" for r in risks):
                fail_p = unsafe_fail_p
            elif any(r == "risky" for r in risks):
                fail_p = risky_fail_p

        if random.random() < fail_p:
            disruptions += 1
        else:
            completed_tasks += 1
            weighted_work += PRIORITY_WEIGHT[task.priority] * task.duration_h

    return {
        "completed_tasks": completed_tasks,
        "weighted_work": weighted_work,
        "disruptions": disruptions
    }

def monte_carlo_compare(n_trials=500) -> pd.DataFrame:
    rows = []
    for _ in range(n_trials):
        b = simulate_execution(base_details)
        a = simulate_execution(ai_details)
        rows.append({
            "baseline_completed": b["completed_tasks"],
            "ai_completed": a["completed_tasks"],
            "baseline_weighted_work": b["weighted_work"],
            "ai_weighted_work": a["weighted_work"],
            "baseline_disruptions": b["disruptions"],
            "ai_disruptions": a["disruptions"],
        })
    return pd.DataFrame(rows)

mc = monte_carlo_compare(800)
mc.describe().T

# Summaries
summary = pd.DataFrame({
    "metric": ["avg tasks completed", "avg weighted work", "avg disruptions (lower is better)"],
    "baseline": [mc["baseline_completed"].mean(), mc["baseline_weighted_work"].mean(), mc["baseline_disruptions"].mean()],
    "ai": [mc["ai_completed"].mean(), mc["ai_weighted_work"].mean(), mc["ai_disruptions"].mean()],
})
summary


fig = plt.figure(figsize=(7,4))
plt.hist(mc["baseline_weighted_work"], bins=25, alpha=0.5, label="baseline")
plt.hist(mc["ai_weighted_work"], bins=25, alpha=0.5, label="ai")
plt.title("Monte Carlo distribution: weighted work completed")
plt.xlabel("weighted_work")
plt.ylabel("count")
plt.legend()
plt.show()


def plan_day(tasks: List[Task], forecast_df: pd.DataFrame, beam_width: int = 12) -> Dict:
    """Plan a day given tasks and a forecast dataframe with predicted risk per hour/location."""
    global forecast_today
    old = forecast_today
    forecast_today = forecast_df.copy()
    try:
        order, score, details = beam_search_schedule(tasks, beam_width=beam_width)
        return {"order": order, "score": score, "details": details}
    finally:
        forecast_today = old

demo_result = plan_day(tasks, forecast_today, beam_width=12)
[t.name for t in demo_result["order"]], demo_result["score"]


