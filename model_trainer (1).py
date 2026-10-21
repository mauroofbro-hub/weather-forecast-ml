# --------------------------------------------------------------
# model_trainer.py (BOOSTED ACCURACY VERSION)
# Run: python model_trainer.py
# --------------------------------------------------------------

import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Try to import SMOTE, fall back gracefully if not available
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("WARNING: 'imblearn' not installed. Continuing WITHOUT SMOTE balancing.")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Increase dataset size for more stable learning
NUM_RECORDS = 8000   # was 2600

# ---------------------------
# State and Crop mappings
# ---------------------------
STATES_AND_CITIES = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Karnataka": ["Bengaluru", "Mysore", "Mangalore", "Hubli", "Belgaum"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut"],
    "Punjab": ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Bikaner", "Ajmer"],
    "West Bengal": ["Kolkata", "Howrah", "Siliguri", "Durgapur", "Kharagpur"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Purnia"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Hazaribagh"],
    "Andhra Pradesh": ["Vijayawada", "Visakhapatnam", "Guntur", "Tirupati", "Kurnool"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Khammam", "Karimnagar"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur"],
    "Haryana": ["Gurgaon", "Faridabad", "Panipat", "Karnal", "Hisar"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar", "Jorhat", "Tezpur"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Korba", "Bilaspur", "Durg"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Himachal Pradesh": ["Shimla", "Dharamshala", "Kullu", "Solan", "Bilaspur_HP"],
    "Goa": ["Panaji", "Margao", "Vasco", "Ponda", "Mapusa"]
}

CROPS = [
    "Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Groundnut", "Pulses",
    "Millets", "Tea", "Coffee", "Mustard", "Soybean", "Potato", "Onion",
    "Tomato", "Chilli", "Banana", "Coconut", "Rubber", "Barley"
]

MONTHS = list(range(1, 13))
MONTH_NAME = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# ---------------------------
# Crop preferred month windows
# ---------------------------
CROP_PREFERRED_MONTHS = {
    "Rice": [6,7,8,9],
    "Wheat": [11,12,1,2],
    "Maize": [6,7,8,9,10],
    "Sugarcane": [10,11,12,1,2,3],
    "Cotton": [6,7,8,9,10],
    "Groundnut": [7,8,9,10],
    "Pulses": [10,11,12,1,2],
    "Millets": [6,7,8,9],
    "Tea": [3,4,5,6,7,8],
    "Coffee": [10,11,12,1,2,3],
    "Mustard": [10,11,12],
    "Soybean": [6,7,8,9],
    "Potato": [10,11,12,1,2],
    "Onion": [9,10,11],
    "Tomato": [9,10,11,12,1],
    "Chilli": [9,10,11,12],
    "Banana": MONTHS,
    "Coconut": MONTHS,
    "Rubber": [4,5,6,7,8,9],
    "Barley": [11,12,1,2]
}

CROP_STATE_BIAS = defaultdict(list)
CROP_STATE_BIAS.update({
    "Rice": ["Bihar","West Bengal","Assam","Odisha","Andhra Pradesh"],
    "Wheat": ["Punjab","Haryana","Uttar Pradesh","Rajasthan"],
    "Tea": ["Assam","West Bengal"],
    "Coffee": ["Karnataka","Kerala","Tamil Nadu"],
    "Coconut": ["Kerala","Tamil Nadu","Karnataka","Andhra Pradesh","Goa"],
    "Rubber": ["Kerala","Tamil Nadu"],
    "Cotton": ["Maharashtra","Gujarat","Telangana"],
    "Mustard": ["Rajasthan","Haryana","Punjab"],
    "Soybean": ["Madhya Pradesh","Maharashtra"],
})

# ---------------------------
# Crop–soil preferences (for better signal)
# ---------------------------
SOIL_TYPES = ["Loamy", "Alluvial", "Black", "Sandy", "Laterite"]

CROP_SOIL_PREF = {
    "Rice": ("Alluvial", (5.5, 7.0)),
    "Wheat": ("Loamy", (6.0, 7.5)),
    "Maize": ("Loamy", (5.8, 7.2)),
    "Sugarcane": ("Alluvial", (6.0, 7.5)),
    "Cotton": ("Black", (6.0, 8.0)),
    "Groundnut": ("Sandy", (5.5, 7.0)),
    "Pulses": ("Loamy", (6.0, 7.5)),
    "Millets": ("Sandy", (5.0, 7.0)),
    "Tea": ("Laterite", (4.5, 5.5)),
    "Coffee": ("Laterite", (5.0, 6.5)),
    "Mustard": ("Loamy", (6.0, 7.5)),
    "Soybean": ("Black", (6.0, 7.5)),
    "Potato": ("Alluvial", (5.0, 6.5)),
    "Onion": ("Alluvial", (6.0, 7.5)),
    "Tomato": ("Loamy", (5.5, 7.5)),
    "Chilli": ("Loamy", (6.0, 7.5)),
    "Banana": ("Alluvial", (5.5, 7.5)),
    "Coconut": ("Laterite", (5.5, 7.5)),
    "Rubber": ("Laterite", (4.5, 6.0)),
    "Barley": ("Loamy", (6.0, 7.5))
}

# ---------------------------
# Helper functions
# ---------------------------
def random_state_city():
    state = random.choice(list(STATES_AND_CITIES.keys()))
    city = random.choice(STATES_AND_CITIES[state])
    return state, city

def generate_best_month_for_crop(crop, state):
    """Stronger, cleaner rule so model can learn better."""
    pref = CROP_PREFERRED_MONTHS.get(crop, MONTHS)

    # more aggressive weighting: preferred months much more likely
    weights = []
    for m in MONTHS:
        if m in pref:
            w = 5.0    # high weight inside preferred window
        else:
            w = 0.2    # small but non-zero chance outside
        weights.append(w)

    # add state bias
    if state in CROP_STATE_BIAS.get(crop, []):
        weights = [w * (1.3 if m in pref else 1.0) for m, w in zip(MONTHS, weights)]

    total = sum(weights)
    probs = [w / total for w in weights]

    return int(np.random.choice(MONTHS, p=probs))

def generate_temp_rain(month, state):
    """Slightly less noisy temperature & rainfall."""
    base_temp = {
        1: 22, 2: 25, 3: 28, 4: 32, 5: 33, 6: 30,
        7: 28, 8: 28, 9: 29, 10: 30, 11: 27, 12: 24
    }
    base_rain = {
        1: 10, 2: 15, 3: 30, 4: 50, 5: 60, 6: 150,
        7: 200, 8: 180, 9: 120, 10: 40, 11: 20, 12: 10
    }
    t = base_temp[month] + np.random.normal(0, 1.0)
    r = max(0, base_rain[month] + np.random.normal(0, 15.0))
    return round(t, 1), round(r, 1)

# ---------------------------
# Build synthetic dataset
# ---------------------------
records = []
for i in range(NUM_RECORDS):
    crop = random.choice(CROPS)
    state, city = random_state_city()
    best_month = generate_best_month_for_crop(crop, state)
    temp, rain = generate_temp_rain(best_month, state)

    pref = CROP_PREFERRED_MONTHS.get(crop, [])
    base_score = 55  # slightly higher base
    if best_month in pref:
        base_score += random.randint(15, 30)
    if state in CROP_STATE_BIAS.get(crop, []):
        base_score += random.randint(5, 15)

    # less noise in suitability
    suitability = int(min(100, max(10, base_score + np.random.normal(0, 3))))

    # soil and pH linked to crop
    soil_pref, ph_range = CROP_SOIL_PREF.get(
        crop,
        (random.choice(SOIL_TYPES), (5.5, 7.5))
    )
    # 85% chance to follow preferred soil, 15% random
    soil_type = soil_pref if random.random() < 0.85 else random.choice(SOIL_TYPES)
    ph_center = (ph_range[0] + ph_range[1]) / 2
    ph = round(
        min(8.5, max(4.0, np.random.normal(ph_center, 0.3))),
        1
    )

    duration_days = random.randint(60, 180)

    records.append({
        "crop": crop,
        "state": state,
        "city": city,
        "best_month": best_month,
        "month_name": MONTH_NAME[best_month],
        "avg_temp_C": temp,
        "avg_rainfall_mm": rain,
        "suitability_score": suitability,
        "soil_type": soil_type,
        "soil_ph": ph,
        "duration_days": duration_days
    })

df = pd.DataFrame(records)
df.to_csv("synthetic_data.csv", index=False)
print(f"Saved dataset -> synthetic_data.csv (rows: {len(df)})")

# ---------------------------
# Label Encoding
# ---------------------------
encoders = {}
encoders["crop"] = LabelEncoder().fit(df["crop"])
encoders["state"] = LabelEncoder().fit(df["state"])
encoders["city"] = LabelEncoder().fit(df["city"])
encoders["crop_label"] = LabelEncoder().fit(df["crop"])
encoders["soil_type"] = LabelEncoder().fit(df["soil_type"])

# ---------------------------
# Model 1: Predict best month (crop, state, city, climate)
# ---------------------------
X1 = pd.DataFrame({
    "crop": encoders["crop"].transform(df["crop"]),
    "state": encoders["state"].transform(df["state"]),
    "city": encoders["city"].transform(df["city"]),
    "avg_temp_C": df["avg_temp_C"],
    "avg_rainfall_mm": df["avg_rainfall_mm"],
    "suitability_score": df["suitability_score"],
})

y1 = df["best_month"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=RANDOM_STATE, stratify=y1
)

if HAS_SMOTE:
    sm1 = SMOTE(random_state=RANDOM_STATE)
    X1_train_res, y1_train_res = sm1.fit_resample(X1_train, y1_train)
    print("Model 1 - SMOTE applied:", X1_train.shape, "->", X1_train_res.shape)
else:
    X1_train_res, y1_train_res = X1_train, y1_train

model1 = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
model1.fit(X1_train_res, y1_train_res)

y1_pred = model1.predict(X1_test)
print("\nModel 1 (crop,state,city -> best_month)")
print("Accuracy:", round(accuracy_score(y1_test, y1_pred), 4))
print(classification_report(y1_test, y1_pred, zero_division=0))

# ---------------------------
# Model 2: Predict best crop (month, state, city, climate, soil)
# ---------------------------
X2 = pd.DataFrame({
    "month": df["best_month"],
    "state": encoders["state"].transform(df["state"]),
    "city": encoders["city"].transform(df["city"]),
    "avg_temp_C": df["avg_temp_C"],
    "avg_rainfall_mm": df["avg_rainfall_mm"],
    "suitability_score": df["suitability_score"],
    "soil_type": encoders["soil_type"].transform(df["soil_type"]),
    "soil_ph": df["soil_ph"],
})

y2 = encoders["crop_label"].transform(df["crop"])

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=RANDOM_STATE, stratify=y2
)

if HAS_SMOTE:
    sm2 = SMOTE(random_state=RANDOM_STATE)
    X2_train_res, y2_train_res = sm2.fit_resample(X2_train, y2_train)
    print("Model 2 - SMOTE applied:", X2_train.shape, "->", X2_train_res.shape)
else:
    X2_train_res, y2_train_res = X2_train, y2_train

model2 = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
model2.fit(X2_train_res, y2_train_res)

y2_pred = model2.predict(X2_test)
print("\nModel 2 (month,state,city -> crop)")
print("Accuracy:", round(accuracy_score(y2_test, y2_pred), 4))
print(classification_report(y2_test, y2_pred, zero_division=0))

# ---------------------------
# Save models & encoders
# ---------------------------
joblib.dump(model1, "crop_month_model.pkl")
joblib.dump(model2, "month_crop_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\nSaved: crop_month_model.pkl, month_crop_model.pkl, encoders.pkl")
print("Training Completed Successfully!")
