from flask import Flask, render_template, request, redirect, flash, session, url_for
import json
import os
import pandas as pd
import random
import mysql.connector
from werkzeug.security import check_password_hash, generate_password_hash
import requests
from datetime import datetime, timedelta
import joblib
import numpy as np


app = Flask(__name__)
app.secret_key = "1122"

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_MODEL_PATH = os.path.join(BASE_DIR, "month_crop_model.pkl")
MONTH_MODEL_PATH = os.path.join(BASE_DIR, "crop_month_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")
GUIDELINES_PATH = os.path.join(BASE_DIR, "guidelines.json")

# --- LOAD MODELS & DATA ---
try:
    model_for_crop = joblib.load(CROP_MODEL_PATH)
    model_for_month = joblib.load(MONTH_MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    print("ML Models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load ML models: {e}")
    model_for_crop = None
    model_for_month = None
    encoders = None

# --- LOAD GUIDELINES ---
try:
    with open(GUIDELINES_PATH, 'r') as f:
        CROP_GUIDELINES = json.load(f)
except Exception as e:
    print(f"Warning: guidelines.json not found. {e}")
    CROP_GUIDELINES = {}

# --- HELPER: Month Number to Name ---
MONTH_NUM_TO_NAME = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# ---- Helper for safe label encoding (case-insensitive) ----
def safe_encode_label(value, encoder, field_name="value"):
    """
    Try to find a case-insensitive match for `value`
    in encoder.classes_. Raises ValueError if not found.
    """
    if not value:
        raise ValueError(f"{field_name} is empty")

    value_stripped = value.strip().lower()
    for cls in encoder.classes_:
        if cls.lower() == value_stripped:
            return encoder.transform([cls])[0]  # use canonical class string

    raise ValueError(f"{field_name} '{value}' not found in training data. "
                     f"Valid examples: {', '.join(list(encoder.classes_)[:5])}...")


# -------------------------
# CLIMATE ADJUSTMENT TABLES
# -------------------------

BASE_TEMP = {
    1:22, 2:25, 3:28, 4:32, 5:33, 6:30,
    7:28, 8:28, 9:29, 10:30, 11:27, 12:24
}
BASE_RAIN = {
    1:10, 2:15, 3:30, 4:50, 5:60, 6:150,
    7:200, 8:180, 9:120, 10:40, 11:20, 12:10
}

STATE_TEMP_OFFSET = {
    "Karnataka": -1, "Kerala": 0, "Tamil Nadu": +2,
    "Maharashtra": +1, "Gujarat": +2, "Punjab": -3,
    "Haryana": -3, "Uttar Pradesh": -2, "West Bengal": 0,
    "Assam": -1, "Bihar": -2, "Odisha": 0, "Rajasthan": +3,
    "Madhya Pradesh": +1, "Telangana": +1, "Andhra Pradesh": 0, "Goa": 0
}

STATE_RAIN_OFFSET = {
    "Karnataka": +20, "Kerala": +60, "Tamil Nadu": -10,
    "Maharashtra": +10, "Gujarat": -20, "Punjab": -30,
    "Haryana": -30, "Uttar Pradesh": -15, "West Bengal": +20,
    "Assam": +70, "Bihar": -10, "Odisha": +30, "Rajasthan": -50,
    "Madhya Pradesh": -10, "Telangana": -5, "Andhra Pradesh": +10, "Goa": +40
}

CROP_STATE_BIAS = {
    "Rice": ["Bihar","West Bengal","Assam","Odisha","Andhra Pradesh"],
    "Wheat": ["Punjab","Haryana","Uttar Pradesh","Rajasthan"],
    "Tea": ["Assam","West Bengal"],
    "Coffee": ["Karnataka","Kerala","Tamil Nadu"],
    "Coconut": ["Kerala","Tamil Nadu","Karnataka","Andhra Pradesh","Goa"],
    "Rubber": ["Kerala","Tamil Nadu"],
    "Cotton": ["Maharashtra","Gujarat","Telangana"],
    "Mustard": ["Rajasthan","Haryana","Punjab"],
    "Soybean": ["Madhya Pradesh","Maharashtra"],
}

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


def approximate_climate(month_num, state):
    base_temp = BASE_TEMP.get(month_num, 28)
    base_rain = BASE_RAIN.get(month_num, 80)

    temp_adj = STATE_TEMP_OFFSET.get(state, 0)
    rain_adj = STATE_RAIN_OFFSET.get(state, 0)

    temp = base_temp + temp_adj + np.random.normal(0, 1.0)
    rain = base_rain + rain_adj + np.random.normal(0, 10.0)

    return round(temp, 1), max(0, round(rain, 1))

def approximate_soil_and_ph(crop_name):
    if crop_name in CROP_SOIL_PREF:
        soil, ph_range = CROP_SOIL_PREF[crop_name]
    else:
        soil = random.choice(list(encoders["soil_type"].classes_))
        ph_range = (5.5, 7.5)

    soil_enc = encoders["soil_type"].transform([soil])[0]
    ph = round(np.random.uniform(ph_range[0], ph_range[1]), 1)
    return soil_enc, ph

def approximate_suitability(crop_name, state):
    base = 60
    if state in CROP_STATE_BIAS.get(crop_name, []):
        base += random.randint(10, 20)
    return min(100, max(30, base + np.random.normal(0, 5)))


# --- LOAD DATASET FOR LOOKUPS
try:
    # Make sure this path matches your folder structure
    DATA_PATH = os.path.join(BASE_DIR, "synthetic_data.csv")
    synthetic_data_df = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"Warning: synthetic_data.csv not found. Details will be missing. {e}")
    # Create an empty DataFrame to prevent crashes
    synthetic_data_df = pd.DataFrame(columns=['crop', 'state', 'avg_temp_C', 'avg_rainfall_mm', 'soil_type', 'soil_ph', 'duration_days', 'month_name'])

# DATABASE CONNECTION
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",               # Your MySQL password
        database="weatherapp"
    )

# REGISTER ROUTE
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # VALIDATION
        if not name or not email or not password:
            flash("All fields are required!", "error")
            return redirect(url_for("register"))

        if "@" not in email or ".com" not in email:
            flash("Enter a valid email address!", "error")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters long!", "error")
            return redirect(url_for("register"))

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        existing = cursor.fetchone()

        if existing:
            flash("Email already registered!", "error")
            cursor.close()
            conn.close()
            return redirect(url_for("register"))

        # HASH PASSWORD
        hashed_pw = generate_password_hash(password)

        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
            (name, email, hashed_pw)
        )
        conn.commit()

        cursor.close()
        conn.close()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")



# LOGIN ROUTE
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    # Already logged in
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        # VALIDATION
        if "@" not in email or ".com" not in email:
            flash("Invalid email format! Example: user@example.com", "error")
            return redirect(url_for("login"))

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return redirect(url_for("login"))

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        cursor.close()
        db.close()

        if not user:
            flash("No account found with this email.", "error")
            return redirect(url_for("login"))

        # CHECK PASSWORD
        if not check_password_hash(user['password_hash'], password):
            flash("Incorrect password!", "error")
            return redirect(url_for("login"))

        # LOGIN SUCCESS
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]

        flash("Logged in successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")



# DASHBOARD (Protected Route)
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    # ---------- LOGIN PROTECTION ----------
    if "user_id" not in session:
        return redirect("/login")

    API_KEY = "65ec2a1c272bd841e00a8bdd49f1c9d3"  # Your OpenWeatherMap API key
    
    # Default values
    city_name = "Mangalore"
    state_name = "Karnataka"
    lat = None
    lon = None

    # ---------- HANDLE LOCATION INPUT ----------
    if request.method == "POST":
        # Check if coordinates were provided (from geolocation)
        geo_lat = request.form.get("latitude")
        geo_lon = request.form.get("longitude")
        
        # Manual city/state input
        manual_city = request.form.get("city", "").strip()
        manual_state = request.form.get("state", "").strip()
        
        # PRIORITY 1: Use geolocation coordinates if available
        if geo_lat and geo_lon:
            try:
                lat = float(geo_lat)
                lon = float(geo_lon)
                
                # Reverse geocoding to get city and state from coordinates
                REVERSE_GEO_URL = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={API_KEY}"
                geo_response = requests.get(REVERSE_GEO_URL).json()
                
                if geo_response and len(geo_response) > 0:
                    city_name = geo_response[0].get("name", "Unknown")
                    state_name = geo_response[0].get("state", geo_response[0].get("country", "Unknown"))
                
            except (ValueError, KeyError, requests.RequestException) as e:
                print(f"Geolocation error: {e}")
                # Fall back to manual input or defaults
                if manual_city and manual_state:
                    city_name = manual_city
                    state_name = manual_state
        
        # PRIORITY 2: Use manual city/state input if no coordinates
        elif manual_city and manual_state:
            city_name = manual_city
            state_name = manual_state
            lat = None
            lon = None

    # ---------- FETCH WEATHER FROM API ----------
    # Use coordinates if available, otherwise use city name
    if lat and lon:
        WEATHER_URL = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    else:
        WEATHER_URL = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name},{state_name}&units=metric&appid={API_KEY}"
    
    try:
        data = requests.get(WEATHER_URL).json()
        
        # Check if API call was successful
        if data.get("cod") != "200":
            # Fallback to default location on error
            city_name = "Mangalore"
            state_name = "Karnataka"
            WEATHER_URL = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&units=metric&appid={API_KEY}"
            data = requests.get(WEATHER_URL).json()
    
    except requests.RequestException as e:
        print(f"Weather API error: {e}")
        # Use mock data or return error page
        return render_template("error.html", message="Unable to fetch weather data")

    # ---------- CURRENT WEATHER ----------
    current = data["list"][0]
    current_temp = current["main"]["temp"]
    feels_like = current["main"]["feels_like"]
    humidity = current["main"]["humidity"]
    wind_speed = current["wind"]["speed"]
    description = current["weather"][0]["description"].title()
    icon_code = current["weather"][0]["icon"]
    weather_icon = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"

    min_temp = current["main"]["temp_min"]
    max_temp = current["main"]["temp_max"]

    updated_time = datetime.now().strftime("%I:%M %p")

    # ---------- HOURLY WIND SPEED CHART (NEXT 12 HOURS) ----------
    hourly_windspeed = []
    hourly_labels = []

    for entry in data["list"][:12]:
        hourly_windspeed.append(entry["wind"]["speed"])
        hour = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S").strftime("%I %p")
        hourly_labels.append(hour)

    # ---------- HOURLY TEMPERATURE CHART ----------
    hourly_temp_values = []
    hourly_temp_labels = []

    for entry in data["list"][:12]:
        hourly_temp_values.append(entry["main"]["temp"])
        hour = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S").strftime("%I %p")
        hourly_temp_labels.append(hour)

    # ---------- NEXT 5 DAYS TEMPERATURE ----------
    next7_temp_values = []
    next7_temp_labels = []

    # take forecast every 24 hours → index 0, 8, 16, 24, 32
    for i in [0, 8, 16, 24, 32]:
        if i < len(data["list"]):
            entry = data["list"][i]
            next7_temp_values.append(entry["main"]["temp"])

            day_name = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S").strftime("%a")
            next7_temp_labels.append(day_name)
 
    # ---------- RETURN TO DASHBOARD ----------
    return render_template(
        "dashboard.html",

        # LOCATION DATA (for form and display)
        city_name=city_name,
        state_name=state_name,
        lat=lat if lat else "",
        lon=lon if lon else "",

        # WEATHER OVERVIEW
        description=description,
        weather_icon=weather_icon,
        current_temp=current_temp,
        feels_like=feels_like,
        min_temp=min_temp,
        max_temp=max_temp,
        updated_time=updated_time,

        # DONUT VALUES
        humidity=humidity,
        wind_speed=wind_speed,

        # HOURLY WIND CHART
        hourly_windspeed=hourly_windspeed,
        hourly_labels=hourly_labels,

        # HOURLY TEMP CHART
        hourly_temp_values=hourly_temp_values,
        hourly_temp_labels=hourly_temp_labels,

        # NEXT 5 DAYS TEMPERATURE
        next7_temp_values=next7_temp_values,
        next7_temp_labels=next7_temp_labels
    )


@app.route('/month-suggestion', methods=['GET', 'POST'])
def month_suggestion():
    if "user_id" not in session:
        return redirect("/login")

    month_results = []
    upcoming_crops = []
    form_data = {}
    crop_guidelines = None

    if request.method == 'POST':
        crop_name = request.form.get('crop_name', '').strip()
        state = request.form.get('state', '').strip()
        city = request.form.get('city', '').strip()

        form_data = {"crop_name": crop_name, "state": state, "city": city}

        # -------------------------
        # SAFE ENCODING (case-insensitive)
        # -------------------------
        try:
            crop_enc = safe_encode_label(crop_name, encoders['crop'], "Crop")
            state_enc = safe_encode_label(state, encoders['state'], "State")
            city_enc = safe_encode_label(city, encoders['city'], "City")

        except ValueError as e:
            flash(str(e), "error")
            return render_template("month-suggestion.html",
                                   month_results=[],
                                   upcoming_crops=[],
                                   form_data=form_data)

        # -------------------------
        # LOAD GUIDELINES
        # -------------------------
        crop_guidelines = CROP_GUIDELINES.get(crop_name, None)

        # -------------------------
        # CLIMATE + SUITABILITY (realistic inputs)
        # -------------------------
        approx_temp, approx_rain = approximate_climate(6, state)
        suitability = approximate_suitability(crop_name, state)

        # -------------------------
        # MODEL 1 → Top 3 Best Months
        # Correct Feature Order:
        # [crop, state, city, temp, rain, suitability]
        # -------------------------
        X1 = np.array([[crop_enc,
                        state_enc,
                        city_enc,
                        approx_temp,
                        approx_rain,
                        suitability]])

        month_probs = model_for_month.predict_proba(X1)[0]
        top3_idx = np.argsort(month_probs)[-3:][::-1]
        top3_months = model_for_month.classes_[top3_idx]

        month_results = [MONTH_NUM_TO_NAME[m] for m in top3_months]

        # -------------------------
        # MODEL 2 → Next 3 Months Crop Suggestion
        # Correct Feature Order:
        # [month, state, city, temp, rain, suitability, soil, ph]
        # -------------------------
        for i in range(3):
            future_date = datetime.now() + timedelta(days=30 * i)
            month_num = future_date.month
            month_name = future_date.strftime("%B")

            temp2, rain2 = approximate_climate(month_num, state)
            suitability2 = approximate_suitability(crop_name, state)
            soil_enc, ph = approximate_soil_and_ph(crop_name)

            X2 = np.array([[month_num,
                            state_enc,
                            city_enc,
                            temp2,
                            rain2,
                            suitability2,
                            soil_enc,
                            ph]])

            pred_crop_enc = model_for_crop.predict(X2)[0]
            pred_crop_name = encoders['crop_label'].inverse_transform([pred_crop_enc])[0]

            guideline = CROP_GUIDELINES.get(pred_crop_name,
                                            "Suitable for current seasonal conditions.")

            upcoming_crops.append({
                "month": month_name,
                "crop": pred_crop_name,
                "guideline": guideline
            })

    return render_template(
        "month-suggestion.html",
        month_results=month_results,
        upcoming_crops=upcoming_crops,
        form_data=form_data,
        crop_guidelines=crop_guidelines
    )


# --- CROP SUGGESTION ROUTE ---
@app.route('/crop-suggestion', methods=['GET', 'POST'])
def crop_suggestion():
    # Login Check
    if "user_id" not in session:
        return redirect("/login")

    crop_results = []
    form_data = {}

    # Month name → month number
    MONTH_MAP = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }

    if request.method == 'POST':
        try:
            # ----- Get Inputs -----
            month_input = request.form.get('month', '').strip().capitalize()
            state = request.form.get('state', '').strip()
            city = request.form.get('city', '').strip()

            form_data = {'month': month_input, 'state': state, 'city': city}

            # Validate month
            month_num = MONTH_MAP.get(month_input)
            if not month_num:
                flash("Invalid Month. Example: January", "error")
                return render_template("crop-suggestion.html",
                                       crop_results=[], form_data=form_data)

            # Validate state/city exist in encoders
            try:
                state_enc = safe_encode_label(state, encoders['state'], "State")
                city_enc = safe_encode_label(city, encoders['city'], "City")
            except ValueError as e:
                flash(str(e), "error")
                return render_template("crop-suggestion.html",
                                       crop_results=[], form_data=form_data)

            # ----- Generate realistic climate -----
            avg_temp, avg_rain = approximate_climate(month_num, state)

            # ----- Soil & pH -----
            soil_enc, ph = approximate_soil_and_ph("dummy")  # dummy because crop unknown

            # ----- Suitability score for this region -----
            suitability = 60  # neutral base – because crop is unknown

            # ----- Prepare feature vector -----
            X = np.array([[month_num, state_enc, city_enc,
                           avg_temp, avg_rain, suitability,
                           soil_enc, ph]])

            # ----- Predict all crops -----
            crop_probs = model_for_crop.predict_proba(X)[0]

            top3_idx = np.argsort(crop_probs)[-3:][::-1]
            top3_crop_names = encoders['crop_label'].inverse_transform(top3_idx)

            # ----- Build full result using CSV -----
            for i, crop_name in enumerate(top3_crop_names):

                # Try finding data for the selected state
                row = synthetic_data_df[
                    (synthetic_data_df['crop'] == crop_name) &
                    (synthetic_data_df['state'] == state)
                ]

                if not row.empty:
                    row = row.iloc[0]
                else:
                    # Fallback: any entry for that crop
                    row = synthetic_data_df[synthetic_data_df['crop'] == crop_name].iloc[0]

                avg_temp_csv = float(row['avg_temp_C'])
                avg_rain_csv = float(row['avg_rainfall_mm'])

                crop_data = {
                    'name': crop_name,
                    'suitability': int(crop_probs[top3_idx[i]] * 100),
                    'temp': f"{int(avg_temp_csv - 5)}–{int(avg_temp_csv + 5)}°C",
                    'rainfall': f"{int(avg_rain_csv * 0.8)}–{int(avg_rain_csv * 1.2)} mm",
                    'soil': f"{row['soil_type']} / pH {row['soil_ph']}",
                    'duration': f"{int(row['duration_days'])} days",
                    'sowing': row['month_name']
                }

                crop_results.append(crop_data)

        except Exception as e:
            flash(f"Unexpected Error: {e}", "error")
            print("Crop Suggestion Error:", e)

    return render_template("crop-suggestion.html",
                           crop_results=crop_results,
                           form_data=form_data)



# --- VISUAL CROSSING CONFIG ---
VC_API_KEY = "KHHGZRGXSV32ZGG9MACKVTAL8"  # put your key here
VC_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"


@app.route("/old-and-latest-weather", methods=["GET", "POST"])
def old_and_latest_weather():

    if "user_id" not in session:
        return redirect(url_for("login"))

    # Default location fallback
    city_name = "Mangalore"
    state_name = "Karnataka"
    lat = None
    lon = None

    # Read form data (from geolocation or manual form)
    if request.method == "POST":

        geo_lat = request.form.get("latitude")
        geo_lon = request.form.get("longitude")

        manual_city = request.form.get("city", "").strip()
        manual_state = request.form.get("state", "").strip()

        # PRIORITY 1: Use geolocation coordinates
        if geo_lat and geo_lon:
            try:
                lat = float(geo_lat)
                lon = float(geo_lon)

                # openweathermap reverse geocode using timeline
                # Use OpenWeatherMap free reverse geocode for city/state
                OWM_KEY = "65ec2a1c272bd841e00a8bdd49f1c9d3"  # same key you use in dashboard

                reverse_url = (
                    f"http://api.openweathermap.org/geo/1.0/reverse"
                    f"?lat={lat}&lon={lon}&limit=1&appid={OWM_KEY}"
                )

                geo_data = requests.get(reverse_url).json()

                if geo_data and len(geo_data) > 0:
                    city_name = geo_data[0].get("name", "Unknown")
                    state_name = geo_data[0].get("state", geo_data[0].get("country", "Unknown"))
                else:
                    city_name = manual_city or city_name
                    state_name = manual_state or state_name


            except:
                # If reverse geocode fails → fallback to manual
                if manual_city and manual_state:
                    city_name = manual_city
                    state_name = manual_state

        # PRIORITY 2: Manual input
        elif manual_city and manual_state:
            city_name = manual_city
            state_name = manual_state

    # ------------------------------
    # SET UP VISUAL CROSSING REQUEST
    # ------------------------------

    # If LAT/LON available → use coordinate URL
    if lat and lon:
        VC_URL = (
            f"{VC_BASE_URL}/{lat},{lon}/last7days/next7days"
            f"?unitGroup=metric&include=days&key={VC_API_KEY}&contentType=json"
        )
    else:
        # Use city + state location string
        location_str = f"{city_name},{state_name}"
        VC_URL = (
            f"{VC_BASE_URL}/{location_str}/last7days/next7days"
            f"?unitGroup=metric&include=days&key={VC_API_KEY}&contentType=json"
        )

    # ------------------------------
    # FETCH DATA
    # ------------------------------
    try:
        resp = requests.get(VC_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        flash(f"Could not load weather data: {e}", "error")
        return render_template(
            "old-and-latest-weather.html",
            past_days=[],
            next_days=[],
            city_name=city_name,
            state_name=state_name,
            lat=lat,
            lon=lon
        )

    # Parse days
    all_days = data.get("days", [])
    past_days = []
    next_days = []

    today_str = datetime.now().strftime("%Y-%m-%d")

    for d in all_days:
        day_obj = {
            "date": d.get("datetime"),
            "temp": round(d.get("temp", 0), 1),
            "feels_like": round(d.get("feelslike", 0), 1),
            "max_temp": round(d.get("tempmax", 0), 1),
            "min_temp": round(d.get("tempmin", 0), 1),
            "humidity": int(d.get("humidity", 0)),
            "wind_speed": round(d.get("windspeed", 0), 1),
            "clouds": int(d.get("cloudcover", 0)),
            "rainfall": round(d.get("precip", 0), 1),
            "sunrise": d.get("sunrise"),
            "sunset": d.get("sunset"),
            "description": d.get("description") or d.get("conditions"),
        }

        if d.get("datetime") < today_str:
            past_days.append(day_obj)
        elif d.get("datetime") > today_str:
            next_days.append(day_obj)

    # FINAL TRIM + ORDER
    past_days = sorted(past_days[-7:], key=lambda x: x["date"], reverse=True)
    next_days = sorted(next_days[:7], key=lambda x: x["date"])

    return render_template(
        "old-and-latest-weather.html",
        past_days=past_days,
        next_days=next_days,
        city_name=city_name,
        state_name=state_name,
        lat=lat if lat else "",
        lon=lon if lon else ""
    )


# LOGOUT
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# RUN APP
if __name__ == "__main__":
    app.run(debug=True)
