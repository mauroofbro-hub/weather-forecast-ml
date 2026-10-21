# data_generator.py
# Run: python data_generator.py
# Requirements: pandas, numpy
# Install with: pip install pandas numpy

import random
import numpy as np
import pandas as pd
from collections import defaultdict

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

NUM_RECORDS = 10000  # More data = better accuracy

# ---------------------------
# Basic lists (India-focused)
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

MONTHS = list(range(1, 13))  # 1 = January ... 12 = December
MONTH_NAME = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# ---------------------------
# Crop preferred month windows (synthetic knowledge)
# ---------------------------
CROP_PREFERRED_MONTHS = {
    "Rice": [6,7,8,9],        # monsoon
    "Wheat": [11,12,1,2],     # rabi
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
    "Banana": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Coconut": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Rubber": [4,5,6,7,8,9],
    "Barley": [11,12,1,2]
}

# Some crops are more successful in particular states (bias)
CROP_STATE_BIAS = defaultdict(list)
CROP_STATE_BIAS.update({
    "Rice": ["Bihar","West Bengal","Assam","Odisha","Andhra Pradesh","Tamil Nadu"],
    "Wheat": ["Punjab","Haryana","Uttar Pradesh","Rajasthan","Madhya Pradesh"],
    "Tea": ["Assam","West Bengal","Tamil Nadu","Kerala"],
    "Coffee": ["Karnataka","Kerala","Tamil Nadu"],
    "Coconut": ["Kerala","Tamil Nadu","Karnataka","Andhra Pradesh","Goa"],
    "Rubber": ["Kerala","Tamil Nadu","Karnataka"],
    "Cotton": ["Maharashtra","Gujarat","Telangana","Andhra Pradesh","Haryana"],
    "Mustard": ["Rajasthan","Haryana","Punjab","Uttar Pradesh"],
    "Soybean": ["Madhya Pradesh","Maharashtra","Rajasthan"],
    "Sugarcane": ["Uttar Pradesh","Maharashtra","Karnataka","Tamil Nadu"],
    "Maize": ["Karnataka","Rajasthan","Maharashtra","Bihar"],
    "Groundnut": ["Gujarat","Tamil Nadu","Andhra Pradesh"],
    "Potato": ["Uttar Pradesh","West Bengal","Bihar"],
    "Onion": ["Maharashtra","Karnataka","Gujarat"],
})


# ---------------------------
# Helper functions
# ---------------------------
def random_state_city():
    """Select a random state and city pair."""
    state = random.choice(list(STATES_AND_CITIES.keys()))
    city = random.choice(STATES_AND_CITIES[state])
    return state, city

def generate_best_month_for_crop(crop, state):
    """Generate best planting month with STRONG deterministic patterns."""
    pref = CROP_PREFERRED_MONTHS.get(crop, MONTHS)
    
    # If state has strong bias for this crop, ALWAYS pick from preferred months
    if state in CROP_STATE_BIAS.get(crop, []):
        # 95% chance to pick preferred month
        if np.random.random() < 0.95:
            return int(np.random.choice(pref))
        else:
            return int(np.random.choice(MONTHS))
    else:
        # 85% chance to pick preferred month even without state bias
        if np.random.random() < 0.85:
            return int(np.random.choice(pref))
        else:
            return int(np.random.choice(MONTHS))



def generate_temp_rain_for_month_state(month, state):
    """Generate synthetic temperature and rainfall based on month."""
    base_temp = {
        1: 22, 2: 25, 3: 28, 4: 32, 5: 33, 6: 30,
        7: 28, 8: 28, 9: 29, 10: 30, 11: 27, 12: 24
    }
    base_rain = {
        1: 10, 2: 15, 3: 30, 4: 50, 5: 60, 6: 150,
        7: 200, 8: 180, 9: 120, 10: 40, 11: 20, 12: 10
    }
    t = base_temp[month] + np.random.normal(0, 2)
    r = max(0, base_rain[month] + np.random.normal(0, 30))
    return round(t,1), round(r,1)

# ---------------------------
# Build synthetic dataset
# ---------------------------
def generate_synthetic_data():
    """Generate synthetic agricultural dataset."""
    print("Generating synthetic agricultural data...")
    records = []
    
    for i in range(NUM_RECORDS):
        crop = random.choice(CROPS)
        state, city = random_state_city()
        
        # Generate best month based on crop & state
        best_month = generate_best_month_for_crop(crop, state)
        
        # Generate climatic features
        temp, rainfall = generate_temp_rain_for_month_state(best_month, state)
        
        # Generate suitability score (0-100)
        pref = CROP_PREFERRED_MONTHS.get(crop, [])
        base_score = 50
        if best_month in pref:
            base_score += random.randint(10, 35)
        if state in CROP_STATE_BIAS.get(crop, []):
            base_score += random.randint(5, 15)
        
        suitability = int(min(100, max(10, base_score + np.random.normal(0,5))))
        
        # Additional agronomic features
        soil_type = random.choice(["Loamy", "Alluvial", "Black", "Sandy", "Laterite"])
        ph = round(random.uniform(5.0, 8.0), 1)
        duration_days = random.randint(60, 180)

        records.append({
            "crop": crop,
            "state": state,
            "city": city,
            "best_month": best_month,
            "month_name": MONTH_NAME[best_month],
            "avg_temp_C": temp,
            "avg_rainfall_mm": rainfall,
            "suitability_score": suitability,
            "soil_type": soil_type,
            "soil_ph": ph,
            "duration_days": duration_days
        })
    
    return pd.DataFrame(records)

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    # Generate dataset
    df = generate_synthetic_data()
    
    # Save to CSV
    csv_path = "synthetic_data.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Synthetic dataset saved -> {csv_path}")
    print(f"✓ Total records: {len(df)}")
    print(f"✓ Unique crops: {df['crop'].nunique()}")
    print(f"✓ Unique states: {df['state'].nunique()}")
    print(f"✓ Unique cities: {df['city'].nunique()}")
    print("\nDataset preview:")
    print(df.head())
    print("\nData generation complete!")
