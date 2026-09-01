# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

print("[*] STARTING: Data Training Process...")

# 1. Load Data
try:
    df = pd.read_csv('india_house_data.csv')
    print(f"📊 Dataset Loaded: {len(df)} rows.")
except FileNotFoundError:
    print("[ERROR] 'india_house_data.csv' not found. Please upload the file.")
    exit()

# 2. Rename Columns for consistency
df.columns = df.columns.str.strip() 
df = df.rename(columns={
    'City': 'location',
    'Size_in_SqFt': 'sqft',
    'BHK': 'bhk',
    'Crime_Rate': 'crime_rate',
    'Pollution Index': 'pollution_index',
    'Price_in_Thousand': 'price'
})

# Keep only necessary columns
df = df[['location', 'sqft', 'bhk', 'crime_rate', 'pollution_index', 'price']]

# Convert to numeric and drop bad data
for col in ['sqft', 'bhk', 'crime_rate', 'pollution_index', 'price']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

original_count = len(df)

# ---------------------------------------------------------
# 3. REMOVE OUTLIERS (The Fix for High Prices)
# ---------------------------------------------------------
print("[*] Removing Outliers...")

# Filter 1: Area (Keep houses between 200 and 10,000 sqft)
df = df[(df['sqft'] >= 200) & (df['sqft'] <= 10000)]

# Filter 2: BHK (Keep houses with 10 or fewer rooms)
df = df[df['bhk'] <= 10]

# Filter 3: PRICE (Crucial!)
# We remove the top 10% most expensive properties because they ruin the average calculation.
price_cap = df['price'].quantile(0.90) 
df = df[df['price'] < price_cap]

print(f"[*] Removed {original_count - len(df)} outlier rows. Training on {len(df)} clean rows.")

# ---------------------------------------------------------
# 4. TRAIN MODEL (Random Forest)
# ---------------------------------------------------------
X = df[['location', 'sqft', 'bhk', 'crime_rate', 'pollution_index']]
y = df['price']

# Create Transformer
# handle_unknown='ignore' prevents crashes if a city name is slightly different
column_trans = make_column_transformer(
    (OneHotEncoder(sparse_output=True, handle_unknown='ignore'), ['location']),
    remainder='passthrough'
)

# Random Forest captures complex location logic better than Linear Regression
# n_estimators=50 and max_depth=15 keep the file size small enough for Streamlit Cloud
model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)

pipe = make_pipeline(column_trans, model)

print("[*] Training Model (This takes about 1-2 minutes)...")
pipe.fit(X, y)

# 5. Save Model
print("[*] Saving 'HousePriceModel.pkl'...")
pickle.dump(pipe, open('HousePriceModel.pkl', 'wb'))

print("[OK] SUCCESS! Model created successfully.")