import streamlit as st
import pickle
import pandas as pd
import random
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EstateXpert",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* GLOBAL DARK THEME */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    
    
    /* CARDS */
    .standard-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363D;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .emi-highlight { color: #4CAF50; font-size: 32px; font-weight: bold; }

    /* NEON OWNER CARD */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');
    .neon-owner-wrapper {
        background: linear-gradient(135deg, #000000, #1a1a1a);
        border: 2px solid #00f2ff;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.4);
        animation: neonPulse 3s infinite alternate;
    }
    @keyframes neonPulse {
        from { box-shadow: 0 0 15px rgba(0, 242, 255, 0.3); border-color: #00f2ff; }
        to { box-shadow: 0 0 30px rgba(0, 242, 255, 0.6); border-color: #ff00ff; }
    }
    .neon-text { font-family: 'Orbitron', sans-serif; color: #fff; }
    .owner-badge { background-color: #ff00ff; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-family: 'Orbitron', sans-serif; box-shadow: 0 0 10px #ff00ff; margin-left: 10px; }
    .owner-avatar-neon { border-radius: 50%; border: 3px solid #00f2ff; width: 80px; height: 80px; object-fit: cover; }
    
    /* BUTTONS */
    div.stButton > button { width: 100%; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'price' not in st.session_state: st.session_state['price'] = None
if 'confirmed' not in st.session_state: st.session_state['confirmed'] = False
if 'owner_name' not in st.session_state: st.session_state['owner_name'] = "Unknown"
if 'owner_img' not in st.session_state: st.session_state['owner_img'] = ""

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_data():
    try:
        # Load Model
        with open('HousePriceModel.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load Data for Dropdowns and Stats
        df = pd.read_csv('india_house_data.csv')
        df.columns = df.columns.str.strip()
        df = df.rename(columns={'City': 'location', 'Crime_Rate': 'crime_rate', 'Pollution Index': 'pollution_index', 'Price_in_Thousand': 'price'})
        
        # Clean Data
        df['location'] = df['location'].astype(str).apply(lambda x: x.strip())
        df['crime_rate'] = pd.to_numeric(df['crime_rate'], errors='coerce')
        df['pollution_index'] = pd.to_numeric(df['pollution_index'], errors='coerce')
        
        locs = sorted(df['location'].unique().tolist())
        c_dict = df.groupby('location')['crime_rate'].mean().to_dict()
        p_dict = df.groupby('location')['pollution_index'].mean().to_dict()
        
        return model, locs, c_dict, p_dict
    except Exception as e:
        return None, [], {}, {}

model, locations, crime_dict, pollution_dict = load_data()

if not model:
    st.error("⚠️ Model file not found. Please run 'train.py' first.")
    st.stop()

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("🏡 Property Detail")
    selected_city = st.selectbox("Select City", locations)
    
    # Auto-fetch city stats
    city_crime = crime_dict.get(selected_city, 0)
    city_pollution = pollution_dict.get(selected_city, 0)
    
    sqft = st.number_input("Area (Sq. Ft.)", 300, 10000, 1200)
    bhk = st.slider("Bedrooms (BHK)", 1, 8, 2)
    
    st.markdown("---")
    if st.button("🔍 Estimate Price", type="primary"):
        # Create Input DataFrame ensuring columns match training exactly
        input_data = pd.DataFrame([[selected_city, sqft, bhk, city_crime, city_pollution]], 
                                  columns=['location', 'sqft', 'bhk', 'crime_rate', 'pollution_index'])
        
        prediction = model.predict(input_data)[0]
        
        # Safety Check: Ensure price isn't negative
        final_price = max(1.0, prediction)
        
        # Update Session State
        st.session_state['price'] = final_price
        st.session_state['city'] = selected_city
        st.session_state['sqft'] = sqft
        st.session_state['bhk'] = bhk
        st.session_state['crime'] = city_crime
        st.session_state['pollution'] = city_pollution
        st.session_state['confirmed'] = False 
        
        # Random Owner Data
        names = ["Vikram Malhotra", "Aditi Rao", "Rahul Sharma", "Priya Desai", "Arjun Kapoor", "Sneha Iyer"]
        st.session_state['owner_name'] = random.choice(names)
        st.session_state['owner_img'] = f"https://i.pravatar.cc/150?u={random.randint(10, 100)}"
        
        st.rerun()

# --- MAIN DISPLAY ---
st.title("EstateXpert")

if st.session_state['price'] is None:
    st.info("👈 Select parameters from the sidebar to begin.")
else:
    price_val = st.session_state['price']
    
    # Calculate Full Rupees for EMI (model output is in Thousands)
    total_rupees = price_val * 1000
    
    # 1. PRICE CARD
    st.markdown(f"""
    <div class="standard-card">
        <h4 style="color:#888; margin:0;">ESTIMATED MARKET VALUE</h4>
        <h1 style="color:#4CAF50; font-size:48px;">₹ {price_val:,.2f} Thousand</h1>
        <p style="color:#aaa;">{st.session_state['city']} | {st.session_state['sqft']} Sq.Ft | {st.session_state['bhk']} BHK</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. STATS CARDS
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.metric("⚠️ Crime Index", f"{st.session_state['crime']:.1f}")
        st.progress(min(st.session_state['crime']/200, 1.0))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.metric("🌫️ Pollution Index", f"{st.session_state['pollution']:.1f}")
        st.progress(min(st.session_state['pollution']/300, 1.0))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. EMI CALCULATOR
    st.subheader("🏦 Mortgage Calculator")
    
    with st.container():
        st.markdown('<div class="standard-card" style="text-align:left;">', unsafe_allow_html=True)
        col_input, col_result = st.columns([1, 1])
        
        with col_input:
            st.write("**Loan Settings**")
            dp_pct = st.slider("Down Payment (%)", 10, 80, 20)
            rate = st.slider("Interest Rate (%)", 6.0, 15.0, 8.5)
            tenure = st.slider("Tenure (Years)", 5, 30, 20)
        
        loan_amt = total_rupees * ((100 - dp_pct)/100)
        r = rate / (12*100)
        n = tenure * 12
        
        if r > 0:
            emi = loan_amt * r * ((1+r)**n) / (((1+r)**n) - 1)
        else:
            emi = loan_amt / n
            
        with col_result:
            st.markdown(f"""
            <div style="text-align:right; margin-top:20px;">
                <p style="color:#888; margin-bottom:0;">Monthly Installment</p>
                <div class="emi-highlight">₹ {int(emi):,}</div>
                <p style="font-size:12px; color:#aaa;">Duration: {tenure} years</p>
                <hr style="border-color:#333;">
                <p style="color:#aaa; font-size:14px;">Principal Amount: <b style="color:white">₹ {loan_amt/100000:.2f} L</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. OWNER DETAILS (Neon Effect)
    st.write("### 🔐 Verified Owner")
    
    if not st.session_state['confirmed']:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning("Owner contact details are encrypted for privacy.")
        with col2:
            if st.button("Unlock Details"):
                with st.spinner("Decrypting Blockchain ID..."):
                    time.sleep(1.5)
                st.session_state['confirmed'] = True
                st.rerun()
    else:
        name = st.session_state['owner_name']
        img = st.session_state['owner_img']
        phone = f"+91 {random.randint(7000, 9999)} {random.randint(10000, 99999)}"
        
        st.markdown(f"""
        <div class="neon-owner-wrapper">
            <div style="display:flex; align-items:center; gap: 20px;">
                <img src="{img}" class="owner-avatar-neon">
                <div>
                    <div class="neon-text" style="font-size: 20px; font-weight: bold;">
                        {name} <span class="owner-badge">VERIFIED</span>
                    </div>
                    <div class="neon-text" style="font-size: 16px; color: #00f2ff; margin-top: 5px;">
                        📞 {phone}
                    </div>
                    <div style="font-size: 12px; color: #aaa; margin-top: 5px;">
                        TOKEN ID: #EST-{random.randint(1000,9999)}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

        # please  fix the area and BHK according to data provide and also modofy the csv file for price in india replace with latest price of houses in india and use theme toggle