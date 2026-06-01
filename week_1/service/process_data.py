import pandas as pd
import numpy as np
import glob
import os
import pandas_ta as ta  # pip install pandas_ta

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
RAW_DATA_FOLDER = "raw_data"
TRAIN_DATA_FOLDER = "train_data"

# ==========================================
# 1. STRUCTURE: SWINGS, BSL, SSL & DISTANCE
# ==========================================
def process_structure(df):
    """Identifies Swing Highs/Lows and projects Liquidity Pools."""
    # 5-Candle Fractal Logic
    df['is_swing_high'] = (df['high'].shift(2) > df['high'].shift(4)) & \
                          (df['high'].shift(2) > df['high'].shift(3)) & \
                          (df['high'].shift(2) > df['high'].shift(1)) & \
                          (df['high'].shift(2) > df['high'])

    df['is_swing_low'] = (df['low'].shift(2) < df['low'].shift(4)) & \
                         (df['low'].shift(2) < df['low'].shift(3)) & \
                         (df['low'].shift(2) < df['low'].shift(1)) & \
                         (df['low'].shift(2) < df['low'])

    # Carry forward the last established BSL / SSL
    df['active_bsl'] = np.where(df['is_swing_high'], df['high'].shift(2), np.nan)
    df['active_ssl'] = np.where(df['is_swing_low'], df['low'].shift(2), np.nan)
    
    df['active_bsl'] = df['active_bsl'].ffill()
    df['active_ssl'] = df['active_ssl'].ffill()

    # Calculate exact distance to liquidity pools
    df['dist_to_bsl'] = df['active_bsl'] - df['close']
    df['dist_to_ssl'] = df['close'] - df['active_ssl']

    return df

# ==========================================
# 2. BOS & TRAP DETECTION (Logical Filters)
# ==========================================
def process_bos_and_traps(df):
    """Differentiates between a True Breakout (BOS) and a Liquidity Sweep (Trap)."""
    
    # Calculate Candle Anatomy for logical checks
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - np.maximum(df['close'], df['open'])
    df['lower_wick'] = np.minimum(df['close'], df['open']) - df['low']
    
    # --- TRUE BOS LOGIC ---
    # To be a true breakout, price must CLOSE above BSL, and the body must be strong (not a massive wick rejection)
    df['true_bos_bullish'] = np.where(
        (df['close'] > df['active_bsl']) & (df['body_size'] > df['upper_wick']), 1, 0
    )
    df['true_bos_bearish'] = np.where(
        (df['close'] < df['active_ssl']) & (df['body_size'] > df['lower_wick']), 1, 0
    )

    # --- SWEEP / TRAP LOGIC ---
    # A trap is when price pierces the liquidity pool, but closes back inside with a massive rejection wick
    df['sweep_trap_bsl'] = np.where(
        (df['high'] > df['active_bsl']) & (df['close'] < df['active_bsl']) & (df['upper_wick'] > (2 * df['body_size'])), 1, 0
    )
    df['sweep_trap_ssl'] = np.where(
        (df['low'] < df['active_ssl']) & (df['close'] > df['active_ssl']) & (df['lower_wick'] > (2 * df['body_size'])), 1, 0
    )

    return df

# ==========================================
# 3. OPEN INTEREST (The Fuel)
# ==========================================
def process_oi_features(df):
    """Calculates OI Momentum, Z-Scores, and Institutional Ratios."""
    # Handle spot charts where OI might be missing or 0
    if 'oi' not in df.columns:
        df['oi'] = 0
        
    df['oi'] = df['oi'].replace(0, np.nan) # Prevent division by zero
    
    # 1. OI Percentage Change
    df['oi_pct_change'] = df['oi'].pct_change() * 100
    
    # 2. Volume to OI Ratio (Detects churning vs actual positioning)
    df['vol_oi_ratio'] = df['volume'] / df['oi']
    
    # 3. OI Z-Score (Statistical extreme detection over 20 periods)
    oi_mean = df['oi'].rolling(window=20).mean()
    oi_std = df['oi'].rolling(window=20).std()
    df['oi_zscore'] = (df['oi'] - oi_mean) / oi_std
    
    # Clean up NaNs created by rolling windows or spot data
    df.fillna({'oi': 0, 'oi_pct_change': 0, 'vol_oi_ratio': 0, 'oi_zscore': 0}, inplace=True)
    
    return df

# ==========================================
# 4. VOLUME & VWAP (The Institutional Anchor)
# ==========================================
def process_volume_vwap(df):
    """Calculates Intraday VWAP and Price-to-VWAP interaction."""
    df['date'] = df['timestamp'].dt.date
    
    df['typ_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['typ_price'] * df['volume']
    
    # Daily Cumulative calculations for VWAP
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    
    df['vwap'] = df['cum_pv'] / df['cum_vol']
    
    # Distance from VWAP (Momentum extension indicator)
    df['dist_to_vwap_pct'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
    
    # Trend alignment logic (1 = Bullish Anchor, -1 = Bearish Anchor)
    df['vwap_trend'] = np.where(df['close'] > df['vwap'], 1, -1)
    
    df.drop(columns=['typ_price', 'pv', 'cum_vol', 'cum_pv'], inplace=True)
    return df

# ==========================================
# 5 & 6. ENTRY TRIGGERS & EXIT TARGETS (ATR)
# ==========================================
def process_entries_exits(df):
    """Defines exact logical entry triggers and calculates dynamic Stop Loss / Take Profit targets."""
    
    # --- ATR for Dynamic Exits ---
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # --- ENTRY LOGIC MATRIX ---
    # BUY ENTRY: Price swept SSL (Trap), New OI was added (Fuel), and Price is above VWAP (Trend)
    df['trigger_long'] = np.where(
        (df['sweep_trap_ssl'] == 1) & (df['oi_pct_change'] > 0.5) & (df['vwap_trend'] == 1), 1, 0
    )
    
    # SELL ENTRY: Price swept BSL (Trap), New OI was added (Short Fuel), and Price is below VWAP
    df['trigger_short'] = np.where(
        (df['sweep_trap_bsl'] == 1) & (df['oi_pct_change'] > 0.5) & (df['vwap_trend'] == -1), 1, 0
    )

    # --- EXIT LOGIC (Risk Management) ---
    # Placed exactly at the moment of entry. 
    # SL = 1.5x ATR (Give it room to breathe). TP = 3.0x ATR (1:2 Risk/Reward).
    df['long_sl'] = np.where(df['trigger_long'] == 1, df['close'] - (1.5 * df['atr']), np.nan)
    df['long_tp'] = np.where(df['trigger_long'] == 1, df['close'] + (3.0 * df['atr']), np.nan)
    
    df['short_sl'] = np.where(df['trigger_short'] == 1, df['close'] + (1.5 * df['atr']), np.nan)
    df['short_tp'] = np.where(df['trigger_short'] == 1, df['close'] - (3.0 * df['atr']), np.nan)

    df.dropna(subset=['atr'], inplace=True)
    return df

# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def run_processing(target_symbol=None):
    if target_symbol:
        print(f"🎯 Processing ONLY for symbol: {target_symbol}")
        search_path = f"{RAW_DATA_FOLDER}/{target_symbol}"
    else:
        print("🌍 Processing ALL available data...")
        search_path = f"{RAW_DATA_FOLDER}/*"

    stock_folders = glob.glob(search_path)
    
    if not stock_folders:
        print(f"❌ No data found in {search_path}")
        return

    for folder in stock_folders:
        symbol = os.path.basename(folder)
        save_folder = f"{TRAIN_DATA_FOLDER}/{symbol}"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        csv_files = glob.glob(f"{folder}/*.csv")
        
        for file in csv_files:
            try:
                print(f"  ⚙️ Processing {os.path.basename(file)}...")
                df = pd.read_csv(file)
                
                # Basic Cleaning
                df.columns = [c.lower() for c in df.columns]
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Fast Vectorized Pipeline Execution (STRICT LOGIC)
                df = process_structure(df)        # 1. Swings & Liquidity
                df = process_bos_and_traps(df)    # 2. BOS vs Traps
                df = process_oi_features(df)      # 3. Open Interest Fuel
                df = process_volume_vwap(df)      # 4. Institutional VWAP
                df = process_entries_exits(df)    # 5 & 6. Triggers and ATR Targets
                
                # Save Output
                filename = os.path.basename(file)
                save_path = f"{save_folder}/PROC_{filename}"
                df.to_csv(save_path, index=False)
                
                last_time = df['timestamp'].max()
                print(f"      ✅ Saved. Data valid up to: {last_time} (Rows: {len(df)})")
                
            except Exception as e:
                print(f"      ❌ Error on {file}: {e}")

    print("\n✅ Algorithmic Data Processing Complete.")

if __name__ == "__main__":
    print("\n" + "="*50)
    user_input = input("Enter Symbol to Process (Leave empty for ALL): ").upper().strip()
    print("="*50 + "\n")
    
    if user_input:
        run_processing(user_input)
    else:
        run_processing()