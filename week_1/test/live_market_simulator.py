import os
import pandas as pd
import numpy as np
import xgboost as xgb

class LiveMarketSimulator:
    def __init__(self, model_path, csv_path):
        self.model_path = model_path
        self.csv_path = csv_path
        self.model = None
        self.df = None
        
        self._bootstrap_system()

    def _bootstrap_system(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Trained model binary not found at: {self.model_path}\nTrain the model first to generate the file.")
        
        print(" Loading optimized XGBoost model parameters...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"❌ Historical data source missing at: {self.csv_path}")
            
        print(" Preparing historical database for streaming simulation...")
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = self.df.columns.str.strip().str.lower()
        
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df.set_index('timestamp', inplace=True)
        self.df.sort_index(inplace=True)

        if 'delta_oi' not in self.df.columns and 'oi' in self.df.columns:
            self.df['delta_oi'] = self.df['oi'].diff().fillna(0)
        elif 'delta_oi' not in self.df.columns:
            self.df['delta_oi'] = 0.0

        self.df['trade_risk'] = (self.df['high'] - self.df['close']).clip(lower=3.0)
        self.df['target_reward'] = self.df['trade_risk'] * 3.0
        
        if 'dist_to_ssl' in self.df.columns:
            self.df['has_rr_room'] = (self.df['dist_to_ssl'] >= self.df['target_reward']).astype(int)
        else:
            self.df['has_rr_room'] = 1

    def start_live_simulation(self, start_str, end_str, confidence_threshold=0.55):
        try:
            start_dt = pd.to_datetime(start_str)
            end_dt = pd.to_datetime(end_str)
        except Exception:
            print("❌ Execution aborted. Invalid date string format. Use YYYY-MM-DD HH:MM.")
   
        sim_data = self.df.loc[(self.df.index >= start_dt) & (self.df.index <= end_dt)]
        
        if sim_data.empty:
            print(f" No market data windows discovered between {start_str} and {end_str}")
            return

        print(f"\n Launching Live Engine Simulation: {len(sim_data)} incoming candles queued.")
        print("=" * 85)
        print(f"{'TIMESTAMP':<20} | {'CLOSE':<9} | {'VWAP':<9} | {'AI CONF':<8} | {'ACTION / STATE'}")
        print("=" * 85)

        active_trades = []
        closed_trades = []

        features_to_drop = ['open', 'high', 'low', 'close', 'target_is_massive_drop', 'oi', 'of_state_name']

        for current_time, row in sim_data.iterrows():
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']

            feature_vector = row.drop([col for col in features_to_drop if col in row.index]).to_frame().T
            feature_vector = feature_vector[self.model.feature_names_in_]

            feature_vector = feature_vector.astype(float)
            raw_prob = self.model.predict_proba(feature_vector)[0, 1]

            vwap_val = row['vwap'] if 'vwap' in row.index else 0
            is_below_vwap = close_price < vwap_val if vwap_val > 0 else True
            
            has_trap_signal = False
            if 'sweep_trap_bsl' in row.index and 'oi_zscore' in row.index:
                if (row['sweep_trap_bsl'] == 1 or row['oi_zscore'] >= 1.5):
                    has_trap_signal = True

            action_string = " ...Scanning..."
            for trade in active_trades[:]:
                # Check if price hit the Stop Loss (High of entry candle)
                if high_price >= trade['stop_loss']:
                    trade['exit_time'] = current_time
                    trade['exit_price'] = trade['stop_loss']
                    trade['result'] = -1.0  # Lost 1R
                    closed_trades.append(trade)
                    active_trades.remove(trade)
                    action_string = f"💥 STOP LOSS HIT (-1R) at {trade['stop_loss']:.2f}"
                
                # Check if price reached the 1:3 Profit Target
                elif low_price <= trade['target_price']:
                    trade['exit_time'] = current_time
                    trade['exit_price'] = trade['target_price']
                    trade['result'] = 3.0  
                    closed_trades.append(trade)
                    active_trades.remove(trade)
                    action_string = f" TARGET HIT (+3R) at {trade['target_price']:.2f}"

            if has_trap_signal and (raw_prob >= confidence_threshold) and is_below_vwap:
                calculated_risk = max(high_price - close_price, 3.0)
                sl_target = high_price
                profit_target = close_price - (calculated_risk * 3.0)

                has_room = True
                if 'dist_to_ssl' in row.index and 'target_reward' in row.index:
                    has_room = row['dist_to_ssl'] >= (calculated_risk * 3.0)
                
                if has_room:
                    new_trade = {
                        'entry_time': current_time,
                        'entry_price': close_price,
                        'stop_loss': sl_target,
                        'target_price': profit_target
                    }
                    active_trades.append(new_trade)
                    action_string = f"🚨 EXECUTE SHORT | Entry: {close_price:.2f} | SL: {sl_target:.2f} | TG: {profit_target:.2f}"

            vwap_display = f"{vwap_val:<9.2f}" if vwap_val > 0 else "N/A      "
            print(f"{str(current_time):<20} | {close_price:<9.2f} | {vwap_display} | {raw_prob * 100:<7.1f}% | {action_string}")

        print("\n" + "="*60)
        print(" SIMULATION PERFORMANCE STATISTICS REPORT")
        print("="*60)
        total_signals = len(closed_trades) + len(active_trades)
        print(f"Total Trade Signals Executed : {total_signals}")
        print(f"Closed Positions              : {len(closed_trades)}")
        print(f"Positions Still Open          : {len(active_trades)}")
        
        if len(closed_trades) > 0:
            wins = sum(1 for t in closed_trades if t['result'] > 0)
            losses = sum(1 for t in closed_trades if t['result'] < 0)
            win_rate = (wins / len(closed_trades)) * 100
            net_r = sum(t['result'] for t in closed_trades)
            
            print(f"Perfect Trades (Wins)         : {wins}")
            print(f"Stopped Out Trades (Losses)   : {losses}")
            print(f"Calculated System Win Rate    : {win_rate:.2f}%")
            print(f"Gross Expected Return Output  : {net_r:+.2f}R")
            
            monetary_pnl = net_r * 3000
            print(f"Net Financial Performance     : {'+₹' if monetary_pnl >=0 else '-₹'}{abs(monetary_pnl):,}")
        print("="*60)

if __name__ == "__main__":
    MODEL_BINARY = "week_1/models/optimized_best_trap_engine.json"
    HISTORICAL_CSV = "week_1/data/train/NIFTY/PROC_NIFTY_FIVE_MINUTE.csv"
    try:
        simulator = LiveMarketSimulator(MODEL_BINARY, HISTORICAL_CSV)

        while True:
            print("\n" + "-"*50)
            start_window = input("🗓️ Set Start Range (e.g., 2026-05-14 09:15) or 'exit': ").strip()
            if start_window.lower() == 'exit': 
                break
            end_window = input("🗓️ Set End Range   (e.g., 2026-05-14 15:30): ").strip()
            simulator.start_live_simulation(start_window, end_window, confidence_threshold=0.55)
            
    except Exception as e:
        print(f" Simulation Engine Terminated Unexpectedly: {e}")