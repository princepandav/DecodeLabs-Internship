import os
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

class TrapClassifierEngine:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.model = None

    def load_and_engineer(self, forward_look=5):
        """Loads data and engineers a 1:3 Risk/Reward target."""
        print("🔄 Ingesting dataset and engineering 1:3 RR Classification target...")
        self.df = pd.read_csv(self.csv_path)
        
        # 1. Standardize index & Booleans
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df.set_index('timestamp', inplace=True)
        self.df.sort_index(inplace=True)

        bool_cols = ['is_swing_high', 'is_swing_low', 'true_bos_bullish', 
                     'true_bos_bearish', 'sweep_trap_bsl', 'sweep_trap_ssl']
        for col in bool_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(int)

        object_cols = self.df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            self.df.drop(columns=object_cols, inplace=True)

        if 'delta_oi' not in self.df.columns and 'oi' in self.df.columns:
            self.df['delta_oi'] = self.df['oi'].diff()
            self.df['delta_oi'].fillna(0, inplace=True)

        # =================================================================
        # THE 1:3 RISK/REWARD ENGINE
        # =================================================================
        
        # 1. Calculate the exact Risk (Stop Loss is the high of the trap candle)
        self.df['trade_risk'] = self.df['high'] - self.df['close']
        # Floor the risk at 3 points to avoid division-by-zero math errors on flat candles
        self.df['trade_risk'] = self.df['trade_risk'].clip(lower=3.0) 
        
        # 2. Calculate Required Reward for 1:3 RR
        self.df['target_reward'] = self.df['trade_risk'] * 3.0
        
        # 3. Structure Check: Does the chart actually have room to fall?
        # Compare 3R target against the distance to Sell Side Liquidity (SSL)
        self.df['has_rr_room'] = self.df['dist_to_ssl'] >= self.df['target_reward']
        
        # 4. Did the market ACTUALLY hit the 1:3 target in the next 'N' candles?
        # We use rolling min() to check the lowest point the price reached before bouncing
        future_lowest_low = self.df['low'].rolling(window=forward_look).min().shift(-forward_look)
        max_price_drop = self.df['close'] - future_lowest_low
        
        # 1 = Massive Drop hit the 3x target, 0 = Failed / Stopped out
        self.df['target_is_massive_drop'] = np.where(max_price_drop >= self.df['target_reward'], 1, 0)
        
        # Clean up NaNs from shifting
        self.df['temp_future'] = future_lowest_low
        self.df.dropna(subset=['temp_future'], inplace=True)
        self.df.drop(columns=['temp_future'], inplace=True)
        self.df.fillna(0, inplace=True)

        # =================================================================
        # 5. THE ADVANCED INSTITUTIONAL FILTER
        # Only train the AI IF: It's a Trap AND there is structural room for 1:3 RR
        # =================================================================
        trap_mask = ((self.df['sweep_trap_bsl'] == 1) | (self.df['oi_zscore'] >= 1.5)) & (self.df['has_rr_room'] == True)
        
        self.df = self.df[trap_mask]
        print(f"🎯 Filtered dataset: {len(self.df)} Traps with valid 1:3 Risk/Reward room.")

    def train_model(self, confidence_threshold=0.70):
        """Trains the XGBoost Classifier to hunt traps with strict risk parameters."""
        features_to_drop = ['open', 'high', 'low', 'close', 'target_is_massive_drop', 'oi'] 
        
        X = self.df.drop(columns=[col for col in features_to_drop if col in self.df.columns])
        y = self.df['target_is_massive_drop']

        # Time-Series Split (80% train, 20% test)
        split_idx = int(len(self.df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Calculate dynamic weight for imbalanced data
        num_negatives = (y_train == 0).sum()
        num_positives = (y_train == 1).sum()
        dynamic_weight = num_negatives / num_positives if num_positives > 0 else 1
        
        print(f"⚖️ Applied Dynamic Class Weighting: {dynamic_weight:.2f}x")

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=dynamic_weight, # <--- AI is now forced to care about crashes
            eval_metric='auc'
        )

        self.model.fit(X_train, y_train)

        # --- FIX 2 & 3: STRICT EXECUTION LOGIC ---
        # Get raw probabilities instead of default 50% predictions
        self.pred_probs = self.model.predict_proba(X_test)[:, 1] 
        
        # Extract Close and VWAP for the test set to apply hard filters
        close_test = self.df['close'].iloc[split_idx:].values
        
        if 'vwap' in self.df.columns:
            vwap_test = self.df['vwap'].iloc[split_idx:].values
            is_below_vwap = close_test < vwap_test
        else:
            print("⚠️ 'vwap' column missing. Skipping VWAP trend filter.")
            is_below_vwap = True # Default pass if missing

        # Execute ONLY if AI confidence is >= 70% AND Price is below VWAP
        is_confident = self.pred_probs >= confidence_threshold
        
        # Combine filters to generate final predictions
        self.predictions = (is_confident & is_below_vwap).astype(int)
        
        # Calculate Advanced Evaluation Metrics
        accuracy = accuracy_score(y_test, self.predictions)
        precision = precision_score(y_test, self.predictions, zero_division=0)
        recall = recall_score(y_test, self.predictions, zero_division=0)
        f1 = f1_score(y_test, self.predictions, zero_division=0)
        auc = roc_auc_score(y_test, self.pred_probs)
        cm = confusion_matrix(y_test, self.predictions)
        
        print("\n" + "="*60)
        print(f"✅ ADVANCED MODEL METRICS (OUT-OF-SAMPLE DATA)")
        print("="*60)
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%  (Targeting >50% for high RR)")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1-Score:  {f1 * 100:.2f}%")
        print(f"ROC-AUC:   {auc:.3f}")
        print("-" * 60)
        print("📊 CONFUSION MATRIX (Trade Breakdown):")
        
        # Handle cases where the matrix might be smaller if no trades were taken
        tn = cm[0][0] if len(cm) > 0 else 0
        fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
        fn = cm[1][0] if len(cm) > 1 else 0
        tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0

        print(f" [ TN: {tn:<5} ]  Safely Ignored")
        print(f" [ FP: {fp:<5} ]  FAILED TRADES (Hit Stop Loss)")
        print(f" [ FN: {fn:<5} ]  Missed Opportunities")
        print(f" [ TP: {tp:<5} ]  PERFECT TRADES (Massive Wins)")
        print("="*60)

        # =================================================================
        # NEW LOGIC: AUTOMATED OPTIMIZED MODEL SERIALIZATION
        # =================================================================
        # Ensure our models directory structure exists safely
        os.makedirs("week_1/models", exist_ok=True)
        latest_model_path = os.path.join("week_1/models", "latest_trap_engine.json")
        best_model_path = os.path.join("week_1/models", "optimized_best_trap_engine.json")
        
        # Save the current state as the latest executable run
        self.model.save_model(latest_model_path)
        
        # Establish a strict validation check before stamping it as our deployment binary
        # It must perform profitably out-of-sample against our strict risk engine parameters
        if precision >= 0.50 and tp > 0:
            self.model.save_model(best_model_path)
            print(f"💾 PRODUCTION EXPORT SUCCESS: Model cleared out-of-sample edge conditions.")
            print(f"   -> Precision: {precision * 100:.2f}% | Perfect Trades: {tp}")
            print(f"   -> Optimized Production Binary Stored at: {best_model_path}")
        else:
            print("⚠️ PRODUCTION UPGRADE SKIPPED: Current training iteration did not beat out-of-sample constraints.")
            print("   -> Latest pipeline checkpoint saved for debugging purposes.")
        print("="*60)

    def plot_feature_importance(self):
        """Visualizes exactly what the AI prioritizes."""
        importances = self.model.feature_importances_
        features = self.model.feature_names_in_
        
        indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in indices][:10]
        sorted_importances = importances[indices][:10]

        print("\n🏆 NEW TOP FEATURES (FOCUSING ON DELTA OI & TRAPS):")
        for f, imp in zip(sorted_features, sorted_importances):
            print(f"   {f:>15} : {imp:.4f}")

        fig = go.Figure(go.Bar(
            x=sorted_importances[::-1],
            y=sorted_features[::-1],
            orientation='h',
            marker_color='#ef5350'
        ))
        
        fig.update_layout(
            title="AI Feature Importance: Institutional Trap Detection",
            template="plotly_dark",
            xaxis_title="Relative Impact on Prediction",
            yaxis_title="SMC & OI Features",
            height=600
        )
        fig.show()

    def plot_ai_signals(self):
        """Visualizes the AI's predictions directly on the Candlestick chart."""
        print("📈 Generating AI Signal Visualizer...")
        
        split_idx = int(len(self.df) * 0.8)
        test_df = self.df.iloc[split_idx:].copy()
        
        test_df['ai_signal'] = self.predictions
        y_test = self.df['target_is_massive_drop'].iloc[split_idx:]
        test_df['actual_drop'] = y_test.values

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=test_df.index,
            open=test_df['open'], high=test_df['high'],
            low=test_df['low'], close=test_df['close'],
            name="Price Action",
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ))

        ai_sells = test_df[test_df['ai_signal'] == 1]
        if not ai_sells.empty:
            fig.add_trace(go.Scatter(
                x=ai_sells.index, y=ai_sells['high'] + (test_df['high'].max() * 0.0005),
                mode='markers', 
                marker=dict(symbol='triangle-down', size=16, color='#ff1744', line=dict(width=2, color='white')),
                name="AI Signal: SELL (High Confidence)"
            ))

        actual_crashes = test_df[test_df['actual_drop'] == 1]
        if not actual_crashes.empty:
            fig.add_trace(go.Scatter(
                x=actual_crashes.index, y=actual_crashes['low'] - (test_df['low'].max() * 0.0005),
                mode='markers', 
                marker=dict(symbol='star', size=12, color='#ffeb3b'),
                name="Confirmation: Massive Drop Occurred"
            ))

        fig.update_layout(
            title="Machine Learning Execution: Strict 55% Confidence + VWAP Filter",
            template="plotly_dark", hovermode="x unified",
            xaxis_rangeslider_visible=False, height=800,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Nifty 50 Price")
        fig.show()

if __name__ == "__main__":
    CSV_FILE_PATH = "week_1/data/train/NIFTY/PROC_NIFTY_THREE_MINUTE.csv"
    
    try:
        classifier = TrapClassifierEngine(CSV_FILE_PATH)
        classifier.load_and_engineer(forward_look=5) 
        classifier.train_model(confidence_threshold=0.55)
        classifier.plot_feature_importance()
        classifier.plot_ai_signals()
            
    except Exception as e:
        print(f"💥 Runtime Exception: {e}")
