import pandas as pd
import numpy as np
import glob
import os
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from utils import (
    calculate_streak, 
    calculate_tie_stats, 
    load_all_games
)

# ==========================================
# New Advanced Feature Functions
# ==========================================

def calculate_tie_proximity(history):
    """ระยะห่างจากเสมอครั้งล่าสุด ยิ่งใกล้ยิ่งสวิงเยอะ"""
    if not history:
        return 20 # ค่า Default ไกลๆ
    
    distance = 0
    for value in reversed(history):
        if value == 2:
            return distance
        distance += 1
    return 20 # ถ้าไม่เจอ Tie เลย

def calculate_symmetry(history, window=6):
    """ตรวจจับว่าไพ่ทำทรงสมมาตรหรือไม่ (เช่น P-B-B-P)"""
    non_tie = [h for h in history if h != 2]
    if len(non_tie) < window:
        return 0.0
    
    recent = non_tie[-window:]
    
    # แบ่งครึ่งเพื่อเช็คการสะท้อน
    half = window // 2
    left = recent[:half]
    right = recent[half:]
    
    # P-B-B-P ลักษณะสมมาตร
    if left == list(reversed(right)):
         return 1.0 # 100% Symmetry
         
    return 0.0

def calculate_markov_features(history, window=20):
    """คำนวณ Transition Probabilities จาก N ตาล่าสุด"""
    non_tie = [h for h in history if h != 2]
    if len(non_tie) < 2:
        return {'p_to_p': 0.0, 'p_to_b': 0.0, 'b_to_b': 0.0, 'b_to_p': 0.0}
    
    recent = non_tie[-window:]
    transitions = {'P->P': 0, 'P->B': 0, 'B->B': 0, 'B->P': 0}
    p_count = 0
    b_count = 0
    
    for i in range(len(recent) - 1):
        prev, curr = recent[i], recent[i+1]
        if prev == 0:
            p_count += 1
            if curr == 0: transitions['P->P'] += 1
            else: transitions['P->B'] += 1
        elif prev == 1:
            b_count += 1
            if curr == 1: transitions['B->B'] += 1
            else: transitions['B->P'] += 1
            
    return {
        'p_to_p': transitions['P->P'] / p_count if p_count > 0 else 0.0,
        'p_to_b': transitions['P->B'] / p_count if p_count > 0 else 0.0,
        'b_to_b': transitions['B->B'] / b_count if b_count > 0 else 0.0,
        'b_to_p': transitions['B->P'] / b_count if b_count > 0 else 0.0
    }

def calculate_momentum(history, window=10):
    """คำนวณ momentum (การเปลี่ยนแปลงของ P/B ratio)"""
    non_tie = [h for h in history if h != 2]
    if len(non_tie) < window * 2:
        return 0.0
    
    recent = non_tie[-window:]
    previous = non_tie[-(window*2):-window]
    
    recent_b_ratio = recent.count(1) / len(recent)
    prev_b_ratio = previous.count(1) / len(previous)
    
    # Momentum: positive = banker trending, negative = player trending
    return recent_b_ratio - prev_b_ratio

def calculate_alternating_ratio(history, window=10):
    """คำนวณสัดส่วนการสลับ P-B-P-B ใน window ล่าสุด"""
    non_tie = [h for h in history if h != 2]
    if len(non_tie) < 2:
        return 0.5
    
    last_n = non_tie[-window:] if len(non_tie) >= window else non_tie
    if len(last_n) < 2:
        return 0.5
    
    switches = 0
    for i in range(1, len(last_n)):
        if last_n[i] != last_n[i-1]:
            switches += 1
    
    return switches / (len(last_n) - 1)

def get_position_phase(current_index, total_expected=70):
    """ระบุ phase ของ shoe (0=early, 1=mid, 2=late)"""
    ratio = current_index / total_expected
    if ratio < 0.3:
        return 0  # Early
    elif ratio < 0.7:
        return 1  # Mid
    else:
        return 2  # Late

def calculate_streak_break_frequency(history, min_streak=3):
    """คำนวณความถี่ที่ streak ยาวๆ ถูกตัด"""
    non_tie = [h for h in history if h != 2]
    if len(non_tie) < 5:
        return 0.5
    
    long_streaks = 0
    broken_streaks = 0
    current_streak = 1
    
    for i in range(1, len(non_tie)):
        if non_tie[i] == non_tie[i-1]:
            current_streak += 1
        else:
            if current_streak >= min_streak:
                long_streaks += 1
                broken_streaks += 1
            current_streak = 1
    
    # Check if current streak is long (not broken yet)
    if current_streak >= min_streak:
        long_streaks += 1
    
    return broken_streaks / long_streaks if long_streaks > 0 else 0.5

def count_consecutive_pairs(history, window=20):
    """นับจำนวน PP และ BB pairs ใน history"""
    non_tie = [h for h in history if h != 2]
    last_n = non_tie[-window:] if len(non_tie) >= window else non_tie
    
    pp_count = 0
    bb_count = 0
    
    for i in range(1, len(last_n)):
        if last_n[i] == 0 and last_n[i-1] == 0:
            pp_count += 1
        elif last_n[i] == 1 and last_n[i-1] == 1:
            bb_count += 1
    
    total_pairs = len(last_n) - 1 if len(last_n) > 1 else 1
    return pp_count / total_pairs, bb_count / total_pairs

def build_big_road_columns(history):
    """สร้างโครงสร้างคอลัมน์ของ Big Road"""
    columns = []
    current_col = []
    prev_winner = None
    
    for result in history:
        if result == 2:
            continue
        if prev_winner is None:
            current_col = [result]
            prev_winner = result
        elif result == prev_winner:
            current_col.append(result)
        else:
            columns.append(current_col)
            current_col = [result]
            prev_winner = result
    
    if current_col:
        columns.append(current_col)
    return columns

def get_derived_road_result(columns, road_type='big_eye'):
    """คำนวณเค้าไพ่รอง (Derived Roads)"""
    offset_map = {'big_eye': 1, 'small': 2, 'cockroach': 3}
    offset = offset_map.get(road_type, 1)
    derived_results = []
    
    if len(columns) < offset + 1:
        return derived_results
    
    for i in range(offset + 1, len(columns) + 1):
        current_cols = columns[:i]
        if len(current_cols) < offset + 1:
            continue
        current_col = current_cols[-1]
        compare_col = current_cols[-(offset + 1)]
        
        for row_idx in range(len(current_col)):
            if row_idx == 0:
                if len(compare_col) == len(current_cols[-2]) if len(current_cols) > 1 else True:
                    derived_results.append(1)
                else:
                    derived_results.append(0)
            else:
                if row_idx < len(compare_col):
                    derived_results.append(1)
                else:
                    derived_results.append(0)
    
    return derived_results

def get_derived_roads_features(history):
    """สร้าง Features จากเค้าไพ่รองทั้ง 3 แบบ"""
    columns = build_big_road_columns(history)
    
    big_eye = get_derived_road_result(columns, 'big_eye')
    small_road = get_derived_road_result(columns, 'small')
    cockroach = get_derived_road_result(columns, 'cockroach')
    
    def get_red_ratio(road, n=5):
        if not road:
            return 0.5
        last_n = road[-n:]
        return sum(last_n) / len(last_n) if last_n else 0.5
    
    big_eye_stability = get_red_ratio(big_eye)
    small_road_stability = get_red_ratio(small_road)
    cockroach_stability = get_red_ratio(cockroach)
    
    overall_stability = (big_eye_stability + small_road_stability + cockroach_stability) / 3
    
    return {
        'big_eye_stability': big_eye_stability,
        'small_road_stability': small_road_stability,
        'cockroach_stability': cockroach_stability,
        'overall_stability': overall_stability,
        'is_stable': 1 if overall_stability > 0.6 else 0,
        'is_volatile': 1 if overall_stability < 0.4 else 0,
        'big_eye_seq': big_eye,
        'small_road_seq': small_road,
        'cockroach_seq': cockroach
    }

def get_shoe_type(history):
    """Classify the current shoe type based on history"""
    if not history or len(history) < 5:
        return "Waiting...", "⏳"
        
    derived = get_derived_roads_features(history)
    stability = derived['overall_stability']
    
    # Check for Ping Pong (Choppy)
    last_6 = [h for h in history[-6:] if h != 2]
    is_pingpong = False
    if len(last_6) >= 4:
        switches = 0
        for i in range(1, len(last_6)):
            if last_6[i] != last_6[i-1]:
                switches += 1
        if switches >= len(last_6) - 1:
            is_pingpong = True
            
    if is_pingpong:
        return "Ping Pong (Choppy)", "🏓"
    elif stability > 0.6:
        return "Stable (Mangkorn)", "🐉"
    elif stability < 0.4:
         return "Volatile (Hew)", "🎢"
    else:
        return "Normal", "⚖️"

def analyze_road_quality(road_sequence):
    """วิเคราะห์คุณภาพของเค้าไพ่ (Pattern Analysis)"""
    if not road_sequence or len(road_sequence) < 4:
        return 0
    
    score = 0
    last_n = road_sequence[-6:]
    
    # 1. Dragon Pattern
    streak = 0
    if len(last_n) > 0:
        last_val = last_n[-1]
        for x in reversed(last_n):
            if x == last_val:
                streak += 1
            else:
                break
    
    if streak >= 4:
        score += streak * 2
    elif streak == 3:
        score += 2
        
    # 2. Ping Pong Pattern
    if len(last_n) >= 4:
        is_pp = True
        for i in range(len(last_n) - 1, 0, -1):
            if last_n[i] == last_n[i-1]:
                is_pp = False
                break
        if is_pp:
            score += len(last_n) * 2
            
    return score

def calculate_shoe_profile(history):
    """วิเคราะห์โปรไฟล์ของสำรับไพ่ (Shoe Profile Metrics)"""
    if not history or len(history) < 10:
        return None
        
    non_tie = [h for h in history if h != 2]
    if not non_tie:
        return None
        
    # 1. Banker Ratio
    b_ratio = non_tie.count(1) / len(non_tie)
    
    # 2. Stability (Derived Roads)
    derived = get_derived_roads_features(history)
    stability = derived['overall_stability']
    
    # 3. Alternating Ratio (Ping Pong frequency)
    switches = 0
    for i in range(1, len(non_tie)):
        if non_tie[i] != non_tie[i-1]:
            switches += 1
    alt_ratio = switches / (len(non_tie) - 1) if len(non_tie) > 1 else 0.5
    
    # 4. Tie Frequency
    tie_freq = history.count(2) / len(history)
    
    return {
        'b_ratio': b_ratio,
        'stability': stability,
        'alt_ratio': alt_ratio,
        'tie_freq': tie_freq
    }

def calculate_shoe_similarity(profile1, profile2):
    """คำนวณความเหมือนระหว่าง 2 โปรไฟล์ (0.0 - 1.0)"""
    if not profile1 or not profile2:
        return 0.0
    
    # ความต่างของ Banker Ratio, Stability, Alternating Ratio, Tie Frequency
    diffs = [
        abs(profile1['b_ratio'] - profile2['b_ratio']),
        abs(profile1['stability'] - profile2['stability']),
        abs(profile1['alt_ratio'] - profile2['alt_ratio']),
        abs(profile1['tie_freq'] - profile2['tie_freq'])
    ]
    
    # ยิ่งต่างน้อย ยิ่งเหมือนมาก
    avg_diff = sum(diffs) / len(diffs)
    return max(0, 1.0 - avg_diff)

def simulate_next_move(history, next_result):
    """จำลองว่าถ้าตาถัดไปออก next_result แล้ว road จะเป็นยังไง"""
    simulated = history + [next_result]
    return get_derived_roads_features(simulated)

def get_technician_vote(history):
    """Module B: The Technician"""
    if len([h for h in history if h != 2]) < 5:
        return None, 0
    
    player_sim = simulate_next_move(history, 0)
    banker_sim = simulate_next_move(history, 1)
    
    p_score = analyze_road_quality(player_sim['big_eye_seq'])
    b_score = analyze_road_quality(banker_sim['big_eye_seq'])
    
    p_score += analyze_road_quality(player_sim['small_road_seq']) * 0.5
    b_score += analyze_road_quality(banker_sim['small_road_seq']) * 0.5
    
    p_score += analyze_road_quality(player_sim['cockroach_seq']) * 0.5
    b_score += analyze_road_quality(banker_sim['cockroach_seq']) * 0.5
    
    confidence_display = max(p_score, b_score)
    
    if p_score > b_score + 2:
        return 0, int(confidence_display)
    elif b_score > p_score + 2:
        return 1, int(confidence_display)
    else:
        return None, 0

def get_expert_vote(history, patterns):
    """Module D: Expert Rules"""
    if not patterns or not history:
        return None, None
    
    non_tie = [h for h in history if h != 2]
    
    for p in patterns:
        if p['expected'] is None:
            continue
            
        pattern = p['pattern']
        pattern_len = len(pattern)
        
        if len(non_tie) >= pattern_len:
            last_n = non_tie[-pattern_len:]
            if last_n == pattern:
                return p['expected'], p['name']
        
        # Check against full history (including ties) for exact sequence match
        if len(history) >= pattern_len:
            last_n_full = history[-pattern_len:]
            if last_n_full == pattern:
                 return p['expected'], p['name']
    
    return None, None


@st.cache_data(ttl=300, show_spinner=False)
def process_data_from_folder(folder_path):
    """อ่านไฟล์ .txt ทั้งหมดใน data folder และสร้าง features ข้อมูลทั้งหมด"""
    all_files = glob.glob(os.path.join(folder_path, "*.txt"))
    data_rows = []
    pattern_sequences = []
    
    if not all_files:
        return pd.DataFrame(), []

    for filepath in all_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                raw = content.split(']')[1] if ']' in content else content
                game_seq = [int(x) for x in raw.replace(',', ' ').split() if x.strip().isdigit()]
                
                for i in range(5, len(game_seq)):
                    history = game_seq[:i]
                    target = game_seq[i]
                    
                    # Use full history (including ties) for pattern recognition if length permits
                    # Note: We take the last 5 outcomes including Ties
                    if len(history) < 5:
                        continue
                        
                    p1, p2, p3, p4, p5 = history[-5:]
                    
                    derived_features = get_derived_roads_features(history)
                    current_streak = calculate_streak(history)
                    tie_rate, gap_since_tie = calculate_tie_stats(history, 20)
                    last_10 = [h for h in history[-10:] if h != 2]
                    b_ratio = last_10.count(1) / len(last_10) if last_10 else 0.5
                    
                    last_20 = [h for h in history[-20:] if h != 2]
                    p_rate_20 = last_20.count(0) / len(last_20) if last_20 else 0.5
                    b_rate_20 = last_20.count(1) / len(last_20) if last_20 else 0.5
                    
                    b_rate_20 = last_20.count(1) / len(last_20) if last_20 else 0.5
                    
                    streak_owner = 0.5
                    non_tie_history = [h for h in history if h != 2]
                    if len(non_tie_history) >= 2:
                        if non_tie_history[-1] == 0: streak_owner = 0
                        elif non_tie_history[-1] == 1: streak_owner = 1
                    
                    markov = calculate_markov_features(history)
                    
                    row = {
                        'pattern_1': p1, 'pattern_2': p2, 'pattern_3': p3,
                        'pattern_4': p4, 'pattern_5': p5,
                        'current_streak': min(current_streak, 10),
                        'streak_owner': streak_owner,
                        'banker_trend': b_ratio,
                        'player_rate_20': p_rate_20,
                        'banker_rate_20': b_rate_20,
                        'tie_rate_20': tie_rate,
                        'gap_since_tie': min(gap_since_tie, 30),
                        'is_stable': derived_features['is_stable'],
                        'overall_stability': derived_features['overall_stability'],
                        # New Advanced Features
                        'position_phase': get_position_phase(i),
                        'momentum': calculate_momentum(history),
                        'alternating_ratio': calculate_alternating_ratio(history),
                        'streak_break_freq': calculate_streak_break_frequency(history),
                        'pp_ratio': count_consecutive_pairs(history)[0],
                        'bb_ratio': count_consecutive_pairs(history)[1],
                        'tie_proximity': calculate_tie_proximity(history),
                        'baccarat_symmetry': calculate_symmetry(history),
                        'p_to_p': markov['p_to_p'],
                        'p_to_b': markov['p_to_b'],
                        'b_to_b': markov['b_to_b'],
                        'b_to_p': markov['b_to_p'],
                        'target': target
                    }
                    data_rows.append(row)
                    
                    pattern_sequences.append({
                        'pattern': history[-5:], # Use history with Ties
                        'target': target # Keep target as is (even if Tie)
                    })
                    
        except Exception as e:
            print(f"Error processing data from {filepath}: {e}")
            
    return pd.DataFrame(data_rows), pattern_sequences

def train_ensemble_models(df, pattern_sequences):
    """Train RF, KNN, and XGBoost models"""
    models = {}
    
    if not df.empty:
        X = df.drop(columns=['target'])
        y = df['target']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        models['scaler'] = scaler
        
        # Module C: Random Forest (Statistician)
        rf_model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42, class_weight='balanced')
        rf_model.fit(X_scaled, y)
        models['rf'] = rf_model
        models['rf_features'] = list(X.columns)
        
        # ส่วนที่ 4: สูตรคำนวณ (XGBoost) - Tuned to prevent overfitting on small datasets
        try:
            xgb_model = XGBClassifier(
                n_estimators=100, 
                max_depth=3,  # ลดความลึกเพื่อป้องกันการ Overfit บนข้อมูลน้อย
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.2, # เพิ่ม gamma
                reg_alpha=0.5, # เพิ่ม regularization (L1)
                reg_lambda=2.0, # เพิ่ม regularization (L2)
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X_scaled, y)
            models['xgb'] = xgb_model
        except Exception as e:
            print(f"ฝึกสูตรล้มเหลว: {e}")
            
        # ส่วนที่ 5: โครงข่ายประสาทเทียม (MLP / Neural Network) - Reduced size
        try:
            mlp_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
            mlp_model.fit(X_scaled, y)
            models['mlp'] = mlp_model
        except Exception as e:
            print(f"ฝึก MLP ล้มเหลว: {e}")

        # Meta-Model (Stacking Classifier using CV)
        try:
            rf_proba = cross_val_predict(models['rf'], X_scaled, y, cv=5, method='predict_proba')
            xgb_proba = cross_val_predict(models['xgb'], X_scaled, y, cv=5, method='predict_proba') if 'xgb' in models else np.zeros((len(X), 3))
            mlp_proba = cross_val_predict(models['mlp'], X_scaled, y, cv=5, method='predict_proba') if 'mlp' in models else np.zeros((len(X), 3))
            
            X_meta = np.hstack((rf_proba, xgb_proba, mlp_proba))
            meta_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            meta_model.fit(X_meta, y)
            models['meta'] = meta_model
        except Exception as e:
            print(f"ฝึก Meta-Model ล้มเหลว: {e}")
        
        # ส่วนที่ 1: เซียนเฝ้าขอน (KNN)
        if pattern_sequences:
            knn_X = np.array([p['pattern'] for p in pattern_sequences])
            knn_y = np.array([p['target'] for p in pattern_sequences])
            
            if len(knn_X) > 10:
                knn_model = KNeighborsClassifier(n_neighbors=min(7, len(knn_X)//3), weights='distance')
                knn_model.fit(knn_X, knn_y)
                models['knn'] = knn_model
    
    return models

def get_pattern_stats(history, pattern_sequences):
    """คำนวณสถิติย้อนหลังของแพทเทิร์นปัจจุบัน"""
    if len(history) < 5 or not pattern_sequences:
        return None
        
    non_tie_hist = [h for h in history if h != 2]
    if len(non_tie_hist) < 5:
        return None
        
    curr_pat = history[-5:]
    matches = [p for p in pattern_sequences if p['pattern'] == curr_pat]
    
    if not matches:
        return None
        
    total = len(matches)
    p_count = sum(1 for m in matches if m['target'] == 0)
    b_count = sum(1 for m in matches if m['target'] == 1)
    
    return {
        'total': total,
        'p_rate': (p_count / total) * 100 if total > 0 else 0,
        'b_rate': (b_count / total) * 100 if total > 0 else 0
    }

def ensemble_predict(history, models, module_performance=None, st_session_state=None, pattern_sequences=None):
    """ระบบโหวตสภาเซียน (Council Thinking)"""
    votes = {'player': 0, 'banker': 0, 'tie': 0}
    vote_details = {}
    
    non_tie_hist = [h for h in history if h != 2]
    if len(non_tie_hist) < 5:
        return None, 0, {"error": "ข้อมูลไม่พอ"}
    
    # วิเคราะห์ทรงไพ่เพื่อปรับน้ำหนักเซียน
    shoe_name, _ = get_shoe_type(history)
    multipliers = {'จำทางไพ่': 1.0, 'อ่านสามเกลอ': 1.0, 'สถิติรวม': 1.0, 'วิเคราะห์': 1.0, 'สูตรเด็ด': 1.0}
    
    if "Ping Pong" in shoe_name:
        multipliers['อ่านสามเกลอ'] = 1.5
    elif "Stable" in shoe_name:
        multipliers['จำทางไพ่'] = 1.5
        multipliers['สูตรเด็ด'] = 1.5

    # ส่วนที่ 1: เซียนเฝ้าขอน (KNN)
    if 'knn' in models:
        if len(history) >= 5:
            pattern = np.array(history[-5:]).reshape(1, -1)
            knn_pred = models['knn'].predict(pattern)[0]
            knn_proba = models['knn'].predict_proba(pattern)[0]
            knn_conf = max(knn_proba) * 100
            
            w = 2 * multipliers['จำทางไพ่']
            if knn_pred == 0:
                votes['player'] += w
                vote_details['เฝ้าขอน'] = {'vote': 'PLAYER', 'conf': knn_conf, 'emoji': '📜'}
            elif knn_pred == 1:
                votes['banker'] += w
                vote_details['เฝ้าขอน'] = {'vote': 'BANKER', 'conf': knn_conf, 'emoji': '📜'}
    
    # ส่วนที่ 2: เซียนเค้าไพ่ (Technician)
    tech_vote, roads_agree = get_technician_vote(history)
    derived_overall = get_derived_roads_features(history)
    tech_weight = 1.0 * multipliers['อ่านสามเกลอ']
    if derived_overall['overall_stability'] < 0.4:
        tech_weight *= 0.5 
        
    if tech_vote is not None:
        target = 'player' if tech_vote == 0 else ('banker' if tech_vote == 1 else 'tie')
        votes[target] += tech_weight
        vote_details['เค้าไพ่'] = {'vote': target.upper(), 'roads': roads_agree, 'emoji': '🛣️'}
    
    # Machine Learning Models Block (RF, XGB, MLP, Meta-Model)
    if 'rf' in models and len(history) >= 5:
        p1, p2, p3, p4, p5 = history[-5:]
        derived_features = get_derived_roads_features(history)
        current_streak = calculate_streak(history)
        tie_rate, gap_since_tie = calculate_tie_stats(history, 20)
        last_10 = [h for h in history[-10:] if h != 2]
        b_ratio = last_10.count(1) / len(last_10) if last_10 else 0.5
        
        last_20 = [h for h in history[-20:] if h != 2]
        p_rate_20 = last_20.count(0) / len(last_20) if last_20 else 0.5
        b_rate_20 = last_20.count(1) / len(last_20) if last_20 else 0.5
        
        streak_owner = 0.5
        non_tie_hist_local = [h for h in history if h != 2]
        if len(non_tie_hist_local) >= 2:
            if non_tie_hist_local[-1] == 0: streak_owner = 0
            elif non_tie_hist_local[-1] == 1: streak_owner = 1

        markov = calculate_markov_features(history)

        input_data = pd.DataFrame([{
            'pattern_1': p1, 'pattern_2': p2, 'pattern_3': p3,
            'pattern_4': p4, 'pattern_5': p5,
            'current_streak': min(current_streak, 10),
            'streak_owner': streak_owner,
            'banker_trend': b_ratio,
            'player_rate_20': p_rate_20,
            'banker_rate_20': b_rate_20,
            'tie_rate_20': tie_rate,
            'gap_since_tie': min(gap_since_tie, 30),
            'is_stable': derived_features['is_stable'],
            'overall_stability': derived_features['overall_stability'],
            'position_phase': get_position_phase(len(history)),
            'momentum': calculate_momentum(history),
            'alternating_ratio': calculate_alternating_ratio(history),
            'streak_break_freq': calculate_streak_break_frequency(history),
            'pp_ratio': count_consecutive_pairs(history)[0],
            'bb_ratio': count_consecutive_pairs(history)[1],
            'tie_proximity': calculate_tie_proximity(history),
            'baccarat_symmetry': calculate_symmetry(history),
            'p_to_p': markov['p_to_p'],
            'p_to_b': markov['p_to_b'],
            'b_to_b': markov['b_to_b'],
            'b_to_p': markov['b_to_p']
        }])
        
        input_scaled = models['scaler'].transform(input_data) if 'scaler' in models else input_data
        
        rf_proba = np.zeros((1, 3))
        xgb_proba = np.zeros((1, 3))
        mlp_proba = np.zeros((1, 3))
        
        # 1. Random Forest
        rf_pred = models['rf'].predict(input_scaled)[0]
        rf_proba = models['rf'].predict_proba(input_scaled)
        rf_conf = max(rf_proba[0]) * 100
        vote_text = 'PLAYER' if rf_pred == 0 else ('BANKER' if rf_pred == 1 else 'TIE')
        vote_details['สถิติ'] = {'vote': vote_text, 'conf': rf_conf, 'emoji': '🧠'}
        
        # 2. XGBoost
        if 'xgb' in models:
            try:
                xgb_pred = models['xgb'].predict(input_scaled)[0]
                xgb_proba = models['xgb'].predict_proba(input_scaled)
                xgb_conf = max(xgb_proba[0]) * 100
                xgb_vote_text = 'PLAYER' if xgb_pred == 0 else ('BANKER' if xgb_pred == 1 else 'TIE')
                vote_details['วิเคราะห์'] = {'vote': xgb_vote_text, 'conf': xgb_conf, 'emoji': '⚡'}
            except Exception as e:
                print(f"XGBoost error: {e}")
                
        # 3. MLP (Neural Network)
        if 'mlp' in models:
            try:
                mlp_pred = models['mlp'].predict(input_scaled)[0]
                mlp_proba = models['mlp'].predict_proba(input_scaled)
                mlp_conf = max(mlp_proba[0]) * 100
                mlp_vote_text = 'PLAYER' if mlp_pred == 0 else ('BANKER' if mlp_pred == 1 else 'TIE')
                vote_details['โครงข่ายประสาท'] = {'vote': mlp_vote_text, 'conf': mlp_conf, 'emoji': '🧬'}
            except Exception as e:
                print(f"MLP error: {e}")
                
        # 4. Meta-Model Integration (Stacking) & Statistical Additive Voting
        if 'meta' in models:
            # นำความน่าจะเป็นจากเซียนแต่ละสาย (RF, XGB, MLP) ส่งตั้งต้นให้เซียนระดับสูง (Meta) ตัดสิน
            X_meta_input = np.hstack((rf_proba, xgb_proba, mlp_proba))
            try:
                meta_pred = models['meta'].predict(X_meta_input)[0]
                meta_proba = models['meta'].predict_proba(X_meta_input)
                meta_conf = max(meta_proba[0]) * 100
                
                # เชื่อการโหวตจาก Meta-Model แทนการบวกน้ำหนักด้วยมือ
                w = 3.0 if meta_conf > 60 else 1.5
                if meta_pred == 0: votes['player'] += w * multipliers['สถิติรวม']
                elif meta_pred == 1: votes['banker'] += w * multipliers['สถิติรวม']
                elif meta_pred == 2: votes['tie'] += w * 1.5 * multipliers['สถิติรวม']
                
                meta_vote_text = 'PLAYER' if meta_pred == 0 else ('BANKER' if meta_pred == 1 else 'TIE')
                vote_details['สถิติรวม'] = {'vote': meta_vote_text, 'conf': meta_conf, 'emoji': '🤖'}
            except Exception as e:
                print(f"Meta-Model Error: {e}")
        else:
            # Fallback (กรณีเทรน Meta ไม่สำเร็จ)
            if max(rf_proba[0]) > 0.55:
                if rf_pred == 0: votes['player'] += 1.0 * multipliers['สถิติรวม']
                elif rf_pred == 1: votes['banker'] += 1.0 * multipliers['สถิติรวม']
                elif rf_pred == 2: votes['tie'] += 1.5 * multipliers['สถิติรวม']
                
            if 'xgb' in models and max(xgb_proba[0]) > 0.5:
                w = 2.0 if max(xgb_proba[0]) > 0.75 else 1.5
                if xgb_pred == 0: votes['player'] += w * multipliers['วิเคราะห์']
                elif xgb_pred == 1: votes['banker'] += w * multipliers['วิเคราะห์']
                elif xgb_pred == 2: votes['tie'] += w * 1.2 * multipliers['วิเคราะห์']
                
            if 'mlp' in models and max(mlp_proba[0]) > 0.5:
                w = 1.5 if max(mlp_proba[0]) > 0.7 else 1.0
                if mlp_pred == 0: votes['player'] += w
                elif mlp_pred == 1: votes['banker'] += w
                elif mlp_pred == 2: votes['tie'] += w * 1.5
    
    # ส่วนที่ 5: ล็อคแพทเทิร์น (Expert Rules)
    if 'patterns' in models and models['patterns']:
        expert_vote, pattern_name = get_expert_vote(history, models['patterns'])
        if expert_vote is not None:
            target = 'player' if expert_vote == 0 else ('banker' if expert_vote == 1 else 'tie')
            w = (2 if target != 'tie' else 3) * multipliers['สูตรเด็ด']
            votes[target] += w
            vote_details['สูตรเด็ด'] = {'vote': target.upper(), 'pattern': pattern_name, 'emoji': '🎲'}
    
    # Adaptive Learning
    if st_session_state and 'session_mistakes' in st_session_state:
        current_pat = non_tie_hist[-5:]
        for mistake in st_session_state['session_mistakes']:
            if len(current_pat) >= 3 and current_pat == mistake['pattern']:
                penalty = 0.5
                if mistake['wrong_predict'] == 0: votes['player'] -= penalty
                elif mistake['wrong_predict'] == 1: votes['banker'] -= penalty

    # Dynamic Weighting from live performance
    if module_performance:
        name_map = {'historian': 'จำทางไพ่', 'technician': 'อ่านสามเกลอ', 'statistician': 'สถิติรวม', 'booster': 'วิเคราะห์', 'expert': 'สูตรเด็ด'}
        for mod, perf_score in module_performance.items():
            local_name = name_map.get(mod, mod)
            if local_name in vote_details:
                mod_vote = vote_details[local_name].get('vote')
                bonus = 0.5 if perf_score >= 2 else (-0.5 if perf_score <= -2 else 0)
                if bonus != 0 and mod_vote in ['PLAYER', 'BANKER', 'TIE']:
                    target = mod_vote.lower()
                    votes[target] += bonus

    # สถิติย้อนหลัง (Pattern Stats)
    pat_stats = get_pattern_stats(history, pattern_sequences)

    # Final Decision
    max_v = max(votes.values())
    sorted_v = sorted(votes.values(), reverse=True)
    margin = sorted_v[0] - sorted_v[1] if len(sorted_v) > 1 else max_v
    
    # Meta-Labeling (Skip/Wait) Logic Check
    if max_v < 1.0 or (margin < 0.5 and sorted_v[0] > 0):
        final_prediction = 3 # 3 is ACTION_SKIP
    elif votes['tie'] == max_v and votes['tie'] >= 1.5:
        final_prediction = 2
    elif votes['player'] > votes['banker']:
        final_prediction = 0
    else:
        final_prediction = 1

    # Consensus Checking (How many experts agree on the final choice)
    consensus_count = 0
    if final_prediction in [0, 1, 2]:
        target_name = 'PLAYER' if final_prediction == 0 else ('BANKER' if final_prediction == 1 else 'TIE')
        for v in vote_details.values():
            if v.get('vote') == target_name:
                consensus_count += 1

    return final_prediction, max_v, vote_details, pat_stats, consensus_count

def snap_to_chip(amount, min_bet=10):
    """
    ปัดเศษเงินให้ตรงกับชิปจริงในคาสิโนออนไลน์ และให้เป็นตัวเลขกลมๆ คลิกง่าย
    เช่น หลักสิบลงท้ายด้วย 0, หลักร้อยปัดทีละ 50 หรือ 100
    """
    if amount < min_bet:
        return min_bet
        
    if amount <= 100:
        # หลักสิบ ปัดให้เป็นสูตรคูณของ 10
        return max(min_bet, round(amount / 10) * 10)
    elif amount <= 500:
        # หลักร้อยต้น ปัดให้เป็นสูตรคูณของ 50 (เช่น 150, 200, 250)
        return round(amount / 50) * 50
    else:
        # ยอดใหญ่ ปัดให้เป็นสูตรคูณของ 100 (เช่น 600, 700, 1000)
        return round(amount / 100) * 100

def calculate_dog_oh_advice(capital, target_profit, current_profit, risk_level, urgency, score, consensus, ai_streak=0, last_big_miss=False, min_bet=10.0):
    """
    กุนซือการเงินด๊อกโอ: คำนวณยอดเงินที่ควรแทงตามสภาพจิตใจ เป้าหมาย และ โหมดกู้ชีพ (Loss Recovery)
    """
    current_balance = capital + current_profit
    
    if current_profit >= target_profit:
        return 0, "💰 ภารกิจสำเร็จ! ถึงเป้าหมายแล้ว แนะนำให้ถอนเงินและเลิกเล่นทันทีครับ"
        
    # Stop Loss Check ป้องกันการแทงจนติดลบเกินทุน
    if current_balance <= 0:
        return 0, "⚠️ พอร์ตแตกแล้ว (Balance = 0)! ขาดทุนเต็มเพดาน ระบบสั่งหยุดการทำงานเพื่อเซฟเงินครับ"
    
    if current_balance < min_bet:
        return 0, "⚠️ ยอดเงินปัจจุบันเหลือน้อยกว่าขั้นต่ำที่จะแทงได้ แนะนำให้พอแค่นี้ครับ"
    
    # 1. Base Unit calculation (5% ของทุนปัจจุบันถ้า < 1000, 2% ถ้าทุนหนา)
    unit_pct = 0.05 if capital < 1000 else 0.02
    if urgency == "ร้อนเงิน (ต้องชัวร์)": unit_pct *= 0.7
    if risk_level == "สายซิ่ง (Aggressive)": unit_pct *= 1.5
    elif risk_level == "เน้นปลอดภัย (Safe)": unit_pct *= 0.5
    
    base_unit = max(min_bet, capital * unit_pct)
    
    # 2. Reality Check (เพิ่งเสียหนักมา ห้ามหัวร้อนทบ สั่ง skip)
    if last_big_miss:
        return 0, "🚨 ตาที่แล้วพลาดไม้ใหญ่! กฎเหล็กกู้ชีพคือ 'หยุดพัก' (Skip) 1 ตาเพื่อรอดูทรงไพ่ (ห้ามหัวร้อน)"

    # 3. ตรวจสอบโหมดกู้ชีพ (Loss Recovery Mode)
    in_recovery_mode = current_profit < 0
    
    if in_recovery_mode:
        # โหมดกู้ชีพ: ซุ่มยิง เน้นชัวร์เท่านั้น
        if score < 4.0:
            return 0, f"🛡️ [โหมดกู้ชีพ] ความมั่นใจ {score:.1f} ต่ำไป กุนซือแนะนำให้ 'รอไพ่' (Skip) เพื่อเซฟทุนครับ"
        if ai_streak == 0:
            return 0, f"🛡️ [โหมดกู้ชีพ] เซียนเพิ่งทายพลาดไป รอให้ตีบวกถูกสัก 1 ตาก่อนค่อยสวนเงินครับ"
    else:
        # โหมดเก็บกำไรปกติ
        entry_threshold = 3.5 if urgency == "ร้อนเงิน (ต้องชัวร์)" else 2.5
        if score < entry_threshold:
            return 0, f"⏸️ ความมั่นใจ {score:.1f} ต่ำเกินไป ด๊อกโอขอให้ผ่านครับ"
    
    # 4. Dynamic Paroli / Pressing based on ai_streak (ทบเมื่อชนะ เพื่อทำกำไรรวดเร็ว)
    multiplier = 1.0
    if ai_streak == 1:
        multiplier = 1.5
    elif ai_streak == 2:
        multiplier = 2.0
    elif ai_streak >= 3:
        multiplier = 3.0 # Maximum pressing during hot streak
        
    # 5. Score Factor
    score_multiplier = 1.0
    if score >= 4.5 and consensus >= 3:
        score_multiplier = 1.5
    if score >= 5.5 and consensus >= 4:
        score_multiplier = 2.0
        
    bet_amount = base_unit * multiplier * score_multiplier
    
    # โหมดกู้ชีพ จะใช้วิธีดึงทุนคืนทีละ 1/3 เพื่อไม่ให้ Overtrade
    if in_recovery_mode:
        recovery_bet = max(min_bet, abs(current_profit) / 3)
        # เลือกค่าที่ปลอดภัยที่สุดระหว่างไม้ปกติหรือไม้กู้ชีพ
        bet_amount = min(bet_amount, recovery_bet)
        
    # 6. Max allowed bet per hand to protect capital
    max_bet = capital * 0.25
    bet_amount = min(max_bet, bet_amount)
    
    # 7. บังคับไม่ให้แทงเกินเงินทุนที่มีอยู่จริงตอนนี้ (ป้องกันยอดติดลบทะลุโลก)
    bet_amount = min(bet_amount, current_balance)
    
    # 8. ปัดเศษชิปตามคาสิโนจริง
    bet_amount = snap_to_chip(bet_amount, min_bet)
    
    # เช็คก๊อกสุดท้าย ถ้าปัดเศษแล้วเกินทุน ให้อัดหมดหน้าตักที่มี (แต่ต้องมากกว่าขั้นต่ำ)
    if bet_amount > current_balance:
        bet_amount = snap_to_chip(current_balance, min_bet)

    # Generating correct advice text
    if in_recovery_mode:
        advice_text = "🛡️ [โหมดกู้ชีพ] กราฟเริ่มเข้าทาง ยิงไม้กู้คืนแบบปลอดภัย"
        if score >= 5.0:
            advice_text = "🎯 [โหมดกู้ชีพ + ไม้เด็ด] โอกาสทองมาแล้ว ทวงเงินคืน!"
    else:
        advice_text = "✅ จังหวะดี ความเสี่ยงต่ำ ลงเบาๆ 1 อัตรา"
        if ai_streak >= 3:
            advice_text = "🔥 ระบบกำลังไหลลื่น! อัดกำไรสู้ตามสูตรทบไม้ (Paroli)"
        elif ai_streak >= 1:
            advice_text = "✨ กราฟกำลังมา ด๊อกโอแนะนำเพิ่มน้ำหนักเดิมพันบวกกำไร"
        elif score >= 5.0:
            advice_text = "🚀 **ไม้เด็ดด๊อกโอ!** โอกาสทองที่เซียนทุกสายโหวตตรงกัน ใส่เต็มรอบนี้"
        elif not urgency == "ร้อนเงิน (ต้องชัวร์)":
            advice_text = "🛡️ เดินเงินขั้นต่ำตามระบบ"

    if current_profit <= -(capital * 0.4) and in_recovery_mode:
        advice_text = "⚠️ [วิกฤต] ทุนลดลงหนักมาก! โปรดแทงอย่างระมัดระวังที่สุด " + advice_text

    return bet_amount, advice_text
