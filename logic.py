import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
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

def process_data_from_folder(folder_path):
    """อ่านไฟล์ .txt ทั้งหมดใน data folder และสร้าง features"""
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
        
        # Module C: Random Forest (Statistician)
        rf_model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
        rf_model.fit(X, y)
        models['rf'] = rf_model
        models['rf_features'] = list(X.columns)
        
        # Module E: XGBoost (Booster) - Tuned to prevent Overfitting!
        try:
            xgb_model = XGBClassifier(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X, y)
            models['xgb'] = xgb_model
        except Exception as e:
            print(f"XGBoost training failed: {e}")
        
        # Module A: KNN (Historian)
        if pattern_sequences:
            knn_X = np.array([p['pattern'] for p in pattern_sequences])
            knn_y = np.array([p['target'] for p in pattern_sequences])
            
            if len(knn_X) > 10:
                knn_model = KNeighborsClassifier(n_neighbors=min(7, len(knn_X)//3), weights='distance')
                knn_model.fit(knn_X, knn_y)
                models['knn'] = knn_model
    
    return models

def ensemble_predict(history, models, module_performance=None, st_session_state=None):
    """Hybrid Ensemble Voting System"""
    votes = {'player': 0, 'banker': 0, 'tie': 0}
    vote_details = {}
    
    non_tie_hist = [h for h in history if h != 2]
    if len(non_tie_hist) < 5:
        return None, 0, {"error": "Need more data"}
    
    # Module A: KNN
    if 'knn' in models:
        # Use full history (including ties) to match training data structure
        if len(history) >= 5:
            pattern = np.array(history[-5:]).reshape(1, -1)
            knn_pred = models['knn'].predict(pattern)[0]
            knn_proba = models['knn'].predict_proba(pattern)[0]
            knn_conf = max(knn_proba) * 100
            
            if knn_pred == 0:
                votes['player'] += 2
                vote_details['historian'] = {'vote': 'PLAYER', 'conf': knn_conf, 'emoji': '📜'}
            elif knn_pred == 1:
                votes['banker'] += 2
                vote_details['historian'] = {'vote': 'BANKER', 'conf': knn_conf, 'emoji': '📜'}
            else:
                 vote_details['historian'] = {'vote': 'TIE/SKIP', 'conf': knn_conf, 'emoji': '📜'}
    
    # Module B: Technician 
    # (ลดน้ำหนักถ้าห้องแกว่งมากๆ เพราะเค้าไพ่หลัก/รอง จะโดนทำลาย)
    tech_vote, roads_agree = get_technician_vote(history)
    derived_overall = get_derived_roads_features(history)
    tech_weight = 1.0
    if derived_overall['overall_stability'] < 0.4:
        tech_weight = 0.5 # ห้อง Volatile อย่าไว้ใจ Technician มาก
        
    if tech_vote is not None:
        target = 'player' if tech_vote == 0 else ('banker' if tech_vote == 1 else 'tie')
        votes[target] += tech_weight
        vote_details['technician'] = {'vote': target.upper(), 'roads': roads_agree, 'emoji': '🛣️'}
    
    # Module C: RF
    if 'rf' in models:
        if len(history) >= 5:
            p1, p2, p3, p4, p5 = history[-5:]
            derived_features = get_derived_roads_features(history)
            current_streak = calculate_streak(history)
            tie_rate, gap_since_tie = calculate_tie_stats(history, 20)
            last_10 = [h for h in history[-10:] if h != 2] # Trends still prefer non-tie ratio
            b_ratio = last_10.count(1) / len(last_10) if last_10 else 0.5
            
            last_20 = [h for h in history[-20:] if h != 2]
            p_rate_20 = last_20.count(0) / len(last_20) if last_20 else 0.5
            b_rate_20 = last_20.count(1) / len(last_20) if last_20 else 0.5
            
            streak_owner = 0.5
            non_tie_hist = [h for h in history if h != 2] # Use non-tie for streak owner
            if len(non_tie_hist) >= 2:
                if non_tie_hist[-1] == 0: streak_owner = 0
                elif non_tie_hist[-1] == 1: streak_owner = 1

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
                'baccarat_symmetry': calculate_symmetry(history)
            }])
            
            rf_pred = models['rf'].predict(input_data)[0]
            rf_proba = models['rf'].predict_proba(input_data)[0]
            rf_conf = max(rf_proba) * 100
            
            if rf_conf > 55:
                if rf_pred == 0: votes['player'] += 1
                elif rf_pred == 1: votes['banker'] += 1
                elif rf_pred == 2: votes['tie'] += 1.5 # Tie gets bonus weight as it's rare
            
            vote_text = 'PLAYER' if rf_pred == 0 else ('BANKER' if rf_pred == 1 else 'TIE')
            vote_details['statistician'] = {'vote': vote_text, 'conf': rf_conf, 'emoji': '🧠'}
    
    # Module E: XGBoost (Booster) - New!
    if 'xgb' in models:
        if len(history) >= 5:
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

            xgb_input = pd.DataFrame([{
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
                'baccarat_symmetry': calculate_symmetry(history)
            }])
            
            try:
                xgb_pred = models['xgb'].predict(xgb_input)[0]
                xgb_proba = models['xgb'].predict_proba(xgb_input)[0]
                xgb_conf = max(xgb_proba) * 100
                
                # XGBoost gets base weight of 1.5. If heavily confident (> 75%), boost to 2.0.
                xgb_weight = 2.0 if xgb_conf > 75 else 1.5
                if xgb_conf > 50:
                    if xgb_pred == 0: votes['player'] += xgb_weight
                    elif xgb_pred == 1: votes['banker'] += xgb_weight
                    elif xgb_pred == 2: votes['tie'] += 2
                
                xgb_vote_text = 'PLAYER' if xgb_pred == 0 else ('BANKER' if xgb_pred == 1 else 'TIE')
                vote_details['booster'] = {'vote': xgb_vote_text, 'conf': xgb_conf, 'emoji': '⚡'}
            except Exception as e:
                print(f"XGBoost prediction error: {e}")
    
    # Module D: Expert
    if 'patterns' in models and models['patterns']:
        expert_vote, pattern_name = get_expert_vote(history, models['patterns'])
        if expert_vote is not None:
            target = 'player' if expert_vote == 0 else ('banker' if expert_vote == 1 else 'tie')
            weight = 2 if target != 'tie' else 3 # Higher weight for expert tie patterns
            votes[target] += weight
            vote_details['expert'] = {'vote': target.upper(), 'pattern': pattern_name, 'emoji': '🎲'}
    
    # Session Learning from st_session_state
    if st_session_state and 'session_mistakes' in st_session_state:
        current_pat = non_tie_hist[-5:]
        for mistake in st_session_state['session_mistakes']:
            if len(current_pat) >= 3 and current_pat == mistake['pattern']:
                if mistake['wrong_predict'] == 0:
                    votes['player'] -= 0.5
                elif mistake['wrong_predict'] == 1:
                    votes['banker'] -= 0.5

    # Dynamic Weighting
    if module_performance:
        for mod, perf_score in module_performance.items():
            if mod in vote_details:
                mod_vote = vote_details[mod].get('vote')
                bonus = 0.5 if perf_score >= 2 else (-0.5 if perf_score <= -2 else 0)
                if bonus != 0 and mod_vote in ['PLAYER', 'BANKER', 'TIE']:
                    target = 'player' if mod_vote == 'PLAYER' else ('banker' if mod_vote == 'BANKER' else 'tie')
                    votes[target] += bonus

    # Final Decision
    max_v = max(votes.values())
    if votes['tie'] == max_v and votes['tie'] >= 1.5:
        final_prediction = 2
    elif votes['player'] > votes['banker']:
        final_prediction = 0
    else:
        final_prediction = 1
    
    total_score = max_v
    return final_prediction, total_score, vote_details
