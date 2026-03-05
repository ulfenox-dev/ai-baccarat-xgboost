import os
import glob
import pandas as pd
from datetime import datetime

def load_all_games(data_folder="./data"):
    """โหลดข้อมูลเกมทั้งหมดจาก Data Folder"""
    if not os.path.exists(data_folder):
        return []
    
    all_files = glob.glob(os.path.join(data_folder, "*.txt"))
    all_games = []
    
    for f in all_files:
        try:
            with open(f, 'r') as file:
                content = file.read().strip()
                raw = content.split(']')[1] if ']' in content else content
                seq = [int(x) for x in raw.replace(',', ' ').split() if x.strip().isdigit()]
                all_games.append([x for x in seq if x != 2]) # Filter ties
        except Exception as e:
            print(f"Error loading game {f}: {e}")
    return all_games

def load_custom_patterns(folder_path="./pattern"):
    """โหลด Pattern จากโฟลเดอร์ pattern/"""
    patterns = []
    
    if not os.path.exists(folder_path):
        return patterns
    
    pattern_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    for filepath in pattern_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    if '=' in line:
                        pattern_str, expected_str = line.split('=')
                        pattern = [int(x) for x in pattern_str.split(',')]
                        expected = int(expected_str)
                    else:
                        pattern = [int(x) for x in line.split(',')]
                        expected = None
                        
                    patterns.append({
                        'pattern': pattern,
                        'expected': expected,
                        'name': os.path.basename(filepath).replace('.txt', '')
                    })
        except Exception as e:
            print(f"Error loading pattern {filepath}: {e}")
    
    return patterns

def calculate_streak(history):
    """คำนวณความยาว streak ปัจจุบัน"""
    if not history:
        return 0
    non_tie = [h for h in history if h != 2]
    if not non_tie:
        return 0
    streak = 1
    last_winner = non_tie[-1]
    for i in range(len(non_tie) - 2, -1, -1):
        if non_tie[i] == last_winner:
            streak += 1
        else:
            break
    return streak

def calculate_tie_stats(history, window=20):
    """คำนวณสถิติ Tie"""
    if not history:
        return 0, 0
    last_n = history[-window:]
    tie_rate = last_n.count(2) / len(last_n) if last_n else 0
    gap = 0
    for i in range(len(history) - 1, -1, -1):
        if history[i] == 2:
            break
        gap += 1
    return tie_rate, gap

def save_game_data(history, data_folder="./data"):
    """บันทึกข้อมูลเกมลงไฟล์ .txt"""
    if not history:
        return None
    
    os.makedirs(data_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game_{timestamp}.txt"
    filepath = os.path.join(data_folder, filename)
    
    data_str = ",".join(str(x) for x in history) + ","
    with open(filepath, 'w') as f:
        f.write(data_str)
    
    return filename

def mine_patterns(data_folder="./data", pattern_folder="./pattern", 
                  min_occurrences=5, min_win_rate=0.55, pattern_lengths=[3, 4, 5]):
    """
    Auto-discover high win-rate patterns from historical data.
    Returns list of discovered patterns and saves them to pattern_folder.
    """
    all_games = load_all_games(data_folder)
    if not all_games:
        return [], "No games found in data folder"
    
    # Dictionary to track pattern statistics
    # pattern_stats[pattern_tuple] = {'next_p': count, 'next_b': count}
    pattern_stats = {}
    
    # Analyze all games
    for game in all_games:
        non_tie = [h for h in game if h != 2]
        for pat_len in pattern_lengths:
            for i in range(len(non_tie) - pat_len):
                pattern = tuple(non_tie[i:i+pat_len])
                next_result = non_tie[i + pat_len]
                
                if pattern not in pattern_stats:
                    pattern_stats[pattern] = {'next_p': 0, 'next_b': 0, 'total': 0}
                
                pattern_stats[pattern]['total'] += 1
                if next_result == 0:
                    pattern_stats[pattern]['next_p'] += 1
                elif next_result == 1:
                    pattern_stats[pattern]['next_b'] += 1
    
    # Filter patterns with high win rate
    discovered_patterns = []
    for pattern, stats in pattern_stats.items():
        total = stats['total']
        if total < min_occurrences:
            continue
        
        p_rate = stats['next_p'] / total
        b_rate = stats['next_b'] / total
        
        if p_rate >= min_win_rate:
            discovered_patterns.append({
                'pattern': list(pattern),
                'expected': 0,  # PLAYER
                'win_rate': p_rate,
                'occurrences': total,
                'name': 'auto_mined'
            })
        elif b_rate >= min_win_rate:
            discovered_patterns.append({
                'pattern': list(pattern),
                'expected': 1,  # BANKER
                'win_rate': b_rate,
                'occurrences': total,
                'name': 'auto_mined'
            })
    
    # Sort by win rate (highest first)
    discovered_patterns.sort(key=lambda x: x['win_rate'], reverse=True)
    
    # Limit to top 50 patterns
    discovered_patterns = discovered_patterns[:50]
    
    # Save to pattern folder
    if discovered_patterns:
        os.makedirs(pattern_folder, exist_ok=True)
        
        # ลบไฟล์ auto_mined เก่าออกก่อน เพื่อไม่ให้แพทเทิร์นซ้ำซ้อนเวลาผู้ใช้กดหลายรอบ
        old_pattern_files = glob.glob(os.path.join(pattern_folder, "auto_mined_*.txt"))
        for old_file in old_pattern_files:
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"Error removing old pattern file {old_file}: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(pattern_folder, f"auto_mined_{timestamp}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Auto-mined patterns\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Min occurrences: {min_occurrences}, Min win rate: {min_win_rate}\n\n")
            
            for p in discovered_patterns:
                pattern_str = ','.join(str(x) for x in p['pattern'])
                f.write(f"{pattern_str}={p['expected']}\n")
    
    return discovered_patterns, f"Found {len(discovered_patterns)} high win-rate patterns"
