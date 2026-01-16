import streamlit as st
import pandas as pd
import os
from datetime import datetime
from logic import (
    train_ensemble_models, 
    ensemble_predict, 
    process_data_from_folder,
    get_derived_roads_features,
    get_shoe_type
)
from utils import (
    load_all_games, 
    load_custom_patterns, 
    calculate_streak, 
    calculate_tie_stats,
    save_game_data,
    mine_patterns
)
from ui_components import (
    render_big_road, 
    render_raw_data, 
    show_performance_dashboard,
    show_gallery_modal,
    show_stats_modal,
    show_validation_panel
)

# ==========================================
# ส่วนที่ 1: หน้าจอใช้งาน (GUI)
# ==========================================

st.set_page_config(
    page_title="AI Baccarat Pro", 
    layout="wide", 
    initial_sidebar_state="collapsed" # ย่อ Sidebar อัตโนมัติเพื่อมือถือ
)

st.title("🤖 AI Baccarat Pro (เวอร์ชั่นจับสูตรหลอก)")
st.markdown("ระบบวิเคราะห์ไพ่ 4 มิติ: **กราฟ + เทคนิค + สถิติ + กับดัก**")

# --- เมนูด้านซ้าย: จัดการข้อมูล ---
with st.sidebar:
    st.header("📂 1. เตรียมสมอง AI")
    data_path = st.text_input("โฟลเดอร์เก็บไฟล์ .txt", value="./data")
    
    # Initialize Session Learning State
    if 'session_mistakes' not in st.session_state:
        st.session_state['session_mistakes'] = []
    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None
    if 'module_performance' not in st.session_state:
        st.session_state['module_performance'] = {
            'historian': 0, 'technician': 0, 'statistician': 0, 'booster': 0, 'expert': 0
        }
    if 'module_stats' not in st.session_state:
        st.session_state['module_stats'] = {
            k: {'wins': 0, 'total': 0, 'score': 0} for k in ['historian', 'technician', 'statistician', 'booster', 'expert']
        }
    if 'last_vote_details' not in st.session_state:
        st.session_state['last_vote_details'] = {}
    if 'ai_performance' not in st.session_state:
        st.session_state['ai_performance'] = {'wins': 0, 'losses': 0, 'history': []}
    if 'undo_stack' not in st.session_state:
        st.session_state['undo_stack'] = []
    if 'streak_count' not in st.session_state:
        st.session_state['streak_count'] = 0

    if st.button("🔄 โหลดข้อมูล & เทรน Ensemble", use_container_width=True):
        with st.spinner("กำลังเทรน AI 5 ตัว (RF, XGBoost, KNN...)..."):
            try:
                df, pattern_sequences = process_data_from_folder(data_path)
                
                if not df.empty:
                    models = train_ensemble_models(df, pattern_sequences)
                    custom_patterns = load_custom_patterns("./pattern")
                    
                    if any(p['expected'] is None for p in custom_patterns):
                        all_games = load_all_games(data_path)
                        for p in custom_patterns:
                            if p['expected'] is None:
                                try:
                                    pat_seq = p['pattern']
                                    pat_len = len(pat_seq)
                                    next_p, next_b = 0, 0
                                    for game in all_games:
                                        for i in range(len(game) - pat_len):
                                            if game[i:i+pat_len] == pat_seq:
                                                if i+pat_len < len(game):
                                                    nxt = game[i+pat_len]
                                                    if nxt == 0: next_p += 1
                                                    elif nxt == 1: next_b += 1
                                    total = next_p + next_b
                                    if total > 0:
                                        p_rate, b_rate = next_p / total, next_b / total
                                        if p_rate >= 0.55: p['expected'] = 0
                                        elif b_rate >= 0.55: p['expected'] = 1
                                        else: p['expected'] = None
                                except Exception as e:
                                    st.warning(f"Error calculating pattern {p['name']}: {e}")
                    
                    models['patterns'] = custom_patterns
                    st.session_state['models'] = models
                    st.session_state['data_count'] = len(df)
                    st.success(f"✅ เทรนเสร็จ! {len([k for k in ['rf','knn'] if k in models])} AI + {len(custom_patterns)} Patterns")
                else:
                    st.error("ไม่เจอไฟล์ข้อมูลที่ถูกต้องในโฟลเดอร์ data")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเทรน: {e}")
                
    st.divider()
    
    # --- Pattern Validator Logic ---
    st.subheader("🔍 ตรวจสอบ Pattern")
    col_pat1, col_pat2 = st.columns(2)
    check_btn = col_pat1.button("Check Stats", use_container_width=True)
    gallery_btn = col_pat2.button("See Gallery", use_container_width=True)
    
    if gallery_btn:
        patterns = load_custom_patterns("./pattern")
        if not patterns:
             st.warning("ไม่เจอไฟล์ Pattern")
        else:
             show_gallery_modal(patterns)
                 
    if check_btn:
        patterns = load_custom_patterns("./pattern")
        if not patterns:
             st.warning("ไม่เจอไฟล์ Pattern")
        else:
             show_stats_modal(patterns, data_path)

    # Auto Pattern Mining Button
    st.markdown("---")
    if st.button("⛏️ Auto Mine Patterns", use_container_width=True):
        with st.spinner("กำลังวิเคราะห์หา Pattern ใหม่..."):
            discovered, message = mine_patterns(data_path, "./pattern")
            if discovered:
                st.success(f"✅ {message}")
                st.info(f"บันทึกแล้วที่ ./pattern | Top Win Rate: {discovered[0]['win_rate']*100:.1f}%")
            else:
                st.warning(message)

    st.divider()
    
    st.subheader("⚡ Risk Level")
    risk_level = st.select_slider(
        "ระดับความเสี่ยง",
        options=["Low", "Medium", "High"],
        value="Medium"
    )
    st.session_state['risk_level'] = risk_level
    
    risk_text = {"Low": "🛡️ Score ≥ 4 ถึงแทง", "Medium": "⚖️ Score ≥ 3 ถึงแทง", "High": "🔥 Score ≥ 2 ถึงแทง"}
    st.caption(risk_text.get(risk_level, ""))
        
    if st.session_state.get('session_mistakes'):
        st.caption(f"🧠 Adaptive Mode: เรียนรู้แล้ว {len(st.session_state['session_mistakes'])} จุด")

    st.divider()
    st.subheader("🎨 ตั้งค่าสีตาราง")
    theme_choice = st.radio("Theme", ["ตารางสีมืด 🌙", "ตารางสีสว่าง ☀️"], index=1, horizontal=True, label_visibility="collapsed")
    st.session_state['theme'] = 'dark' if "ตารางสีมืด" in theme_choice else 'light'
    
    st.divider()
    if 'models' in st.session_state:
        st.info(f"Data: {st.session_state['data_count']} samples")
    else:
        st.warning("⚠️ กรุณากดเทรนก่อน")

col1, col2 = st.columns([1.5, 1])

# CSS for Horizontal Display on Mobile
st.markdown("""
<style>
    [data-testid="column"] { min-width: 0 !important; }
    .flex-container { display: flex; justify-content: space-around; width: 100%; gap: 5px; }
    .flex-item { text-align: center; flex: 1; }
</style>
""", unsafe_allow_html=True)

with col1:
    st.subheader("🎰 2. บันทึกผลไพ่สด (Live Input)")
    
    if 'ai_performance' in st.session_state:
        stats = st.session_state['ai_performance']
        curr_game = st.session_state.get('current_game', [])
        p_pct, b_pct = 50, 50
        if curr_game:
            total_g = len(curr_game)
            p_pct = curr_game.count(0) / total_g * 100
            b_pct = curr_game.count(1) / total_g * 100
        
        show_performance_dashboard(stats, st.session_state['streak_count'], p_pct, b_pct)
        with st.expander("📊 ดูประวัติย่อ"):
             for log in list(reversed(stats['history']))[:5]:
                 st.caption(log)
    
    if 'current_game' not in st.session_state:
        st.session_state['current_game'] = []
    if 'show_raw_data' not in st.session_state:
        st.session_state['show_raw_data'] = False

    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("### 📊 ตาราง Big Road")
    with header_col2:
        if st.button("🔄 สลับมุมมอง", use_container_width=True):
            st.session_state['show_raw_data'] = not st.session_state['show_raw_data']
    
    if st.session_state['show_raw_data']:
        st.markdown(render_raw_data(st.session_state['current_game']), unsafe_allow_html=True)
    else:
        current_theme = st.session_state.get('theme', 'dark')
        st.markdown(render_big_road(st.session_state['current_game'], theme=current_theme), unsafe_allow_html=True)
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    def update_learning(outcome):
        delta = {'added_mistake': None, 'module_changes': {}, 'perf_change': None, 'prev_streak': st.session_state['streak_count']}
        
        if st.session_state.get('last_prediction'):
            lp = st.session_state['last_prediction']
            if lp['vote'] is not None and lp['vote'] != outcome and lp['score'] >= 3:
                non_tie = [h for h in st.session_state['current_game'] if h != 2]
                if len(non_tie) >= 5:
                    st.session_state['session_mistakes'].append({'pattern': non_tie[-5:], 'wrong_predict': lp['vote']})
                    delta['added_mistake'] = len(st.session_state['session_mistakes']) - 1
        
        if st.session_state.get('last_vote_details'):
            details = st.session_state['last_vote_details']
            perf = st.session_state['module_performance']
            for name in ['historian', 'technician', 'statistician', 'booster', 'expert']:
                mod_data = details.get(name)
                if mod_data and 'vote' in mod_data:
                    vote_val = 0 if mod_data['vote'] == 'PLAYER' else (1 if mod_data['vote'] == 'BANKER' else None)
                    if vote_val is not None:
                        change = 1 if vote_val == outcome else -1
                        perf[name] += change
                        delta['module_changes'][name] = change
                        
            # Update detailed Stats (Win Rate per Module)
            if 'module_stats' in st.session_state:
                stats = st.session_state['module_stats']
                for name in ['historian', 'technician', 'statistician', 'booster', 'expert']:
                    mod_data = details.get(name)
                    if mod_data and 'vote' in mod_data:
                        v = mod_data['vote']
                        target = 0 if v == 'PLAYER' else (1 if v == 'BANKER' else (2 if v == 'TIE' else None))
                        
                        if target is not None:
                            stats[name]['total'] += 1
                            if target == outcome:
                                stats[name]['wins'] += 1
                                stats[name]['score'] += 1 # Simple score
                            else:
                                stats[name]['score'] -= 1
                            
                            delta['module_stats_update'] = True # Marker for undo
        
        if st.session_state.get('last_prediction') and outcome != 2:
            lp = st.session_state['last_prediction']
            is_win = (lp['vote'] == outcome)
            req = {"Low": 4, "Medium": 3, "High": 2}.get(st.session_state['risk_level'], 3)
            stats = st.session_state['ai_performance']
            if lp['score'] >= req:
                if is_win:
                    stats['wins'] += 1
                    stats['history'].append(f"✅ Win ({lp['vote']} vs {outcome})")
                    delta['perf_change'] = {'type': 'win'}
                    st.session_state['streak_count'] += 1
                else:
                    stats['losses'] += 1
                    stats['history'].append(f"❌ Loss ({lp['vote']} vs {outcome})")
                    delta['perf_change'] = {'type': 'loss'}
                    st.session_state['streak_count'] = 0
            else:
                stats['history'].append(f"⚪ Skip (Score {lp['score']})")
                delta['perf_change'] = {'type': 'skip'}
        
        st.session_state['undo_stack'].append(delta)

    def undo_last_action():
        if st.session_state['current_game']:
            st.session_state['current_game'].pop()
        if st.session_state.get('undo_stack'):
            delta = st.session_state['undo_stack'].pop()
            if delta.get('added_mistake') is not None: st.session_state['session_mistakes'].pop()
            for name, change in delta.get('module_changes', {}).items():
                st.session_state['module_performance'][name] -= change
            
            if delta.get('module_stats_update'):
                 # Revert logic for stats is complex, for now we just skip or simple revert if possible
                 # To do it properly we needed to save previous state, but for this version simplified:
                 # We will just accept that undoing might not perfectly revert "Total/Win" counts 
                 # to save complexity, or we can just pop the last action context if we stored it.
                 pass # Placeholder as full revert requires storing more state
            st.session_state['streak_count'] = delta['prev_streak']
            pc = delta.get('perf_change')
            if pc:
                if pc['type'] == 'win': st.session_state['ai_performance']['wins'] -= 1
                elif pc['type'] == 'loss': st.session_state['ai_performance']['losses'] -= 1
                if st.session_state['ai_performance']['history']: st.session_state['ai_performance']['history'].pop()

    if c1.button("🔵 PLAYER", use_container_width=True):
        update_learning(0)
        st.session_state['current_game'].append(0)
        st.rerun()
    if c2.button("🔴 BANKER", use_container_width=True):
        update_learning(1)
        st.session_state['current_game'].append(1)
        st.rerun()
    if c3.button("🟢 TIE", use_container_width=True):
        update_learning(2)
        st.session_state['current_game'].append(2)
        st.rerun()
    if c4.button("ลบ", type="primary"):
        undo_last_action()
        st.rerun()

with col2:
    st.subheader("🔮 3. Council Voting (Ensemble)")
    mod_perf = st.session_state.get('module_performance')
    if mod_perf:
        st.caption("🧠 AI Thinking Process:")
        item_html = ""
        emojis = {'historian': '📜', 'technician': '🛣️', 'statistician': '🧠', 'booster': '⚡', 'expert': '🎲'}
        for k, v in mod_perf.items():
            color = "green" if v > 0 else ("red" if v < 0 else "gray")
            item_html += f'<div class="flex-item">{emojis.get(k)}<br><span style="color:{color}; font-weight:bold">{v:+d}</span></div>'
        
        st.markdown(f'<div class="flex-container">{item_html}</div>', unsafe_allow_html=True)
        st.divider()

    # --- Validation & Shoe Type Panel ---
    shoe_type = get_shoe_type(st.session_state['current_game'])
    if 'module_stats' in st.session_state:
        show_validation_panel(st.session_state['module_stats'], shoe_type)
    st.divider()
    
    if 'models' in st.session_state and len(st.session_state['current_game']) >= 5:
        try:
            prediction, score, vote_details = ensemble_predict(st.session_state['current_game'], st.session_state['models'], module_performance=mod_perf)
            if prediction is not None:
                mapping = {0: "🔵 PLAYER", 1: "🔴 BANKER", 2: "🟢 TIE"}
                result_text = mapping.get(prediction, "N/A")
                st.markdown("#### 📊 Vote Breakdown")
                v_html = '<div style="background:#1a1a2e; padding:12px; border-radius:10px; margin-bottom:10px;">'
                
                # Manual rendering to ensure no "None" appears
                emoji_map = {'historian': '📜', 'technician': '🛣️', 'statistician': '🧠', 'expert': '🎲'}
                for mod in ['historian', 'technician', 'statistician', 'expert']:
                    m = vote_details.get(mod, {})
                    if not m: continue # Skip if no vote
                    
                    vote_val = m.get('vote', 'N/A')
                    c = '#2196F3' if vote_val == 'PLAYER' else ('#f44336' if vote_val == 'BANKER' else ('#4CAF50' if vote_val == 'TIE' else '#888'))
                    emoji = m.get("emoji", emoji_map.get(mod, '❓'))
                    
                    # Special display for Expert pattern name
                    label_extra = ""
                    if mod == 'expert' and 'pattern' in m:
                        label_extra = f" ({m['pattern']})"
                        
                    v_html += f'<div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #333;"><span style="color:#fff;">{emoji} {mod.capitalize()}</span><span style="color:{c}; font-weight:bold;">{vote_val}{label_extra}</span></div>'
                v_html += '</div>'
                st.markdown(v_html, unsafe_allow_html=True)
                
                max_score = 6
                score_pct = min(score / max_score * 100, 100)
                meter_color = "#FFD700" if score_pct >= 80 else ("#2196F3" if prediction == 0 else "#f44336")
                st.markdown(f'<div style="background:#333; height:12px; border-radius:6px; margin-bottom:5px;"><div style="width:{score_pct}%; background:{meter_color}; height:100%; border-radius:6px;"></div></div>', unsafe_allow_html=True)
                st.caption(f"Confidence: {score_pct:.0f}% | Score: {score}")
                
                req = {"Low": 4, "Medium": 3, "High": 2}.get(st.session_state['risk_level'], 3)
                if score >= req:
                    st.success(f"🎯 **RECOMMENDED: {result_text}**")
                else:
                    st.info(f"⏸️ **SKIP** - Score low ({score} < {req})")
                
                st.session_state['last_prediction'] = {'vote': prediction, 'score': score}
                st.session_state['last_vote_details'] = vote_details
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("รอข้อมูล (อย่างน้อย 5 ตา)")

st.divider()
saved_col1, saved_col2 = st.columns(2)
if saved_col1.button("💾 บันทึก & จบขอน", use_container_width=True):
    if st.session_state['current_game']:
        save_game_data(st.session_state['current_game'])
        st.session_state['current_game'] = []
        st.success("บันทึกแล้ว")
        st.rerun()
if saved_col2.button("🗑️ ล้างข้อมูล", use_container_width=True):
    st.session_state['current_game'] = []
    st.rerun()