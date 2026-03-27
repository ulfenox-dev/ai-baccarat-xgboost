import streamlit as st
import pandas as pd
import os
from datetime import datetime
from logic import (
    train_ensemble_models, 
    ensemble_predict, 
    process_data_from_folder,
    get_derived_roads_features,
    get_shoe_type,
    calculate_shoe_profile,
    calculate_dog_oh_advice
)
from utils import (
    load_all_games, 
    load_custom_patterns, 
    calculate_streak, 
    calculate_tie_stats,
    mine_patterns
)
from ui_components import (
    render_big_road, 
    render_raw_data, 
    show_performance_dashboard,
    show_gallery_modal,
    show_validation_panel,
    show_financial_advice_card,
    show_profit_progress
)

# ==========================================
# ส่วนที่ 1: หน้าจอใช้งาน (GUI)
# ==========================================

st.set_page_config(
    page_title="สูตรบาคาร่า เซียนตัวจริง", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.title("AI บาคาร่า สเต็ปด๊อกโอ")
st.markdown("วิเคราะห์เจาะลึก 4 มิติ **เค้าไพ่ + สถิติ + สูตรลับ + การเดินเงิน**")

# --- เมนูด้านซ้าย: จัดการข้อมูล ---
with st.sidebar:
    st.header("📂 1. จัดการข้อมูล")
    data_path = st.text_input("โฟลเดอร์เก็บสถิติ", value="./data")
    
    # Initialize Session Learning State
    if 'session_mistakes' not in st.session_state:
        st.session_state['session_mistakes'] = []
    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None
    if 'module_performance' not in st.session_state:
        st.session_state['module_performance'] = {
            'historian': 0, 'technician': 0, 'statistician': 0, 'booster': 0, 'expert': 0, 'neural_net': 0
        }
    if 'module_stats' not in st.session_state:
        st.session_state['module_stats'] = {
            k: {'wins': 0, 'total': 0, 'score': 0} for k in ['historian', 'technician', 'statistician', 'booster', 'expert', 'neural_net']
        }
    if 'last_vote_details' not in st.session_state:
        st.session_state['last_vote_details'] = {}
    if 'ai_performance' not in st.session_state:
        st.session_state['ai_performance'] = {'wins': 0, 'losses': 0, 'history': []}
    if 'undo_stack' not in st.session_state:
        st.session_state['undo_stack'] = []
    if 'streak_count' not in st.session_state:
        st.session_state['streak_count'] = 0
    
    # Financial State
    if 'capital_thb' not in st.session_state:
        st.session_state['capital_thb'] = 1000.0
    if 'target_profit_thb' not in st.session_state:
        st.session_state['target_profit_thb'] = 500.0
    if 'current_profit_thb' not in st.session_state:
        st.session_state['current_profit_thb'] = 0.0
    if 'last_bet_amount' not in st.session_state:
        st.session_state['last_bet_amount'] = 0.0
    if 'last_big_miss' not in st.session_state:
        st.session_state['last_big_miss'] = False
    if 'min_bet_thb' not in st.session_state:
        st.session_state['min_bet_thb'] = 10.0
    def trigger_training():
        """ฟังก์ชันสำหรับคำนวณสถิติใหม่"""
        try:
            df, pattern_sequences = process_data_from_folder(data_path)
            
            if not df.empty:
                models = train_ensemble_models(df, pattern_sequences)
                custom_patterns = load_custom_patterns("./pattern")
                models['patterns'] = custom_patterns
                st.session_state['models'] = models
                st.session_state['pattern_sequences'] = pattern_sequences
                st.session_state['data_count'] = len(df)
                st.session_state['trained_with_context'] = False
                return True
            else:
                st.error("ไม่เจอไฟล์ข้อมูลในโฟลเดอร์ data")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
        return False

    if st.button("🔄 เริ่มคำนวณสถิติใหม่", use_container_width=True):
        if trigger_training():
            st.success(f"✅ เตรียมพร้อมหน่วยประมวลผลเซียน 5 สายเรียบร้อย")

    st.divider()
    
    # --- Auto Pattern Mining Logic ---
    st.subheader("⛏️ ค้นหาสูตรลับอัตโนมัติ")
    if st.button("🚀 ขุดสูตรจากสถิติเก่า", use_container_width=True):
        with st.spinner("กำลังค้นหาแพทเทิร์นที่ชนะบ่อย..."):
            discovered, message = mine_patterns(data_path, "./pattern")
            if discovered:
                st.success(f"✅ ค้นพบสูตรใหม่จำนวน {len(discovered)} รายการ เรียบร้อย")
            else:
                st.warning(message)
                
    if st.button("👀 ดูคลังสูตรที่บันทึกไว้", use_container_width=True):
        patterns = load_custom_patterns("./pattern")
        if not patterns:
             st.warning("ยังไม่มีสูตรในคลัง (กดขุดสูตรก่อน)")
        else:
             show_gallery_modal(patterns)

    st.divider()
    
    st.subheader("⚡ ระดับการคัดไพ่")
    risk_level = st.select_slider(
        "ความมั่นใจ",
        options=["ใจรึง", "ปกติ", "เน้นชัวร์"],
        value="ปกติ"
    )
    st.session_state['risk_level'] = risk_level
    
    risk_text = {"เน้นชัวร์": "🛡️ คะแนน ≥ 4 ถึงแนะนำ", "ปกติ": "⚖️ คะแนน ≥ 3 ถึงแนะนำ", "ใจรึง": "🔥 คะแนน ≥ 2 ถึงแนะนำ"}
    st.caption(risk_text.get(risk_level, ""))
        
    if st.session_state.get('session_mistakes'):
        st.caption(f"🧠 Adaptive Mode: เรียนรู้แล้ว {len(st.session_state['session_mistakes'])} จุด")

    def update_learning(outcome):
        delta = {
            'added_mistake': None, 
            'module_changes': {}, 
            'perf_change': None, 
            'prev_streak': st.session_state['streak_count'],
            'prev_big_miss': st.session_state.get('last_big_miss', False)
        }
        
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
            name_map = {'historian': 'เฝ้าขอน', 'technician': 'เค้าไพ่', 'statistician': 'สถิติ', 'booster': 'วิเคราะห์', 'expert': 'สูตรเด็ด', 'neural_net': 'โครงข่ายประสาท'}
            for eng_name, local_name in name_map.items():
                mod_data = details.get(local_name)
                if mod_data and 'vote' in mod_data:
                    vote_val = 0 if mod_data['vote'] == 'PLAYER' else (1 if mod_data['vote'] == 'BANKER' else None)
                    if vote_val is not None:
                        change = 1 if vote_val == outcome else -1
                        perf[eng_name] += change
                        delta['module_changes'][eng_name] = change
                        
            # Update detailed Stats (Win Rate per Module)
            if 'module_stats' in st.session_state:
                stats = st.session_state['module_stats']
                for eng_name, local_name in name_map.items():
                    mod_data = details.get(local_name)
                    if mod_data and 'vote' in mod_data:
                        v = mod_data['vote']
                        target = 0 if v == 'PLAYER' else (1 if v == 'BANKER' else (2 if v == 'TIE' else None))
                        
                        if target is not None:
                            stats[eng_name]['total'] += 1
                            if target == outcome:
                                stats[eng_name]['wins'] += 1
                                stats[eng_name]['score'] += 1
                            else:
                                stats[eng_name]['score'] -= 1
                            
                            delta['module_stats_update'] = True
                            delta.setdefault('module_stats_changes', {})[eng_name] = {'win': (target == outcome), 'score': (1 if target == outcome else -1)}
        
        if st.session_state.get('last_prediction') and outcome != 2:
            lp = st.session_state['last_prediction']
            is_win = (lp['vote'] == outcome)
            risk_level = st.session_state.get('risk_level', 'Medium')
            req = {"เน้นชัวร์": 4, "ปกติ": 3, "ใจรึง": 2}.get(risk_level, 3)
            stats = st.session_state['ai_performance']
            
            if lp['vote'] == 3:
                stats['history'].append(f"⚪ Skip (Meta-Labeling)")
                delta['perf_change'] = {'type': 'skip'}
            elif lp['score'] >= req:
                # Calculate Money Effect
                bet = st.session_state.get('last_bet_amount', 0)
                if is_win:
                    # Banker commission (5%)
                    profit = bet * 0.95 if lp['vote'] == 1 else bet
                    stats['wins'] += 1
                    stats['history'].append(f"✅ Win ({lp['vote']} vs {outcome}) | +{profit:.0f}.-")
                    delta['perf_change'] = {'type': 'win'}
                    st.session_state['streak_count'] += 1
                    st.session_state['current_profit_thb'] += profit
                    delta['profit_change'] = profit
                    st.session_state['last_big_miss'] = False
                else:
                    stats['losses'] += 1
                    stats['history'].append(f"❌ Loss ({lp['vote']} vs {outcome}) | -{bet:.0f}.-")
                    delta['perf_change'] = {'type': 'loss'}
                    st.session_state['streak_count'] = 0
                    st.session_state['current_profit_thb'] -= bet
                    delta['profit_change'] = -bet
                    # Check for "Big Miss" (Risk Protection)
                    if lp['score'] >= 4.5:
                        st.session_state['last_big_miss'] = True
                        delta['big_miss_set'] = True
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
            
            st.session_state['streak_count'] = delta['prev_streak']
            pc = delta.get('perf_change')
            if pc:
                if pc['type'] == 'win': 
                    st.session_state['ai_performance']['wins'] -= 1
                    st.session_state['current_profit_thb'] -= delta.get('profit_change', 0)
                elif pc['type'] == 'loss': 
                    st.session_state['ai_performance']['losses'] -= 1
                    st.session_state['current_profit_thb'] -= delta.get('profit_change', 0)
                
                # Restore previous big miss state correctly
                st.session_state['last_big_miss'] = delta.get('prev_big_miss', False)
                
                if st.session_state['ai_performance']['history']: 
                    st.session_state['ai_performance']['history'].pop()
            
            # Revert detailed stats
            for mod, changes in delta.get('module_stats_changes', {}).items():
                st.session_state['module_stats'][mod]['total'] -= 1
                if changes['win']: st.session_state['module_stats'][mod]['wins'] -= 1
                st.session_state['module_stats'][mod]['score'] -= changes['score']

    st.divider()
    st.subheader("🎨 ตั้งค่าสีตาราง")
    theme_choice = st.radio("Theme", ["ตารางสีมืด 🌙", "ตารางสีสว่าง ☀️"], index=1, horizontal=True, label_visibility="collapsed")
    st.session_state['theme'] = 'dark' if "ตารางสีมืด" in theme_choice else 'light'
    
    st.divider()
    st.header("💰 4. โปรไฟล์การลงทุน")
    # ใช้ key เพื่อให้ Streamlit จัดการ session_state อัตโนมัติ (จะช่วยให้ปุ่ม + / - ตอบสนองทันที)
    st.number_input("ทุนเริ่มต้น (บาท)", key='capital_thb', step=100.0)
    st.number_input("เป้าหมายกำไร (บาท)", key='target_profit_thb', step=100.0)
    st.number_input("เดิมพันขั้นต่ำ (บาท)", key='min_bet_thb', step=10.0)
    
    risk_choice = st.selectbox("ระดับความเสี่ยง", ["เน้นปลอดภัย (Safe)", "สายสมดุล (Balanced)", "สายซิ่ง (Aggressive)"], index=1)
    urgency_choice = st.selectbox("สถานะเงิน", ["เงินเย็น (ใจนิ่ง)", "ร้อนเงิน (ต้องชัวร์)"], index=0)
    
    st.session_state['risk_level_choice'] = risk_choice
    st.session_state['urgency_choice'] = urgency_choice

    # Manual Profit Adjustment
    with st.expander("🛠️ ปรับแก้กำไร/ขาดทุน"):
        # ใช้ value ปกติสำหรับช่องกรอกที่ไม่ได้ต้องการ sync อัตโนมัติทุกลมหายใจ (ต้องการกด Save)
        new_profit = st.number_input("กำไรปัจจุบัน (บาท)", key='temp_profit_adj', value=float(st.session_state['current_profit_thb']), step=10.0)
        if st.button("💾 บันทึกยอดเงินใหม่", use_container_width=True):
            st.session_state['current_profit_thb'] = new_profit
            st.success("อัปเดตยอดเงินเรียบร้อย")
            st.rerun()

    st.divider()
    if 'models' in st.session_state:
        st.info(f"Data: {st.session_state['data_count']} samples")
        if st.session_state.get('trained_with_context'):
            st.success("🎯 **โหมด : วิเคราะห์เจาะจงห้องล่าสุด**")
        else:
            st.caption("🌐 โหมด : วิเคราะห์รวมจากสถิติทั้งหมด")
    else:
        st.warning("⚠️ กรุณากดเทรนก่อน")

col1, col2 = st.columns([1.8, 1])

# CSS for Dashboard Style
st.markdown("""
<style>
    [data-testid="column"] { min-width: 0 !important; }
    /* Default Button Styling */
    .stButton > button { height: 62px; font-size: 18px !important; font-weight: bold !important; border-radius: 12px !important; transition: all 1.3s ease; }
    
    /* Sidebar Primary (Start/Train) -> Blue */
    [data-testid="stSidebar"] button[kind="primary"] { background-color: #2196F3 !important; color: white !important; border: none !important; }

    /* Main Area Primary (Delete/Undo) -> Red */
    [data-testid="stMain"] button[kind="primary"] { background-color: #d32f2f !important; color: white !important; border: 1px solid #ff1744 !important; }
    [data-testid="stMain"] button[kind="primary"]:hover { background-color: #ff1744 !important; box-shadow: 0 0 20px rgba(255,23,68,0.5) !important; }
    
    .expert-emoji { font-size: 24px; margin-bottom: 5px; }
    .expert-label { font-size: 13px; color: #aaa; margin-top: 2px; }
    .expert-value { font-size: 18px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

with col1:
    # --- Top Header: Profit & Performance ---
    if 'ai_performance' in st.session_state:
        # Show Profit bar first
        show_profit_progress(st.session_state['current_profit_thb'], st.session_state['target_profit_thb'])
        
        # Show Win/Loss stats bar
        stats = st.session_state['ai_performance']
        curr_game = st.session_state.get('current_game', [])
        p_pct, b_pct = 50, 50
        if curr_game:
            total_g = len(curr_game)
            p_pct = curr_game.count(0) / total_g * 100
            b_pct = curr_game.count(1) / total_g * 100
        show_performance_dashboard(stats, st.session_state['streak_count'], p_pct, b_pct)

    st.markdown("### 📊 เค้าไพ่หลัก")
    if 'current_game' not in st.session_state:
        st.session_state['current_game'] = []
    
    current_theme = st.session_state.get('theme', 'dark')
    st.markdown(render_big_road(st.session_state['current_game'], theme=current_theme), unsafe_allow_html=True)
    
    with st.expander("📝 บันทึกประวัติล่าขอน"):
        if 'ai_performance' in st.session_state:
            stats = st.session_state['ai_performance']
            for log in list(reversed(stats['history']))[:10]:
                st.caption(log)
        st.markdown(render_raw_data(st.session_state['current_game']), unsafe_allow_html=True)

with col2:
    st.markdown("### 🎮 กรอกผลไพ่")
    
    # --- Prediction & Advice (The Core) ---
    if 'models' in st.session_state and len(st.session_state['current_game']) >= 5:
        try:
            # Predict
            mod_perf = st.session_state.get('module_performance')
            prediction, score, vote_details, pat_stats, consensus = ensemble_predict(
                st.session_state['current_game'], 
                st.session_state['models'], 
                module_performance=mod_perf,
                pattern_sequences=st.session_state.get('pattern_sequences')
            )
            
            if prediction is not None:
                # Calculate Advice
                bet_amount, advice_txt = calculate_dog_oh_advice(
                    st.session_state['capital_thb'],
                    st.session_state['target_profit_thb'],
                    st.session_state['current_profit_thb'],
                    st.session_state.get('risk_level_choice', 'สายสมดุล (Balanced)'),
                    st.session_state.get('urgency_choice', 'เงินเย็น (ใจนิ่ง)'),
                    score,
                    consensus,
                    st.session_state.get('last_big_miss', False),
                    st.session_state.get('min_bet_thb', 10.0)
                )
                st.session_state['last_bet_amount'] = bet_amount
                
                # Show Advice Card (Big Text)
                show_financial_advice_card(
                    bet_amount, 
                    advice_txt, 
                    st.session_state.get('risk_level_choice', 'Balanced'),
                    st.session_state.get('urgency_choice', 'Cool')
                )
                
                # Big Action Buttons right under the advice
                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                mapping = {0: "🔵 PLAYER", 1: "🔴 BANKER", 2: "🟢 TIE", 3: "WAIT"}
                result_text = mapping.get(prediction, "รอผล")
                
                btn_c1, btn_c2 = st.columns(2)
                if btn_c1.button("🔵 PLAYER", key="p_main", use_container_width=True):
                    st.session_state['current_game'].append(0)
                    update_learning(0)
                    st.rerun()
                if btn_c2.button("🔴 BANKER", key="b_main", use_container_width=True):
                    st.session_state['current_game'].append(1)
                    update_learning(1)
                    st.rerun()
                
                btn_c3, btn_c4 = st.columns(2)
                with btn_c3:
                    if st.button("🟢 TIE", key="t_main", use_container_width=True):
                        st.session_state['current_game'].append(2)
                        update_learning(2)
                        st.rerun()
                with btn_c4:
                    if st.button("🗑️ ลบ", key="del_main", use_container_width=True, type="primary"):
                        undo_last_action()
                        st.rerun()
                
                st.session_state['last_prediction'] = {'vote': prediction, 'score': score}
                st.session_state['last_vote_details'] = vote_details
                
                # Show Stats / Experts in expanders below
                with st.expander("🧬 รายละเอียดความเห็นเซียน"):
                    v_html = '<div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:10px; margin-bottom:10px; border:1px solid #444;">'
                    for mod_name, m in vote_details.items():
                        if not m: continue
                        vote_val = m.get('vote', 'N/A')
                        c = '#2196F3' if vote_val == 'PLAYER' else ('#f44336' if vote_val == 'BANKER' else ('#4CAF50' if vote_val == 'TIE' else '#888'))
                        emoji = m.get("emoji", "❓")
                        
                        label_extra = ""
                        if mod_name == 'สูตรเด็ด' and 'pattern' in m:
                            label_extra = f" <span style='font-size:11px; opacity:0.6;'>({m['pattern']})</span>"
                            
                        v_html += f'<div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #333;"><span style="color:#fff; font-size:15px;">{emoji} {mod_name}</span><span style="color:{c}; font-weight:bold; font-size:16px;">{vote_val}{label_extra}</span></div>'
                    v_html += '</div>'
                    st.markdown(v_html, unsafe_allow_html=True)
                
                max_score = 6
                score_pct = min(score / max_score * 100, 100)
                meter_color = "#FFD700" if score_pct >= 80 else ("#2196F3" if prediction == 0 else "#f44336")
                st.markdown(f'<div style="background:#333; height:12px; border-radius:6px; margin-bottom:5px;"><div style="width:{score_pct}%; background:{meter_color}; height:100%; border-radius:6px;"></div></div>', unsafe_allow_html=True)
                st.caption(f"ความมั่นใจ {score_pct:.0f}% | คะแนนรวม {score:.1f}")
                
                req = {"เน้นชัวร์": 4, "ปกติ": 3, "ใจรึง": 2}.get(st.session_state['risk_level'], 3)
                if prediction == 3:
                    st.warning("⚠️ **ระบบป้องกันความเสี่ยง รอไพ่ (Skip)**")
                elif score >= req:
                    st.success(f"🎯 **เซียนแนะนำ {result_text}**")
                else:
                    st.info(f"⏸️ **รอจังหวะใหม่** (คะแนน {score:.1f} ยังไม่ถึง {req})")
                
                st.session_state['last_prediction'] = {'vote': prediction, 'score': score}
                st.session_state['last_vote_details'] = vote_details
                
                # --- Dog-Oh Financial Advisor Integration ---
                bet_amount, advice_txt = calculate_dog_oh_advice(
                    st.session_state['capital_thb'],
                    st.session_state['target_profit_thb'],
                    st.session_state['current_profit_thb'],
                    st.session_state.get('risk_level_choice', 'สายสมดุล (Balanced)'),
                    st.session_state.get('urgency_choice', 'เงินเย็น (ใจนิ่ง)'),
                    score,
                    # Pass consensus count if added to logic (we added it as 5th return)
                    vote_details.get('consensus', 0) # Fallback if not updated
                )
                
                # Wait, ensemble_predict return signature changed, need to handle index
                # prediction, score, vote_details, pat_stats, consensus_count = ensemble_predict(...)
                # but currently called as 4 return values in code. Let me fix the call.
            
            # (Self-correction: I need to fix the ensemble_predict call line 332 first)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("💡 รอข้อมูลไพ่ครบ 5 ตาเพื่อเริ่มวิเคราะห์ (AI จะเริ่มทำงานอัตโนมัติ)")
        
        btn_c1, btn_c2 = st.columns(2)
        if btn_c1.button("🔵 PLAYER", key="p_init", use_container_width=True):
            st.session_state['current_game'].append(0)
            update_learning(0)
            st.rerun()
        if btn_c2.button("🔴 BANKER", key="b_init", use_container_width=True):
            st.session_state['current_game'].append(1)
            update_learning(1)
            st.rerun()
        
        btn_c3, btn_c4 = st.columns(2)
        with btn_c3:
            if st.button("🟢 TIE", key="t_init", use_container_width=True):
                st.session_state['current_game'].append(2)
                update_learning(2)
                st.rerun()
        with btn_c4:
            if st.button("🗑️ ลบ", key="del_init", use_container_width=True, type="primary"):
                undo_last_action()
                st.rerun()

st.divider()

# ฟังก์ชันแสดง Modal ยืนยันการล้างกระดาน
@st.dialog("⚠️ ยืนยันการล้างกระดาน")
def show_clear_board_dialog():
    st.markdown("คุณแน่ใจหรือไม่ว่าต้องการล้างข้อมูลกระดานทั้งหมด?")
    col_y, col_n = st.columns(2)
    if col_y.button("✅ ยืนยันล้างข้อมูล", type="primary", use_container_width=True):
        st.session_state['current_game'] = []
        st.rerun()
    if col_n.button("❌ ยกเลิก", use_container_width=True):
        st.rerun()

if st.button("🗑️ ล้างกระดาน", use_container_width=True):
    show_clear_board_dialog()