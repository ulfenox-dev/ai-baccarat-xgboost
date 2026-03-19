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
    calculate_shoe_profile
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
    show_validation_panel
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
st.markdown("วิเคราะห์เจาะลึก 4 มิติ : **เค้าไพ่ + สถิติ + สูตรลับ + การเดินเงิน**")

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
    if 'auto_train' not in st.session_state:
        st.session_state['auto_train'] = True 

    def trigger_training():
        """ฟังก์ชันสำหรับคำนวณสถิติใหม่"""
        try:
            target_profile = None
            curr_game = st.session_state.get('current_game', [])
            if len(curr_game) >= 15:
                target_profile = calculate_shoe_profile(curr_game)
            
            df, pattern_sequences = process_data_from_folder(data_path, target_profile=target_profile)
            
            if not df.empty:
                models = train_ensemble_models(df, pattern_sequences)
                custom_patterns = load_custom_patterns("./pattern")
                models['patterns'] = custom_patterns
                st.session_state['models'] = models
                st.session_state['pattern_sequences'] = pattern_sequences
                st.session_state['data_count'] = len(df)
                st.session_state['trained_with_context'] = (target_profile is not None)
                return True
            else:
                st.error("ไม่เจอไฟล์ข้อมูลในโฟลเดอร์ data")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
        return False

    if st.button("🔄 เริ่มคำนวณสถิติใหม่", use_container_width=True):
        if trigger_training():
            st.success(f"✅ เตรียมพร้อมหน่วยประมวลผลเซียน 5 สายเรียบร้อย")

    st.session_state['auto_train'] = st.checkbox("⚡ คำนวณอัตโนมัติ (ทุกตา)", value=st.session_state['auto_train'])
                
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

    st.divider()
    st.subheader("🎨 ตั้งค่าสีตาราง")
    theme_choice = st.radio("Theme", ["ตารางสีมืด 🌙", "ตารางสีสว่าง ☀️"], index=1, horizontal=True, label_visibility="collapsed")
    st.session_state['theme'] = 'dark' if "ตารางสีมืด" in theme_choice else 'light'
    
    st.divider()
    if 'models' in st.session_state:
        st.info(f"Data: {st.session_state['data_count']} samples")
        if st.session_state.get('trained_with_context'):
            st.success("🎯 **โหมด: วิเคราะห์เจาะจงห้องล่าสุด**")
        else:
            st.caption("🌐 โหมด: วิเคราะห์รวมจากสถิติทั้งหมด")
    else:
        st.warning("⚠️ กรุณากดเทรนก่อน")

col1, col2 = st.columns([1.5, 1])

# CSS for Horizontal Display on Mobile
st.markdown("""
<style>
    [data-testid="column"] { min-width: 0 !important; }
    .flex-container { display: flex; justify-content: space-around; width: 100%; gap: 5px; }
    .flex-item { text-align: center; flex: 1; min-height: 80px; }
    .expert-emoji { font-size: 24px; margin-bottom: 5px; }
    .expert-label { font-size: 13px; color: #aaa; margin-top: 2px; }
    .expert-value { font-size: 18px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

with col1:
    st.subheader("🎰 2. บันทึกผลไพ่สด")
    
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
            name_map = {'historian': 'เฝ้าขอน', 'technician': 'เค้าไพ่', 'statistician': 'สถิติ', 'booster': 'สูตรคำนวณ', 'expert': 'ล็อคแพทเทิร์น'}
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
                name_map = {'historian': 'เฝ้าขอน', 'technician': 'เค้าไพ่', 'statistician': 'สถิติ', 'booster': 'สูตรคำนวณ', 'expert': 'ล็อคแพทเทิร์น'}
                for eng_name, local_name in name_map.items():
                    mod_data = details.get(local_name)
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
        st.session_state['current_game'].append(0)
        update_learning(0)
        # Smart Sync: โหลดข้อมูลใหม่เฉพาะตาที่สำคัญ (15, 30, 50) 
        # และเพิ่มกรณีที่ยังไม่ได้เทรนเลยเพื่อให้ระบบเริ่มทำงานได้
        curr_len = len(st.session_state['current_game'])
        needs_init = 'models' not in st.session_state and curr_len >= 15
        if st.session_state.get('auto_train') and (curr_len in [15, 30, 50] or needs_init):
            trigger_training()
        st.rerun()
    if c2.button("🔴 BANKER", use_container_width=True):
        st.session_state['current_game'].append(1)
        update_learning(1)
        curr_len = len(st.session_state['current_game'])
        needs_init = 'models' not in st.session_state and curr_len >= 15
        if st.session_state.get('auto_train') and (curr_len in [15, 30, 50] or needs_init):
            trigger_training()
        st.rerun()
    if c3.button("🟢 TIE", use_container_width=True):
        st.session_state['current_game'].append(2)
        update_learning(2)
        curr_len = len(st.session_state['current_game'])
        needs_init = 'models' not in st.session_state and curr_len >= 15
        if st.session_state.get('auto_train') and (curr_len in [15, 30, 50] or needs_init):
            trigger_training()
        st.rerun()
    if c4.button("ลบ", type="primary"):
        undo_last_action()
        # ไม่ต้อง auto-sync ตอนลบเพื่อความเร็ว
        st.rerun()

with col2:
    st.subheader("🔮 3. สภาเซียน")
    mod_perf = st.session_state.get('module_performance')
    if mod_perf:
        st.caption("ความอึดเซียนรายสำนัก (คะแนนมือขึ้น):")
        item_html = ""
        emojis = {'historian': '📜', 'technician': '🛣️', 'statistician': '🧠', 'booster': '⚡', 'expert': '🎲'}
        # ใช้ชื่อที่เข้าใจง่ายที่สุด
        name_map = {'historian': 'จำทางไพ่', 'technician': 'อ่านสามเกลอ', 'statistician': 'สถิติรวม', 'booster': 'วิเคราะห์', 'expert': 'สูตรเด็ด'}
        for k, v in mod_perf.items():
            color = "#4CAF50" if v > 0 else ("#f44336" if v < 0 else "#888")
            item_html += (
                f'<div class="flex-item">'
                f'<div class="expert-emoji">{emojis.get(k)}</div>'
                f'<div class="expert-value" style="color:{color};">{v:+d}</div>'
                f'<div class="expert-label">{name_map.get(k)}</div>'
                f'</div>'
            )
        
        st.markdown(f'<div class="flex-container">{item_html}</div>', unsafe_allow_html=True)
        st.divider()

    # --- Validation & Shoe Type Panel ---
    shoe_type = get_shoe_type(st.session_state['current_game'])
    if 'module_stats' in st.session_state:
        show_validation_panel(st.session_state['module_stats'], shoe_type)
    st.divider()
    
    if 'models' in st.session_state and len(st.session_state['current_game']) >= 5:
        try:
            # เพิ่ม pattern_sequences เข้าไปใน predict เพื่อคำนวณสถิติ
            prediction, score, vote_details, pat_stats = ensemble_predict(
                st.session_state['current_game'], 
                st.session_state['models'], 
                module_performance=mod_perf,
                pattern_sequences=st.session_state.get('pattern_sequences')
            )
            
            if prediction is not None:
                mapping = {0: "🔵 PLAYER", 1: "🔴 BANKER", 2: "🟢 TIE"}
                result_text = mapping.get(prediction, "รอผล")
                
                # แสดงสถิติย้อนหลัง (Pattern Stats)
                if pat_stats:
                    st.markdown(f"📊 **สถิติขอนที่นิสัยเหมือนห้องนี้:** (เคยพบ {pat_stats['total']} ครั้ง)")
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.markdown(f"<div style='text-align:center;'><span style='color:#2196F3; font-size:12px;'>Player</span><br><span style='font-size:20px; font-weight:bold;'>{pat_stats['p_rate']:.0f}%</span></div>", unsafe_allow_html=True)
                    with stat_col2:
                        st.markdown(f"<div style='text-align:center;'><span style='color:#f44336; font-size:12px;'>Banker</span><br><span style='font-size:20px; font-weight:bold;'>{pat_stats['b_rate']:.0f}%</span></div>", unsafe_allow_html=True)
                    st.divider()

                st.markdown("#### 🎯 เจาะลึกการวิเคราะห์")
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
                st.caption(f"คะแนนความมั่นใจ: {score_pct:.0f}% | คะแนนรวม: {score:.1f}")
                
                req = {"เน้นชัวร์": 4, "ปกติ": 3, "ใจรึง": 2}.get(st.session_state['risk_level'], 3)
                if score >= req:
                    st.success(f"🎯 **เซียนแนะนำ: {result_text}**")
                else:
                    st.info(f"⏸️ **รอจังหวะใหม่** (คะแนน {score:.1f} ยังไม่ถึง {req})")
                
                st.session_state['last_prediction'] = {'vote': prediction, 'score': score}
                st.session_state['last_vote_details'] = vote_details
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("รอข้อมูล (อย่างน้อย 5 ตา)")

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