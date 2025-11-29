import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import json

# ---------------------------------------------------------
# 0. 파이어베이스(DB) 설정 (서버 저장용)
# ---------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, firestore

def get_db():
    if not firebase_admin._apps:
        try:
            if 'firebase_key' in st.secrets:
                secret_val = st.secrets['firebase_key']
                if isinstance(secret_val, str):
                    key_dict = json.loads(secret_val)
                else:
                    key_dict = dict(secret_val)
                
                if 'private_key' in key_dict:
                    key_dict['private_key'] = key_dict['private_key'].replace('\\n', '\n')

                cred = credentials.Certificate(key_dict)
                firebase_admin.initialize_app(cred)
            else:
                return None
        except Exception as e:
            st.warning(f"DB 연결 실패: {e}")
            return None
    return firestore.client()

# ---------------------------------------------------------
# 1. 설정 및 종목명 매핑 데이터
# ---------------------------------------------------------
st.set_page_config(page_title="AI 주식 스캐너 Pro", page_icon="📈", layout="wide")

# 세션 상태 초기화
if 'scan_result_df' not in st.session_state:
    st.session_state['scan_result_df'] = None

# 종목 코드와 한글명 매핑
TICKER_MAP = {
    "INTC": "인텔 (Intel)", "005290.KS": "동진쎄미켐", "SOXL": "반도체 3X(Bull)", 
    "316140.KS": "우리금융지주", "WDC": "웨스턴디지털", "NFLX": "넷플릭스", 
    "000990.KS": "DB하이텍", "KLAC": "KLA", "009540.KS": "HD한국조선해양", 
    "006360.KS": "GS건설", "024110.KS": "기업은행", "042660.KS": "대우조선해양(한화오션)", 
    "105560.KS": "KB금융", "BAC": "뱅크오브아메리카", "NEM": "뉴몬트", 
    "FCX": "프리포트맥모란", "272210.KS": "한화시스템", "240810.KS": "크래프톤", 
    "005930.KS": "삼성전자", "010140.KS": "삼성중공업", "006400.KS": "삼성SDI", 
    "267250.KS": "HD현대", "028260.KS": "삼성물산", "SLV": "은(Silver) ETF", 
    "079550.KS": "LIG넥스원", "039030.KS": "이오테크닉스", "C": "씨티그룹", 
    "009830.KS": "한화솔루션", "LLY": "일라이릴리", "128940.KS": "한미약품", 
    "WFC": "웰스파고", "012450.KS": "한화에어로스페이스", "ASML": "ASML", 
    "NVDA": "엔비디아", "GE": "GE에어로스페이스", "V": "비자(Visa)", 
    "XLE": "에너지 ETF", "005935.KS": "삼성전자우", "041510.KS": "에스엠", 
    "BA": "보잉", "000660.KS": "SK하이닉스", "000810.KS": "삼성화재", 
    "000250.KS": "삼천당제약", "TXN": "텍사스인스트루먼트", "122990.KS": "와이지엔터", 
    "GM": "제너럴모터스", "302440.KS": "SK바이오사이언스", "F": "포드", 
    "DELL": "델 테크놀로지스", "JNJ": "존슨앤존슨", "263750.KS": "펄어비스", 
    "012330.KS": "현대모비스", "QCOM": "퀄컴", "XOM": "엑슨모빌", 
    "AVGO": "브로드컴", "OXY": "옥시덴탈", "SLB": "슐럼버거", 
    "086790.KS": "하나금융지주", "TQQQ": "나스닥 3X(Bull)", "UPRO": "S&P500 3X(Bull)", 
    "FNGU": "FANG+ 3X(Bull)", "BULZ": "기술주 3X(Bull)", "TMF": "채권 3X(Bull)", 
    "TSLA": "테슬라", "AMD": "AMD", "BITX": "비트코인 2X", "TSLL": "테슬라 1.5X"
}

# 검색용 리스트
SEARCH_LIST = [f"{name} ({code})" for code, name in TICKER_MAP.items()]
SEARCH_MAP = {f"{name} ({code})": code for code, name in TICKER_MAP.items()}
USER_WATCHLIST = list(TICKER_MAP.keys())

# ---------------------------------------------------------
# 2. 데이터 로드 및 지표 계산
# ---------------------------------------------------------
# 중요: 가격 괴리를 줄이기 위해 캐시 유지 시간을 10초로 대폭 단축
@st.cache_data(ttl=10)
def get_bulk_data(tickers_list):
    """데이터 다운로드 (2년치) - 실시간성 강화"""
    # prepost=True: 장전/장후 거래 데이터 포함 (최신가 반영 확률 높임)
    data = yf.download(tickers_list, period="2y", group_by='ticker', threads=True, prepost=True)
    return data

def calculate_indicators(df):
    if len(df) < 60: return None
    df = df.copy()
    # ffill로 결측치 채우되, 마지막 데이터가 NaN이면 삭제하지 않고 유지
    df['Close'] = df['Close'].ffill()

    # 이평선
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna()

def calculate_net_profit(ticker, avg_price, current_price):
    """
    토스 증권 수수료 반영 수익률 계산
    """
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
    
    if is_kr:
        sell_fee_rate = 0.00015 + 0.0018  # 0.195%
    else:
        sell_fee_rate = 0.001  # 0.1%
        
    net_sell_price = current_price * (1 - sell_fee_rate)
    profit_amt = net_sell_price - avg_price
    profit_pct = (profit_amt / avg_price) * 100
    
    currency = "₩" if is_kr else "$"
    
    return profit_pct, profit_amt, currency

# ---------------------------------------------------------
# 3. 고도화된 전략 분석 & 점수화 (Scoring)
# ---------------------------------------------------------
def analyze_advanced_strategy(df):
    if df is None or df.empty: return "분석 불가", "gray", "데이터 부족", 0
    
    curr = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    sig = df['Signal_Line'].iloc[-1]
    prev_macd = df['MACD'].iloc[-2]
    prev_sig = df['Signal_Line'].iloc[-2]

    # 상태 변수
    trend_up = curr > ma60
    above_ma20 = curr > ma20
    golden_cross = (macd > sig) and (prev_macd <= prev_sig)
    dead_cross = (macd < sig) and (prev_macd >= prev_sig)
    oversold = rsi < 35
    overbought = rsi > 70
    dip_buy = trend_up and (curr <= ma20 * 1.02) and (curr >= ma20 * 0.98) 

    # --- 점수 계산 로직 (0 ~ 100점) ---
    score = 50 # 기본 점수
    
    if trend_up: score += 20
    else: score -= 20
    
    if above_ma20: score += 10
    else: score -= 10
    
    if golden_cross: score += 15
    if dead_cross: score -= 15
    
    if dip_buy: score += 15 # 눌림목 가산점
    
    if oversold: score += 10 # 과매도 반등 기대
    if overbought: score -= 10 # 과열 주의
    
    # 점수 보정 (0~100)
    score = max(0, min(100, score))

    # --- 등급 및 코멘트 ---
    reasons = []
    category = "중립/관망 (Hold)"
    color_name = "gray" # Streamlit color name

    # 점수 기반 등급 분류 (우선순위)
    if score >= 85:
        category = "🚀 강력 매수 (Strong Buy)"
        color_name = "green"
        if dip_buy: reasons.append("상승 추세 속 '눌림목' 완벽한 기회")
        if golden_cross: reasons.append("MACD 골든크로스 발생")
    elif score >= 65:
        category = "📈 매수 (Buy)"
        color_name = "blue"
        if trend_up: reasons.append("상승 추세 유지 중")
        if oversold: reasons.append(f"과매도(RSI {rsi:.0f}) 저점 매수 기회")
    elif score <= 20:
        category = "💥 강력 매도 (Strong Sell)"
        color_name = "red" # orange or red
        reasons.append("하락 추세 가속화, 위험")
    elif score <= 40:
        category = "📉 매도 (Sell)"
        color_name = "red"
        if dead_cross: reasons.append("데드크로스 발생 (하락 전환)")
        if overbought: reasons.append("과열권 차익실현 권고")
    else:
        category = "👀 관망 (Neutral)"
        color_name = "gray"
        if not trend_up and above_ma20: reasons.append("단기 반등 중이나 추세 불안")
        else: reasons.append("뚜렷한 방향성 없음")

    if not reasons:
        if rsi > 50: reasons.append("추세 지속 여부 관찰")
        else: reasons.append("모멘텀 부족")

    return category, color_name, ", ".join(reasons), score

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)"])

# === TAB 1: 스캐너 ===
with tab1:
    st.markdown("### 📋 시장 전체 스캔 및 AI 점수")
    st.caption("AI 점수가 높은 순서대로 '구매 우선순위'를 보여줍니다.")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 분석 새로고침", type="primary"):
            st.session_state['scan_result_df'] = None 
            st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 리스트 정밀 분석 시작"):
            with st.spinner('AI가 전 종목을 채점 중입니다... (15~20초 소요)'):
                raw_data = get_bulk_data(USER_WATCHLIST)
                scan_results = []
                progress_bar = st.progress(0)
                
                for i, ticker_code in enumerate(USER_WATCHLIST):
                    try:
                        if isinstance(raw_data.columns, pd.MultiIndex):
                            try: df_ticker = raw_data.xs(ticker_code, axis=1, level=1)
                            except: df_ticker = raw_data[ticker_code]
                        else: df_ticker = raw_data
                        
                        df_ticker = df_ticker.dropna(how='all')
                        if df_ticker.empty: continue
                        
                        df_indi = calculate_indicators(df_ticker)
                        if df_indi is None: continue

                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)
                        curr_price = df_indi['Close'].iloc[-1]
                        rsi_val = df_indi['RSI'].iloc[-1]
                        name = TICKER_MAP.get(ticker_code, ticker_code)
                        
                        # 화폐 단위 및 티커 표시 처리
                        is_kr = ticker_code.endswith(".KS") or ticker_code.endswith(".KQ")
                        currency_symbol = "₩" if is_kr else "$"
                        
                        # 이름에 코드 추가 (예: 나스닥 3X(Bull) (TQQQ))
                        display_name = f"{name} ({ticker_code})"
                        
                        # 가격 포맷팅 (문자열로 변환하여 단위 표시)
                        if is_kr:
                            fmt_price = f"{currency_symbol}{curr_price:,.0f}"
                        else:
                            fmt_price = f"{currency_symbol}{curr_price:,.2f}"

                        scan_results.append({
                            "종목명": display_name,
                            "점수": score,
                            "현재가": fmt_price,
                            "RSI": rsi_val,
                            "AI 등급": cat,
                            "핵심 요약": reasoning
                        })
                    except: continue
                    progress_bar.progress((i + 1) / len(USER_WATCHLIST))
                
                if scan_results:
                    df_res = pd.DataFrame(scan_results)
                    # 점수 높은 순 정렬
                    df_res = df_res.sort_values('점수', ascending=False)
                    st.session_state['scan_result_df'] = df_res
                    st.success("분석 완료!")
                    st.rerun()
                else:
                    st.error("데이터를 가져오지 못했습니다.")
    
    if st.session_state['scan_result_df'] is not None:
        st.dataframe(
            st.session_state['scan_result_df'],
            use_container_width=True,
            height=700,
            column_config={
                "종목명": st.column_config.TextColumn("종목명 (코드)", width="medium"),
                "점수": st.column_config.ProgressColumn(
                    "AI 구매 매력도", format="%d점", min_value=0, max_value=100
                ),
                "현재가": st.column_config.TextColumn("현재가"), # TextColumn으로 변경하여 화폐 단위 표시
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "AI 등급": st.column_config.TextColumn("AI 판단"),
                "핵심 요약": st.column_config.TextColumn("분석 내용", width="large"),
            },
            hide_index=True
        )

# === TAB 2: 포트폴리오 ===
with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오")
    
    db = get_db()
    if not db:
        st.warning("⚠️ Firebase 설정이 필요합니다.")
    else:
        col_u1, col_u2 = st.columns([1, 3])
        with col_u1:
            # 닉네임 기본값 "장동진" 적용
            user_id = st.text_input("닉네임 입력", value="장동진")
        
        doc_ref = db.collection('portfolios').document(user_id)
        
        try:
            doc = doc_ref.get()
            pf_data = doc.to_dict().get('stocks', []) if doc.exists else []
        except: pf_data = []

        # === 종목 추가 UI ===
        with st.container():
            st.markdown("#### ➕ 종목 추가")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                selected_item = st.selectbox(
                    "종목 검색 (이름 또는 코드 입력)", 
                    options=["선택하세요"] + SEARCH_LIST,
                    index=0
                )
            with c2:
                input_price = st.number_input("내 평단가", min_value=0.0, format="%.2f")
            with c3:
                st.write("")
                st.write("")
                if st.button("추가하기", type="primary"):
                    if selected_item != "선택하세요":
                        target_code = SEARCH_MAP[selected_item]
                        new_pf_data = [p for p in pf_data if p['ticker'] != target_code]
                        new_pf_data.append({"ticker": target_code, "price": input_price})
                        doc_ref.set({'stocks': new_pf_data})
                        st.success(f"{selected_item} 추가 완료!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.warning("종목을 선택해주세요.")

        st.divider()

        if pf_data:
            st.subheader(f"{user_id}님의 보유 종목 진단")
            st.caption("※ 가격은 실시간 업데이트되지만, 무료 데이터 특성상 15~20분 지연될 수 있습니다.")
            
            my_tickers = [p['ticker'] for p in pf_data]
            with st.spinner("최신 시세 조회 중..."):
                # 캐시 TTL 적용된 함수 호출
                my_raw = get_bulk_data(my_tickers)
            
            # 보유 종목도 AI 점수 순으로 정렬해서 보여주기 위해 리스트 생성
            display_list = []

            for item in pf_data:
                tk = item['ticker']
                avg = item['price']
                name = TICKER_MAP.get(tk, tk)
                
                try:
                    if isinstance(my_raw.columns, pd.MultiIndex):
                        try: df_tk = my_raw.xs(tk, axis=1, level=1)
                        except: df_tk = my_raw[tk]
                    else: df_tk = my_raw

                    df_tk = df_tk.dropna(how='all')
                    df_indi = calculate_indicators(df_tk)
                    
                    if df_indi is None: continue

                    cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)
                    curr = df_indi['Close'].iloc[-1]
                    
                    profit_pct, profit_amt, currency = calculate_net_profit(tk, avg, curr)
                    
                    display_list.append({
                        "name": name, "tk": tk, "avg": avg, "curr": curr,
                        "cat": cat, "col_name": col_name, "reasoning": reasoning,
                        "profit_pct": profit_pct, "profit_amt": profit_amt,
                        "currency": currency, "score": score
                    })
                except: pass
            
            # 점수 높은 순(혹은 다른 기준)으로 정렬 가능, 여기선 입력 순서 유지하되 카드형태 출력
            # display_list.sort(key=lambda x: x['score'], reverse=True) # 점수순 정렬을 원하면 주석 해제

            for item in display_list:
                # 카드 UI
                with st.container():
                    c1, c2, c3 = st.columns([1.5, 1.5, 4])
                    
                    with c1:
                        st.markdown(f"### {item['name']}")
                        st.caption(f"{item['tk']}")
                    
                    with c2:
                        # 통화 기호 일치시키기 (미국 주식도 $ 표시)
                        cur_sym = item['currency']
                        
                        # 중요: 달러 기호($)가 Markdown LaTeX로 인식되어 폰트가 깨지는 문제 해결
                        # 화면 표시용 기호에는 Escape 문자(\)를 추가하여 일반 텍스트로 렌더링되게 함
                        display_sym = cur_sym.replace("$", "\$") 
                        
                        if cur_sym == "₩":
                            fmt_curr = f"{item['curr']:,.0f}"
                            fmt_avg = f"{item['avg']:,.0f}"
                            fmt_diff = f"{item['profit_amt']:,.0f}"
                        else:
                            fmt_curr = f"{item['curr']:,.2f}"
                            fmt_avg = f"{item['avg']:,.2f}"
                            fmt_diff = f"{item['profit_amt']:,.2f}"
                            
                        st.metric(
                            "순수익률 (수수료 제)", 
                            f"{item['profit_pct']:.2f}%", 
                            delta=f"{display_sym}{fmt_diff}"
                        )
                        # 평단/현재가 폰트 및 포맷 통일 (Escape된 심볼 사용)
                        st.markdown(f"<small style='color: gray'>평단: {display_sym}{fmt_avg} / 현재: {display_sym}{fmt_curr}</small>", unsafe_allow_html=True)
                        
                    with c3:
                        # 글씨 깨짐 수정: Streamlit 공식 컬러 사용 (:green, :blue 등)
                        st.markdown(f"**AI 점수: {item['score']}점**")
                        st.markdown(f"**판단:** :{item['col_name']}[{item['cat']}]")
                        st.info(f"💡 {item['reasoning']}")
                    
                    st.divider()

            if st.button("🗑️ 포트폴리오 전체 삭제"):
                doc_ref.delete()
                st.rerun()
        else:
            st.info("위 검색창에서 종목을 검색하여 추가해주세요.")
