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

# 세션 상태 초기화 (탭 이동 시 데이터 유지용)
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

# 검색용 리스트 생성 (예: "삼성전자 (005930.KS)")
SEARCH_LIST = [f"{name} ({code})" for code, name in TICKER_MAP.items()]
SEARCH_MAP = {f"{name} ({code})": code for code, name in TICKER_MAP.items()}

USER_WATCHLIST = list(TICKER_MAP.keys())

# ---------------------------------------------------------
# 2. 데이터 로드 및 지표 계산
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_bulk_data(tickers_list):
    """데이터 다운로드 (2년치)"""
    data = yf.download(tickers_list, period="2y", group_by='ticker', threads=True)
    return data

def calculate_indicators(df):
    if len(df) < 60: return None
    df = df.copy()
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
    - 국내(KR): 매도 수수료 0.015% + 증권거래세 0.18% (총 약 0.195%)
    - 해외(US): 매도 수수료 0.1%
    """
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
    
    if is_kr:
        sell_fee_rate = 0.00015 + 0.0018  # 0.195%
    else:
        sell_fee_rate = 0.001  # 0.1%
        
    # 매수 수수료는 평단가에 이미 포함되어 있다고 가정 (보통 앱이 그렇게 보여줌)
    # 매도 시 수수료 차감 후 금액
    net_sell_price = current_price * (1 - sell_fee_rate)
    
    profit_amt = net_sell_price - avg_price
    profit_pct = (profit_amt / avg_price) * 100
    
    currency = "₩" if is_kr else "$"
    
    return profit_pct, profit_amt, currency

# ---------------------------------------------------------
# 3. 고도화된 전략 분석 (다양한 추천 & 이유)
# ---------------------------------------------------------
def analyze_advanced_strategy(df):
    if df is None or df.empty: return "분석 불가", "gray", "데이터 부족"
    
    curr = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    sig = df['Signal_Line'].iloc[-1]
    prev_macd = df['MACD'].iloc[-2]
    prev_sig = df['Signal_Line'].iloc[-2]

    # 분석 변수
    trend_up = curr > ma60
    above_ma20 = curr > ma20
    golden_cross = (macd > sig) and (prev_macd <= prev_sig)
    dead_cross = (macd < sig) and (prev_macd >= prev_sig)
    oversold = rsi < 35
    overbought = rsi > 70
    dip_buy = trend_up and (curr <= ma20 * 1.02) and (curr >= ma20 * 0.98) 

    reasons = []
    
    category = "중립/관망 (Hold)"
    color = "gray" 

    if trend_up and (dip_buy or (golden_cross and not overbought)):
        category = "🚀 강력 매수 (Strong Buy)"
        color = "#00C853" 
        if dip_buy: reasons.append("상승 추세 속 '눌림목' 지지")
        if golden_cross: reasons.append("MACD 골든크로스")

    elif (trend_up and above_ma20) or (oversold and curr > ma20 * 0.95):
        category = "📈 매수 (Buy)"
        color = "#2962FF" 
        if trend_up and above_ma20: reasons.append("정배열 상승 지속")
        if oversold: reasons.append(f"과매도(RSI {rsi:.0f}), 반등 기대")

    elif (not trend_up and not above_ma20) or (overbought and dead_cross):
        category = "📉 매도 (Sell)"
        color = "#FF5252" 
        if not trend_up: reasons.append("추세 이탈")
        if overbought: reasons.append(f"과열(RSI {rsi:.0f})")
        if dead_cross: reasons.append("데드크로스")

    elif not trend_up and curr < ma20 and dead_cross:
        category = "💥 강력 매도 (Strong Sell)"
        color = "#D50000" 
        reasons.append("하락 가속화")

    else:
        category = "👀 관망 (Neutral)"
        color = "#757575" 
        if overbought: reasons.append("과열권 관망")
        elif not trend_up and above_ma20: reasons.append("단기 반등 중")
        else: reasons.append("횡보세")

    if not reasons:
        if rsi > 50: reasons.append("추세 관찰 필요")
        else: reasons.append("모멘텀 부족")

    return category, color, ", ".join(reasons)

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)"])

# === TAB 1: 스캐너 ===
with tab1:
    st.markdown("### 📋 시장 전체 스캔 및 AI 분석")
    st.caption("관심 종목 전체를 분석합니다. 탭을 이동해도 결과는 유지됩니다.")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        # 새로고침 버튼 (데이터를 강제로 다시 불러옴)
        if st.button("🔄 분석 새로고침", type="primary"):
            st.session_state['scan_result_df'] = None # 초기화
            st.rerun()

    # 데이터가 없으면 실행, 있으면 저장된 것 보여줌
    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 리스트 정밀 분석 시작"):
            with st.spinner('AI가 차트 패턴과 보조지표를 분석 중입니다...'):
                raw_data = get_bulk_data(USER_WATCHLIST)
                scan_results = []
                progress_bar = st.progress(0)
                
                for i, ticker_code in enumerate(USER_WATCHLIST):
                    try:
                        if isinstance(raw_data.columns, pd.MultiIndex):
                            try: df_ticker = raw_data.xs(ticker_code, axis=1, level=1)
                            except: df_ticker = raw_data[ticker_code]
                        else:
                            df_ticker = raw_data
                        
                        df_ticker = df_ticker.dropna(how='all')
                        if df_ticker.empty: continue
                        
                        df_indi = calculate_indicators(df_ticker)
                        if df_indi is None: continue

                        cat, color_code, reasoning = analyze_advanced_strategy(df_indi)
                        curr_price = df_indi['Close'].iloc[-1]
                        rsi_val = df_indi['RSI'].iloc[-1]
                        name = TICKER_MAP.get(ticker_code, ticker_code)
                        
                        scan_results.append({
                            "종목명": name,
                            "코드": ticker_code,
                            "현재가": curr_price,
                            "RSI": rsi_val,
                            "AI 추천": cat,
                            "분석 요약": reasoning,
                        })
                    except: continue
                    progress_bar.progress((i + 1) / len(USER_WATCHLIST))
                
                if scan_results:
                    df_res = pd.DataFrame(scan_results)
                    rank_map = {"🚀": 0, "📈": 1, "👀": 2, "📉": 3, "💥": 4}
                    df_res['rank'] = df_res['AI 추천'].apply(lambda x: rank_map.get(x[0], 5))
                    df_res = df_res.sort_values('rank')
                    st.session_state['scan_result_df'] = df_res # 세션에 저장
                    st.success("분석 완료!")
                    st.rerun() # 저장된 데이터 표시를 위해 리런
                else:
                    st.error("데이터를 가져오지 못했습니다.")
    
    # 세션에 저장된 데이터 표시
    if st.session_state['scan_result_df'] is not None:
        st.dataframe(
            st.session_state['scan_result_df'][['종목명', '현재가', 'RSI', 'AI 추천', '분석 요약']],
            use_container_width=True,
            height=700,
            column_config={
                "종목명": st.column_config.TextColumn("종목명"),
                "현재가": st.column_config.NumberColumn("현재가", format="%.0f"),
                "RSI": st.column_config.ProgressColumn(
                    "RSI (강도)", format="%.1f", min_value=0, max_value=100,
                ),
                "AI 추천": st.column_config.TextColumn("AI 종합 의견", width="medium"),
                "분석 요약": st.column_config.TextColumn("상세 분석 사유", width="large"),
            },
            hide_index=True
        )

# === TAB 2: 포트폴리오 ===
with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오")
    st.caption("토스 증권 수수료(국내 0.195%, 해외 0.1%)가 반영된 실질 수익률입니다.")
    
    db = get_db()
    if not db:
        st.warning("⚠️ Firebase 설정이 필요합니다.")
    else:
        col_u1, col_u2 = st.columns([1, 3])
        with col_u1:
            # 닉네임 입력 시 세션 유지 또는 기본값 처리
            user_id = st.text_input("닉네임 입력", value="my_portfolio")
        
        doc_ref = db.collection('portfolios').document(user_id)
        
        # 불러오기
        try:
            doc = doc_ref.get()
            pf_data = doc.to_dict().get('stocks', []) if doc.exists else []
        except: pf_data = []

        # === 종목 추가 UI (검색 기능 강화) ===
        with st.container():
            st.markdown("#### ➕ 종목 추가")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                # 검색 가능한 Selectbox (Autocomplete)
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
                        # "삼성전자 (005930.KS)" -> "005930.KS" 추출
                        target_code = SEARCH_MAP[selected_item]
                        
                        # 기존 리스트에서 동일 종목 제거 (업데이트)
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
            
            my_tickers = [p['ticker'] for p in pf_data]
            with st.spinner("최신 시세 조회 중..."):
                my_raw = get_bulk_data(my_tickers)
            
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
                    
                    if df_indi is None:
                        st.warning(f"{name}: 데이터 부족")
                        continue

                    cat, color_hex, reasoning = analyze_advanced_strategy(df_indi)
                    curr = df_indi['Close'].iloc[-1]
                    
                    # 수수료 반영 수익률 계산
                    profit_pct, profit_amt, currency = calculate_net_profit(tk, avg, curr)
                    
                    # 수익률 색상
                    pct_color = "red" if profit_pct < 0 else "green"
                    
                    # 카드 UI
                    with st.container():
                        c1, c2, c3 = st.columns([1.5, 1.5, 4])
                        
                        with c1:
                            st.markdown(f"### {name}")
                            st.caption(f"{tk}")
                        
                        with c2:
                            # 통화 기호와 포맷 자동 적용
                            if currency == "₩":
                                fmt_curr = f"{curr:,.0f}"
                                fmt_avg = f"{avg:,.0f}"
                                fmt_diff = f"{profit_amt:,.0f}"
                            else:
                                fmt_curr = f"{curr:,.2f}"
                                fmt_avg = f"{avg:,.2f}"
                                fmt_diff = f"{profit_amt:,.2f}"
                                
                            st.metric("순수익률 (수수료 제)", f"{profit_pct:.2f}%", delta=f"{currency}{fmt_diff}")
                            st.caption(f"평단: {currency}{fmt_avg} / 현재: {currency}{fmt_curr}")
                            
                        with c3:
                            st.markdown(f"**AI 판단:** :{color_hex}[{cat}]")
                            st.info(f"💡 {reasoning}")
                        
                        st.divider()
                        
                except Exception as e:
                    st.error(f"{name} 오류: {e}")

            if st.button("🗑️ 포트폴리오 전체 삭제"):
                doc_ref.delete()
                st.rerun()
        else:
            st.info("위 검색창에서 종목을 검색하여 추가해주세요.")
