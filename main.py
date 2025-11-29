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

# 종목 코드와 한글명 매핑 (사용자 요청 리스트 기반)
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

USER_WATCHLIST = list(TICKER_MAP.keys()) # 키 값들을 감시 리스트로 사용

def format_ticker(ticker):
    """입력된 코드를 포맷팅하고 이름을 반환"""
    ticker = ticker.strip().upper()
    # 숫자만 있는 경우 .KS 붙임
    if ticker.isdigit():
        ticker = f"{ticker}.KS"
    
    # 이름 찾기
    name = TICKER_MAP.get(ticker, ticker) # 없으면 티커 그대로
    return ticker, name

# ---------------------------------------------------------
# 2. 데이터 로드 및 지표 계산
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_bulk_data(tickers_list):
    """데이터 다운로드 (2년치)"""
    # 딕셔너리 키(Formatted Ticker)를 그대로 사용
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
    dip_buy = trend_up and (curr <= ma20 * 1.02) and (curr >= ma20 * 0.98) # 상승세 중 눌림목

    reasons = []
    
    # 1. 등급 및 코멘트 결정
    category = "중립/관망 (Hold)"
    color = "gray" # default

    # A. 강력 매수 (Strong Buy)
    # 조건: 장기 상승 추세 + 눌림목 지지 or 골든크로스 + 과열 아님
    if trend_up and (dip_buy or (golden_cross and not overbought)):
        category = "🚀 강력 매수 (Strong Buy)"
        color = "#00C853" # 진한 녹색
        if dip_buy: reasons.append("상승 추세 속 '눌림목' 지지 구간")
        if golden_cross: reasons.append("MACD 골든크로스로 상승 탄력 강화")

    # B. 매수 (Buy)
    # 조건: 추세가 좋거나, 과매도권에서의 기술적 반등
    elif (trend_up and above_ma20) or (oversold and curr > ma20 * 0.95):
        category = "📈 매수 (Buy)"
        color = "#2962FF" # 파란색
        if trend_up and above_ma20: reasons.append("정배열 상승 추세 유지 중")
        if oversold: reasons.append(f"RSI {rsi:.0f}로 과매도 구간, 기술적 반등 기대")

    # C. 매도 (Sell)
    # 조건: 하락 추세 전환 or 심각한 과열
    elif (not trend_up and not above_ma20) or (overbought and dead_cross):
        category = "📉 매도 (Sell)"
        color = "#FF5252" # 붉은색
        if not trend_up: reasons.append("60일선 하회로 중기 추세 꺾임")
        if overbought: reasons.append(f"RSI {rsi:.0f}로 과열, 차익실현 매물 주의")
        if dead_cross: reasons.append("MACD 데드크로스 발생 (조정 신호)")

    # D. 강력 매도 (Strong Sell)
    # 조건: 역배열 + 데드크로스
    elif not trend_up and curr < ma20 and dead_cross:
        category = "💥 강력 매도 (Strong Sell)"
        color = "#D50000" # 진한 빨강
        reasons.append("역배열 하락 추세 + 하락 모멘텀 가속화")

    # E. 관망 (Neutral)
    else:
        category = "👀 관망 (Neutral)"
        color = "#757575" # 회색
        if overbought: reasons.append("추세는 좋으나 과열권, 신규 진입 자제")
        elif not trend_up and above_ma20: reasons.append("단기 반등 중이나 장기 추세 확인 필요")
        else: reasons.append("뚜렷한 방향성 없음, 횡보세")

    # 이유가 비어있으면 기본 코멘트
    if not reasons:
        if rsi > 50: reasons.append("특이 신호 부재, 추세 지속 여부 관찰")
        else: reasons.append("거래량 및 모멘텀 부족")

    return category, color, ", ".join(reasons)

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)"])

# === TAB 1: 스캐너 ===
with tab1:
    st.markdown("### 📋 시장 전체 스캔 및 AI 분석")
    st.caption("사용자님이 지정한 관심 종목 전체를 실시간으로 분석하여 투자 기회를 포착합니다.")

    if st.button("🔍 전체 리스트 정밀 분석 시작", type="primary"):
        with st.spinner('AI가 차트 패턴과 보조지표를 분석 중입니다...'):
            # USER_WATCHLIST는 이미 Formatted Ticker들이므로 바로 사용
            raw_data = get_bulk_data(USER_WATCHLIST)
            
            scan_results = []
            
            progress_bar = st.progress(0)
            for i, ticker_code in enumerate(USER_WATCHLIST):
                try:
                    # 데이터 추출
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        try: df_ticker = raw_data.xs(ticker_code, axis=1, level=1)
                        except: df_ticker = raw_data[ticker_code]
                    else:
                        df_ticker = raw_data
                    
                    df_ticker = df_ticker.dropna(how='all')
                    if df_ticker.empty: continue
                    
                    df_indi = calculate_indicators(df_ticker)
                    if df_indi is None: continue

                    # 분석 수행
                    cat, color_code, reasoning = analyze_advanced_strategy(df_indi)
                    
                    # 표시용 데이터 생성
                    curr_price = df_indi['Close'].iloc[-1]
                    rsi_val = df_indi['RSI'].iloc[-1]
                    name = TICKER_MAP.get(ticker_code, ticker_code) # 한글명 변환
                    
                    scan_results.append({
                        "종목명": name,
                        "코드": ticker_code,
                        "현재가": curr_price,
                        "RSI": rsi_val,
                        "AI 추천": cat,
                        "분석 요약": reasoning,
                        "color": color_code # 정렬/필터링용
                    })
                except:
                    continue
                progress_bar.progress((i + 1) / len(USER_WATCHLIST))
            
            st.success("분석 완료!")
            
            if scan_results:
                df_res = pd.DataFrame(scan_results)
                
                # 정렬: 강력 매수 -> 매수 -> 관망 ... 순으로 보기 위해 커스텀 정렬
                rank_map = {"🚀": 0, "📈": 1, "👀": 2, "📉": 3, "💥": 4}
                df_res['rank'] = df_res['AI 추천'].apply(lambda x: rank_map.get(x[0], 5))
                df_res = df_res.sort_values('rank')
                
                # UI: Streamlit Dataframe Column Config 활용 (깔끔한 디자인)
                st.dataframe(
                    df_res[['종목명', '현재가', 'RSI', 'AI 추천', '분석 요약']],
                    use_container_width=True,
                    height=700,
                    column_config={
                        "종목명": st.column_config.TextColumn("종목명", help="종목의 한글 이름"),
                        "현재가": st.column_config.NumberColumn("현재가", format="%.0f"),
                        "RSI": st.column_config.ProgressColumn(
                            "RSI (강도)", 
                            help="상대강도지수 (30이하:과매도, 70이상:과매수)",
                            format="%.1f",
                            min_value=0, max_value=100,
                        ),
                        "AI 추천": st.column_config.TextColumn("AI 종합 의견", width="medium"),
                        "분석 요약": st.column_config.TextColumn("상세 분석 사유", width="large"),
                    },
                    hide_index=True
                )
            else:
                st.error("분석할 데이터가 없습니다.")

# === TAB 2: 포트폴리오 ===
with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오")
    
    db = get_db()
    if not db:
        st.warning("⚠️ Firebase 설정이 필요합니다.")
    else:
        col_u1, col_u2 = st.columns([1, 3])
        with col_u1:
            user_id = st.text_input("닉네임 입력", value="my_portfolio")
        
        doc_ref = db.collection('portfolios').document(user_id)
        
        # 불러오기
        try:
            doc = doc_ref.get()
            pf_data = doc.to_dict().get('stocks', []) if doc.exists else []
        except: pf_data = []

        # 추가 폼
        with st.expander("➕ 종목 추가하기", expanded=False):
            with st.form("add_pf"):
                c1, c2 = st.columns(2)
                input_ticker = c1.text_input("종목 코드 (예: TSLA, 005930)")
                input_price = c2.number_input("내 평단가", min_value=0.0)
                if st.form_submit_button("리스트에 추가"):
                    fmt_ticker, _ = format_ticker(input_ticker)
                    # 기존 것 삭제 후 추가
                    pf_data = [p for p in pf_data if p['ticker'] != fmt_ticker]
                    pf_data.append({"ticker": fmt_ticker, "price": input_price})
                    doc_ref.set({'stocks': pf_data})
                    st.rerun()

        st.divider()

        if pf_data:
            st.subheader(f"{user_id}님의 보유 종목 진단")
            
            my_tickers = [p['ticker'] for p in pf_data]
            with st.spinner("보유 종목 정밀 분석 중..."):
                my_raw = get_bulk_data(my_tickers)
            
            # 카드 뷰 스타일
            for item in pf_data:
                tk = item['ticker']
                avg = item['price']
                name = TICKER_MAP.get(tk, tk)
                
                try:
                    # 데이터 추출
                    if isinstance(my_raw.columns, pd.MultiIndex):
                        try: df_tk = my_raw.xs(tk, axis=1, level=1)
                        except: df_tk = my_raw[tk]
                    else: df_tk = my_raw

                    df_tk = df_tk.dropna(how='all')
                    df_indi = calculate_indicators(df_tk)
                    
                    if df_indi is None:
                        st.warning(f"{name}: 데이터 부족")
                        continue

                    # 분석
                    cat, color_hex, reasoning = analyze_advanced_strategy(df_indi)
                    curr = df_indi['Close'].iloc[-1]
                    profit_pct = ((curr - avg) / avg) * 100
                    
                    # 수익률 색상
                    pct_color = "red" if profit_pct < 0 else "green"
                    
                    # 카드 UI
                    with st.container():
                        # 다크모드 대응을 위한 HTML/CSS 스타일링 없는 Streamlit 네이티브 활용
                        c1, c2, c3 = st.columns([1.5, 1.5, 4])
                        
                        with c1:
                            st.markdown(f"### {name}")
                            st.caption(f"{tk}")
                        
                        with c2:
                            st.metric("수익률", f"{profit_pct:.2f}%", delta=f"{curr - avg:.0f}")
                            st.caption(f"평단: {avg:,.0f} / 현재: {curr:,.0f}")
                            
                        with c3:
                            # 추천 등급 배지
                            st.markdown(f"**AI 판단:** :{color_hex}[{cat}]")
                            st.info(f"💡 **분석:** {reasoning}")
                        
                        st.divider()
                        
                except Exception as e:
                    st.error(f"{name} 분석 중 오류 발생")

            if st.button("🗑️ 포트폴리오 전체 초기화"):
                doc_ref.delete()
                st.rerun()
        else:
            st.info("저장된 종목이 없습니다. '종목 추가하기'를 눌러 포트폴리오를 구성해보세요.")
