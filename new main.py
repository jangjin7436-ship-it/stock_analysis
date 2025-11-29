import streamlit as st
import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import datetime
import time
import json
import concurrent.futures
import requests

# ---------------------------------------------------------
# 0. 파이어베이스(DB) 설정
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
# 1. 설정 및 매핑
# ---------------------------------------------------------
st.set_page_config(page_title="AI 주식 스캐너 Pro", page_icon="📈", layout="wide")

if 'scan_result_df' not in st.session_state:
    st.session_state['scan_result_df'] = None

TICKER_MAP = {
    "INTC": "인텔", "005290.KS": "동진쎄미켐", "SOXL": "반도체 3X(Bull)", 
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
    "NVDA": "엔비디아", "GE": "GE에어로스페이스", "V": "비자", 
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

SEARCH_LIST = [f"{name} ({code})" for code, name in TICKER_MAP.items()]
SEARCH_MAP = {f"{name} ({code})": code for code, name in TICKER_MAP.items()}
USER_WATCHLIST = list(TICKER_MAP.keys())

# ---------------------------------------------------------
# 2. 데이터 수집 (네이버 / 야후 / FDR)
# ---------------------------------------------------------
def fetch_kr_polling(ticker):
    """국내 주식 실시간"""
    code = ticker.split('.')[0]
    try:
        url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"}
        res = requests.get(url, headers=headers, timeout=3)
        res.raise_for_status()
        data = res.json()
        datas = data.get("datas", [])
        if not datas: raise ValueError("no datas")
        item = datas[0]
        
        over_info = item.get("overMarketPriceInfo") or {}
        over_price_str = str(over_info.get("overPrice", "")).replace(",", "").strip()
        close_price_str = str(item.get("closePrice", "")).replace(",", "").strip()

        over_price = float(over_price_str) if over_price_str not in ("", "0") else None
        close_price = float(close_price_str) if close_price_str not in ("", "0") else None

        # 시간 비교
        def _parse_dt(s):
            try: return datetime.datetime.fromisoformat(s) if s else None
            except: return None
        base_time = _parse_dt(item.get("localTradedAt", ""))
        over_time = _parse_dt(over_info.get("localTradedAt", ""))

        chosen_price = None
        chosen_time = None

        if close_price is not None:
            chosen_price, chosen_time = close_price, base_time
        if over_price is not None:
            if over_time and chosen_time:
                if over_time > chosen_time: chosen_price = over_price
            elif chosen_price is None:
                chosen_price = over_price

        if chosen_price is not None:
            return (ticker, float(chosen_price))
        raise ValueError("no price")
    except:
        # 폴백
        try:
            df = fdr.DataReader(code, "2023-01-01")
            if not df.empty: return (ticker, float(df["Close"].iloc[-1]))
        except: pass
        return (ticker, None)

def fetch_us_1m_candle(ticker):
    """미국 주식 1분봉"""
    try:
        df = yf.download(ticker, period="5d", interval="1m", prepost=True, progress=False)
        if not df.empty: return (ticker, float(df['Close'].iloc[-1]))
        return (ticker, None)
    except: return (ticker, None)

def fetch_history_data(ticker):
    """지표 분석용 일봉"""
    try:
        if ticker.endswith('.KS') or ticker.endswith('.KQ'):
            df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        else:
            df = yf.download(ticker, period="2y", interval="1d", progress=False, prepost=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
        return (ticker, df)
    except: return (ticker, None)

@st.cache_data(ttl=0)
def get_precise_data(tickers_list):
    """데이터 수집 통합 함수"""
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]

    realtime_prices = {}
    hist_map = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        fut_real = []
        for t in kr_tickers: fut_real.append(executor.submit(fetch_kr_polling, t))
        for t in us_tickers: fut_real.append(executor.submit(fetch_us_1m_candle, t))
        fut_hist = [executor.submit(fetch_history_data, t) for t in tickers_list]

        for f in concurrent.futures.as_completed(fut_real):
            tk, p = f.result()
            if p is not None: realtime_prices[tk] = p

        for f in concurrent.futures.as_completed(fut_hist):
            tk, df = f.result()
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.sort_index()
                hist_map[tk] = df

    return hist_map, realtime_prices

# ---------------------------------------------------------
# 3. 로직 및 지표 계산 (공통 엔진)
# ---------------------------------------------------------

def calculate_indicators(df, realtime_price=None):
    """
    [핵심] 실시간 가격을 받아서 지표 계산 직전에 DataFrame의 마지막 종가를 강제로 업데이트
    """
    if len(df) < 60: return None
    df = df.copy()

    if 'Close' in df.columns:
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        close_series = close.copy()
    else: return None

    # ❗ 실시간 가격 반영 (중요)
    if realtime_price is not None and realtime_price > 0:
        close_series.iloc[-1] = realtime_price

    close_series = close_series.ffill()
    df['Close_Calc'] = close_series

    # MA
    df['MA5'] = df['Close_Calc'].rolling(window=5).mean()
    df['MA10'] = df['Close_Calc'].rolling(window=10).mean()
    df['MA20'] = df['Close_Calc'].rolling(window=20).mean()
    df['MA60'] = df['Close_Calc'].rolling(window=60).mean()

    # Volatility / Momentum
    df['STD20'] = df['Close_Calc'].rolling(window=20).std()
    df['MOM10'] = df['Close_Calc'] / df['Close_Calc'].shift(10) - 1

    # Volume
    if 'Volume' in df.columns:
        vol = df['Volume']
        if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]
        df['Volume_Calc'] = vol
        df['VolMA20'] = vol.rolling(window=20).mean()
    else:
        df['Volume_Calc'] = 0
        df['VolMA20'] = 0

    # RSI
    delta = df['Close_Calc'].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)

    return df.dropna()

def analyze_advanced_strategy(df):
    """전략 판단 로직 (스캐너/포트폴리오 공통 사용)"""
    if df is None or df.empty:
        return "분석 불가", "gray", "데이터 부족", 0

    try:
        curr = float(df['Close_Calc'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        ma60 = float(df['MA60'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        sig = float(df['Signal_Line'].iloc[-1])
        prev_macd = float(df['MACD'].iloc[-2])
        prev_sig = float(df['Signal_Line'].iloc[-2])
        std20 = float(df['STD20'].iloc[-1])
        mom10 = float(df['MOM10'].iloc[-1]) if 'MOM10' in df.columns else 0.0
        vol = float(df['Volume_Calc'].iloc[-1]) if 'Volume_Calc' in df.columns else 0.0
        vol_ma = float(df['VolMA20'].iloc[-1]) if 'VolMA20' in df.columns else 0.0
        prev_close = float(df['Close_Calc'].iloc[-2])
    except: return "오류", "gray", "계산 실패", 0

    score = 50
    reasons = []

    # 1) 추세
    if curr > ma60 and ma20 > ma60:
        score += 20
        reasons.append("📈 중기 상승 추세 (60일선 위)")
    elif curr > ma60:
        score += 5
        reasons.append("↗ 60일선 위 (추세 형성 중)")
    else:
        score -= 25
        reasons.append("⚠ 하락 추세 (60일선 아래)")

    # 2) 위치 (눌림목)
    dist_ma20 = (curr - ma20) / ma20 if ma20 > 0 else 0
    if (curr >= ma20) and (curr >= ma60) and (-0.03 <= dist_ma20 <= 0.02):
        score += 20
        reasons.append("💎 황금 눌림목 (20일선 근접)")
    elif 0.02 < dist_ma20 <= 0.07:
        score += 5
        reasons.append("🙂 상승 유지 (과열 아님)")
    elif dist_ma20 > 0.07:
        score -= 15
        reasons.append("🔥 단기 과열 (20일선 이격 과다)")

    # 3) RSI (물결표 사용 금지 -> '-')
    if 40 <= rsi <= 60:
        score += 15
        reasons.append(f"⚖ RSI {rsi:.0f} (40-60 균형)")
    elif 30 <= rsi < 40:
        score += 5
        reasons.append("반등 기대 (약한 과매도)")
    elif rsi < 30:
        score += 15
        reasons.append("심한 과매도 (역발상)")
    elif rsi > 70:
        score -= 20
        reasons.append("🚨 RSI 과열 (조정 주의)")

    # 4) 모멘텀
    if 0.03 <= mom10 <= 0.15:
        score += 10
        reasons.append(f"📊 최근 2주간 {mom10*100:.1f}% 상승")
    elif mom10 > 0.25:
        score -= 15
        reasons.append(f"급등 피로감 (2주간 {mom10*100:.1f}% 폭등)")
    elif mom10 < -0.10:
        score -= 10
        reasons.append("낙폭 과대")

    # 5) MACD
    if macd > sig and prev_macd <= prev_sig:
        score += 15
        reasons.append("⚡ MACD 골든크로스")
    elif macd > sig:
        score += 5
        reasons.append("MACD 상방")
    elif macd < sig and prev_macd >= prev_sig:
        score -= 10
        reasons.append("💧 MACD 데드크로스")

    # 6) 변동성
    vol_ratio = std20 / curr if curr > 0 else 0
    if vol_ratio > 0.08:
        score -= 15
        reasons.append("🎢 변동성 큼")
    elif vol_ratio < 0.03:
        score += 5
        reasons.append("⚙ 안정적 변동성")
    
    if vol_ma > 0 and vol > vol_ma * 1.5 and curr > prev_close:
        score += 10
        reasons.append("🔥 거래량 실린 상승")

    score = max(0, min(100, score))

    if score >= 80: cat, col = "🚀 단기 강력 매수", "green"
    elif score >= 65: cat, col = "📈 매수 우위", "blue"
    elif score >= 45: cat, col = "👀 관망", "gray"
    elif score >= 25: cat, col = "📉 매도/비중 축소", "red"
    else: cat, col = "💥 강력 매도", "red"

    if not reasons: reasons.append("관망")
    return cat, col, " / ".join(reasons[:4]), score

def analyze_single_ticker(ticker, raw_data_dict, realtime_map):
    """
    ⭐ [단일 진실 공급원] ⭐
    스캐너와 포트폴리오가 모두 이 함수를 사용하여 분석 결과를 얻습니다.
    따라서 결과가 다를 수가 없습니다.
    """
    # 1. 원천 데이터 가져오기
    df_raw = raw_data_dict.get(ticker)
    real_p = realtime_map.get(ticker)
    
    # 2. 데이터 유효성 검사
    if df_raw is None or df_raw.empty:
        return None
    
    # 3. 지표 계산 (여기서 실시간 가격 패치됨)
    df_indi = calculate_indicators(df_raw, realtime_price=real_p)
    if df_indi is None or df_indi.empty:
        return None
        
    # 4. 분석 수행
    cat, col, reason, score = analyze_advanced_strategy(df_indi)
    
    # 5. 결과 반환 (현재가는 df_indi의 마지막 값을 사용 = 패치된 값)
    final_price = float(df_indi['Close_Calc'].iloc[-1])
    rsi = float(df_indi['RSI'].iloc[-1])
    
    return {
        "ticker": ticker,
        "name": TICKER_MAP.get(ticker, ticker),
        "price": final_price,
        "rsi": rsi,
        "score": score,
        "category": cat,
        "color": col,
        "reason": reason
    }

def calculate_total_profit(ticker, avg_price, current_price, quantity):
    """순수익 계산"""
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
    qty, avg, curr = float(quantity), float(avg_price), float(current_price)
    
    total_buy = avg * qty
    gross_eval = curr * qty
    
    fee_rate = 0.000295 if is_kr else 0.001965
    tax_rate = 0.0015 if is_kr else 0.0
    
    sell_fee = gross_eval * fee_rate
    sell_tax = gross_eval * tax_rate
    net_eval = gross_eval - sell_fee - sell_tax
    net_profit = net_eval - total_buy
    pct = (net_profit / total_buy) * 100 if total_buy > 0 else 0.0
    
    return {
        "pct": pct, "profit_amt": net_profit, 
        "net_eval_amt": net_eval, "currency": "₩" if is_kr else "$"
    }

# ---------------------------------------------------------
# 4. UI 구성
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2, tab3 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오", "📘 알고리즘 설명서"])

# =========================================================
# TAB 1: 스캐너
# =========================================================
with tab1:
    st.markdown("### 📋 AI 정밀 스캐너")
    st.caption("포트폴리오와 완벽히 동일한 '단일 진실 공급원' 함수 사용")

    if st.button("🔄 분석 새로고침", type="primary"):
        st.session_state['scan_result_df'] = None 
        st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 분석 시작"):
            with st.spinner('데이터 수집 및 통합 분석 중...'):
                raw_data, real_data = get_precise_data(USER_WATCHLIST)
                results = []
                prog = st.progress(0)
                
                for i, tk in enumerate(USER_WATCHLIST):
                    # ⭐ 공통 분석 함수 호출 ⭐
                    res = analyze_single_ticker(tk, raw_data, real_data)
                    if res:
                        is_kr = tk.endswith(".KS") or tk.endswith(".KQ")
                        sym = "₩" if is_kr else "$"
                        fmt_p = f"{sym}{res['price']:,.0f}" if is_kr else f"{sym}{res['price']:,.2f}"
                        
                        results.append({
                            "종목명": f"{res['name']} ({tk})",
                            "점수": res['score'],
                            "현재가": fmt_p,
                            "RSI": res['rsi'],
                            "AI 등급": res['category'],
                            "핵심 요약": res['reason']
                        })
                    prog.progress((i+1)/len(USER_WATCHLIST))
                
                if results:
                    df_res = pd.DataFrame(results).sort_values('점수', ascending=False)
                    st.session_state['scan_result_df'] = df_res
                    st.rerun()
                else:
                    st.error("데이터 수집 실패")

    if st.session_state['scan_result_df'] is not None:
        st.dataframe(
            st.session_state['scan_result_df'],
            use_container_width=True,
            height=700,
            column_config={
                "점수": st.column_config.ProgressColumn("AI 점수", format="%d점", min_value=0, max_value=100),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f")
            },
            hide_index=True
        )

# =========================================================
# TAB 2: 포트폴리오
# =========================================================
with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오")
    
    db = get_db()
    if not db:
        st.warning("⚠️ Firebase 설정 필요")
    else:
        col_u1, _ = st.columns([1, 3])
        user_id = col_u1.text_input("닉네임", value="장동진")
        doc_ref = db.collection('portfolios').document(user_id)
        
        try:
            doc = doc_ref.get()
            pf_data = doc.to_dict().get('stocks', []) if doc.exists else []
        except: pf_data = []

        # 종목 추가 UI
        with st.container():
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            sel = c1.selectbox("종목 검색", ["선택"] + SEARCH_LIST)
            p_in = c2.number_input("평단가", 0.0, format="%.2f")
            q_in = c3.number_input("수량", 1)
            if c4.button("추가") and sel != "선택":
                code = SEARCH_MAP[sel]
                new_data = [x for x in pf_data if x['ticker'] != code]
                new_data.append({"ticker": code, "price": p_in, "qty": q_in})
                doc_ref.set({'stocks': new_data})
                st.rerun()
        
        st.divider()

        # 리스트 출력
        if pf_data:
            my_tickers = [x['ticker'] for x in pf_data]
            with st.spinner("분석 중..."):
                raw_data, real_data = get_precise_data(my_tickers)
            
            disp_list = []
            for item in pf_data:
                tk = item['ticker']
                
                # ⭐ 공통 분석 함수 호출 ⭐ (스캐너와 완벽히 동일)
                res = analyze_single_ticker(tk, raw_data, real_data)
                
                if res:
                    # 수익률 계산만 추가
                    p_res = calculate_total_profit(tk, item['price'], res['price'], item['qty'])
                    
                    disp_list.append({
                        **res,
                        "avg": item['price'], "qty": item['qty'],
                        "profit_pct": p_res['pct'], "profit_amt": p_res['profit_amt'],
                        "eval_amt": p_res['net_eval_amt'], "curr_sym": p_res['currency']
                    })
                else:
                    # 로딩 실패 시
                    disp_list.append({
                        "ticker": tk, "name": TICKER_MAP.get(tk, tk),
                        "avg": item['price'], "qty": item['qty'], "price": item['price'],
                        "score": 0, "category": "로딩 실패", "color": "gray", "reason": "데이터 없음",
                        "profit_pct": 0.0, "profit_amt": 0.0, "eval_amt": 0.0, "curr_sym": "$"
                    })

            disp_list.sort(key=lambda x: x['score'], reverse=True)

            for d in disp_list:
                with st.container():
                    c1, c2, c3 = st.columns([1.5, 1.5, 4])
                    sym = d['curr_sym']
                    safe_sym = sym if sym != "$" else "&#36;"
                    
                    with c1:
                        st.markdown(f"### {d['name']}")
                        st.caption(f"{d['ticker']} | {d['qty']}주")
                    with c2:
                        profit_str = f"{d['profit_amt']:,.0f}" if sym=="₩" else f"{d['profit_amt']:,.2f}"
                        eval_str = f"{d['eval_amt']:,.0f}" if sym=="₩" else f"{d['eval_amt']:,.2f}"
                        st.metric("순수익", f"{d['profit_pct']:.2f}%", delta=f"{sym}{profit_str}")
                        st.markdown(f"**평가금:** {safe_sym}{eval_str}", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"**AI 점수: {d['score']}점**")
                        st.markdown(f"**판단:** :{d['color']}[{d['category']}]")
                        st.info(f"💡 {d['reason']}")
                    st.divider()

            if st.button("🗑️ 전체 삭제"):
                doc_ref.delete()
                st.rerun()

with tab3:
    st.markdown("## 📘 AI 투자 전략 알고리즘")
    st.markdown("스캐너와 포트폴리오는 이제 100% 동일한 로직을 사용합니다.")
