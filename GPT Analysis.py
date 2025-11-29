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
        except Exception:
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
# 2. 데이터 수집 (After-Market 반영 수정)
# ---------------------------------------------------------
@st.cache_data(ttl=60) 
def get_bulk_history_data(us_tickers):
    """지표 계산용 히스토리 (Daily)"""
    if not us_tickers: return {}
    hist_map = {}
    try:
        df_hist = yf.download(us_tickers, period="2y", interval="1d", progress=False, group_by="ticker", auto_adjust=False)
        hist_is_multi = isinstance(df_hist.columns, pd.MultiIndex)
        for t in us_tickers:
            try:
                sub_df = df_hist[t] if hist_is_multi else df_hist
                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                    sub_df = sub_df.dropna(how="all")
                    if "Close" in sub_df.columns:
                        hist_map[t] = sub_df
            except: pass
    except: pass
    return hist_map

@st.cache_data(ttl=5) # 5초마다 갱신 (실시간성 강화)
def get_bulk_realtime_data(us_tickers):
    """
    [수정] 애프터마켓/프리마켓 가격 반영을 위한 로직
    interval='1m', prepost=True 옵션을 사용하여 장외 거래 가격($54.54 등)을 포착함
    """
    if not us_tickers: return {}
    realtime_map = {}
    try:
        # period를 짧게, prepost=True로 장외 데이터 포함
        df_real = yf.download(us_tickers, period="1d", interval="1m", progress=False, group_by="ticker", prepost=True)
        real_is_multi = isinstance(df_real.columns, pd.MultiIndex)

        for t in us_tickers:
            try:
                sub_real = df_real[t] if real_is_multi else df_real
                if isinstance(sub_real, pd.DataFrame) and not sub_real.empty:
                    sub_real = sub_real.dropna(how="all")
                    price_series = sub_real["Close"]
                    if price_series is not None:
                        valid_closes = price_series.dropna()
                        if not valid_closes.empty:
                            # 가장 마지막 틱의 가격 (장외 포함)
                            realtime_map[t] = float(valid_closes.iloc[-1])
            except: pass
    except: pass
    return realtime_map

def fetch_kr_polling(ticker):
    """국내 주식 실시간 (네이버)"""
    code = ticker.split('.')[0]
    try:
        url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=2)
        data = res.json()
        item = data['datas'][0]
        # 네이버 금융 API는 장중/장후 실시간 가격 제공
        close = float(str(item['closePrice']).replace(',', ''))
        return ticker, close
    except Exception:
        return ticker, None

def fetch_kr_history(ticker):
    try:
        df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        return ticker, df
    except: return ticker, None

def get_precise_data(tickers_list):
    if not tickers_list: return {}, {}
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]

    hist_map = get_bulk_history_data(us_tickers)
    realtime_map = get_bulk_realtime_data(us_tickers) # 여기가 수정됨

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        fut_real = [executor.submit(fetch_kr_polling, t) for t in kr_tickers]
        fut_hist = [executor.submit(fetch_kr_history, t) for t in kr_tickers]
        for f in fut_real:
            try:
                tk, p = f.result(timeout=3)
                if p: realtime_map[tk] = p
            except: continue
        for f in fut_hist:
            try:
                tk, df = f.result(timeout=5)
                if df: hist_map[tk] = df
            except: continue
    return hist_map, realtime_map

def get_current_exchange_rate():
    try:
        df = yf.Ticker("KRW=X").history(period="1d")
        if not df.empty: return float(df['Close'].iloc[-1])
        return 1430.0
    except: return 1430.0

# ---------------------------------------------------------
# 3. 분석 엔진
# ---------------------------------------------------------
def calculate_indicators(df, realtime_price=None):
    if df is None or len(df) < 120: return None
    if isinstance(df, pd.Series): df = df.to_frame()
    df = df.copy()

    if 'Close' in df.columns: df['Close_Calc'] = df['Close']
    elif 'Adj Close' in df.columns: df['Close_Calc'] = df['Adj Close']
    else: return None
    df['Close_Calc'] = df['Close_Calc'].astype(float)
    
    if 'High' not in df.columns: df['High'] = df['Close_Calc']
    if 'Low' not in df.columns: df['Low'] = df['Close_Calc']

    # [수정] 실시간 가격(장외 포함)으로 마지막 캔들 업데이트
    if realtime_price is not None:
        try:
            rp = float(realtime_price)
            if rp > 0:
                df.iloc[-1, df.columns.get_loc('Close_Calc')] = rp
                if rp > df.iloc[-1]['High']: df.iloc[-1, df.columns.get_loc('High')] = rp
                if rp < df.iloc[-1]['Low']: df.iloc[-1, df.columns.get_loc('Low')] = rp
        except: pass

    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA10'] = df['Close_Calc'].rolling(10).mean()
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()
    df['MA120'] = df['Close_Calc'].rolling(120).mean()
    df['Disparity_20'] = df['Close_Calc'] / df['MA20']
    df['MA20_Slope'] = df['MA20'].diff()
    df['MA60_Slope'] = df['MA60'].diff()
    df['MA120_Slope'] = df['MA120'].diff()
    
    std = df['Close_Calc'].rolling(20).std()
    df['Upper_Band'] = df['MA20'] + (std * 2)
    df['Lower_Band'] = df['MA20'] - (std * 2)
    
    delta = df['Close_Calc'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)
    
    prev_close = df['Close_Calc'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - prev_close)
    tr3 = abs(df['Low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    if 'Volume' in df.columns:
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    else: df['Vol_Ratio'] = 1.0

    return df.dropna()

def get_ai_score_row(row):
    try:
        score = 50.0
        curr = row['Close_Calc']
        ma5, ma20, ma60 = row['MA5'], row['MA20'], row['MA60']
        rsi = row['RSI']
        
        if row['MA60_Slope'] > 0: score += 10.0
        else: score -= 10.0
        if curr > ma60: score += 5.0
        else: score -= 5.0
        if row['MA120_Slope'] > 0: score += 5.0
        elif row['MA120_Slope'] < 0: score -= 5.0

        if row['MA20_Slope'] > 0:
            if curr > ma20:
                score += 5.0
                if curr < ma5 * 1.01: score += 5.0
        
        disparity = row['Disparity_20']
        if disparity > 1.10: score -= 20.0
        elif disparity > 1.05: score -= 5.0

        if row['MACD_Hist'] > row['Prev_MACD_Hist']: score += 5.0
        if 40 <= rsi <= 60: score += 5.0
        elif rsi > 70: score -= 10.0
        if curr <= row['Lower_Band'] * 1.02: score += 10.0
        if row['Vol_Ratio'] >= 1.5 and curr > row['Open']: score += 5.0

        return max(0.0, min(100.0, score))
    except: return 0.0

def analyze_advanced_strategy(df):
    if df is None or df.empty: return "분석 불가", "gray", "데이터 부족", 0.0
    try:
        row = df.iloc[-1]
        score = float(get_ai_score_row(row))
        curr = float(row['Close_Calc'])
        ma60 = float(row['MA60'])
        rsi = float(row['RSI'])
        atr = float(row['ATR'])
        disparity = float(row['Disparity_20'])
    except: return "오류", "gray", "계산 실패", 0.0

    reasons = []
    if row['MA60_Slope'] > 0 and curr > ma60: reasons.append("상승 추세(60일↑)")
    elif row['MA60_Slope'] < 0: reasons.append("하락 추세(60일↓)")
    if disparity > 1.1: reasons.append("⚠️ 과열(이격도 110%↑)")
    elif 1.0 <= disparity <= 1.03: reasons.append("⚡ 20일선 근접(눌림)")
    elif disparity < 0.97: reasons.append("📉 과매도 구간")
    atr_ratio = atr / curr if curr > 0 else 0
    if atr_ratio > 0.05: reasons.append("⚠️ 고변동성 주의")
    
    is_high_risk = atr_ratio > 0.05
    if score >= 75 and not is_high_risk: cat, col = "🚀 AI 스나이퍼 매수 (강력)", "green"
    elif score >= 60 and not is_high_risk: cat, col = "📈 매수 우위 (양호)", "blue"
    elif disparity > 1.1 or rsi > 70: cat, col = "📉 이익 실현 / 과열", "orange"
    elif score < 40: cat, col = "💥 매도 / 관망 권장", "red"
    else: cat, col = "👀 중립 / 관망", "gray"

    reasoning = " / ".join(reasons[:3]) if reasons else "지표 중립"
    return cat, col, reasoning, round(score, 2)

def calculate_total_profit(ticker, avg_price, current_price, quantity):
    """
    [수정] 사용자 사진 기반 토스증권 역산 요율 적용
    - KR Fee: 0.0295% | KR Tax: 0.15% (총 ~0.1795%)
    - US Fee: 0.1968% (~0.2%) | US Tax: 0% (사진 기준)
    """
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
    qty, avg, curr = float(quantity), float(avg_price), float(current_price)

    # 1. 매수 총액
    total_buy_cost = avg * qty
    
    # 2. 현재 평가금액 (세전)
    gross_eval = curr * qty

    # 3. 매도 시 예상 비용 (사진 기반 요율 적용)
    if is_kr:
        # 사진: 수수료 299/1,013,179 = 0.000295...
        # 사진: 세금 1,522/1,013,179 = 0.001502... (약 0.15%)
        sell_fee_rate = 0.000295 
        sell_tax_rate = 0.0015
    else:
        # 사진: 수수료 $0.75 / $381.03 = 0.001968... (약 0.2%)
        # 사진: 세금 $0.00
        sell_fee_rate = 0.001968
        sell_tax_rate = 0.0

    sell_cost = gross_eval * (sell_fee_rate + sell_tax_rate)

    # 4. 세후 평가금액 (매도 시 내 손에 쥐는 돈)
    net_eval = gross_eval - sell_cost
    
    # 5. 순수익
    net_profit = net_eval - total_buy_cost
    
    pct = (net_profit / total_buy_cost) * 100 if total_buy_cost > 0 else 0.0

    return {
        "pct": pct,
        "profit_amt": net_profit,
        "net_eval_amt": net_eval,
        "currency": "₩" if is_kr else "$",
        "detail": f"수수료율: {sell_fee_rate*100:.3f}%"
    }

# ---------------------------------------------------------
# 4. UI
# ---------------------------------------------------------
st.title("🎯 AI 주식 스캐너 (Real-time)")

tab1, tab2, tab3 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)", "📘 알고리즘 백서"])

with tab1:
    st.markdown("### 📋 AI 정밀 스캐너")
    st.caption("After-Market($) 가격 반영 | AI 스나이퍼 전략 분석")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 분석 새로고침", type="primary"):
            st.session_state['scan_result_df'] = None
            st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 리스트 정밀 분석 시작"):
            with st.spinner('초정밀 데이터(After-Market 포함) 수집 중...'):
                raw_data_dict, realtime_map = get_precise_data(USER_WATCHLIST)
                scan_results = []
                progress_bar = st.progress(0)

                for i, ticker_code in enumerate(USER_WATCHLIST):
                    if ticker_code not in raw_data_dict: continue
                    try:
                        df_tk = raw_data_dict[ticker_code].dropna(how='all')
                        if df_tk.empty: continue

                        # 실시간 가격(애프터마켓) 우선 사용
                        curr_price = realtime_map.get(ticker_code)
                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)

                        if df_indi is None or df_indi.empty: continue

                        name = TICKER_MAP.get(ticker_code, ticker_code)
                        is_leverage = any(x in name for x in ["3X", "2X", "1.5X"])
                        
                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)
                        if is_leverage and score >= 70: reasoning += " (레버리지 주의)"

                        final_price = float(df_indi['Close_Calc'].iloc[-1])
                        rsi_val = float(df_indi['RSI'].iloc[-1])
                        vol_ratio = float(df_indi['Vol_Ratio'].iloc[-1]) if 'Vol_Ratio' in df_indi.columns else 0

                        is_kr = ticker_code.endswith(".KS") or ticker_code.endswith(".KQ")
                        sym = "₩" if is_kr else "$"
                        fmt_price = f"{sym}{final_price:,.0f}" if is_kr else f"{sym}{final_price:,.2f}"

                        scan_results.append({
                            "종목명": f"{name} ({ticker_code})",
                            "점수": score,
                            "현재가": fmt_price,
                            "RSI": rsi_val,
                            "AI 등급": cat,
                            "핵심 요약": reasoning,
                            "거래량비율": vol_ratio,
                        })
                    except: continue
                    progress_bar.progress((i + 1) / len(USER_WATCHLIST))

                if scan_results:
                    df_res = pd.DataFrame(scan_results)
                    df_res = df_res.sort_values('점수', ascending=False)
                    st.session_state['scan_result_df'] = df_res
                    st.success("완료!")
                    st.rerun()
                else: st.error("데이터 수집 실패.")

    if st.session_state['scan_result_df'] is not None:
        df_scan = st.session_state['scan_result_df']
        try:
            if "점수" in df_scan.columns:
                df_high = df_scan[df_scan["점수"] >= 80.0]
                if not df_high.empty:
                    st.markdown("#### 🔥 강력 매수 시그널 (Score 80+)")
                    st.dataframe(df_high[["종목명", "점수", "현재가", "RSI", "AI 등급", "핵심 요약"]], use_container_width=True, hide_index=True)
        except: pass

        st.dataframe(df_scan, use_container_width=True, height=400, hide_index=True)

        st.divider()
        st.markdown("### 💰 AI 시드 머니 분배기")
        c_seed1, c_seed2, c_seed3 = st.columns([2, 1, 1])
        with c_seed1:
            seed_money = st.number_input("투자 가능 총 현금 (KRW)", min_value=100000, value=10000000, step=100000, format="%d")
        with c_seed2:
            target_count = st.slider("분산 종목 수", min_value=1, max_value=10, value=3)
        with c_seed3:
            st.write("") 
            calc_btn = st.button("🧮 분배 계산", type="primary")

        if calc_btn:
            with st.spinner("💱 실시간 환율 조회 중..."):
                usd_krw = get_current_exchange_rate()
            st.info(f"💡 적용 환율: 1달러 = {usd_krw:,.2f}원")

            candidates = df_scan[df_scan['점수'] >= 75]
            if candidates.empty: candidates = df_scan[df_scan['점수'] >= 60]
            if candidates.empty: candidates = df_scan.copy()

            top_n = candidates.sort_values('점수', ascending=False).head(target_count)

            if top_n.empty: st.error("분석된 종목이 없습니다.")
            else:
                per_stock_budget = seed_money / len(top_n)
                alloc_list = []
                for idx, row in top_n.iterrows():
                    raw_price_str = str(row['현재가']).replace(',', '').replace('$', '').replace('₩', '')
                    try: price = float(raw_price_str)
                    except: price = 0.0
                    
                    is_krw = "₩" in str(row['현재가'])
                    if is_krw:
                        price_krw = price
                        price_usd = price / usd_krw if usd_krw > 0 else 0
                    else:
                        price_usd = price
                        price_krw = price * usd_krw
                        
                    qty = int(per_stock_budget / price_krw) if price_krw > 0 else 0
                    invest_krw = qty * price_krw
                    
                    alloc_list.append({
                        "종목명": row['종목명'], "점수": row['점수'], "현재가": row['현재가'],
                        "배정 금액(KRW)": invest_krw, "추천 수량": qty,
                        "비고": "KRW 매수" if is_krw else f"환산 ${price_usd:.2f}"
                    })
                
                df_alloc = pd.DataFrame(alloc_list)
                st.markdown(f"#### 🛒 매수 추천 리스트")
                st.dataframe(df_alloc, hide_index=True, use_container_width=True)

with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오")
    st.caption("토스증권 요율 적용 (국내 세금0.15%+수수료 / 미국 수수료0.2%)")

    db = get_db()
    if not db: st.warning("⚠️ Firebase 설정 필요")
    else:
        col_u1, col_u2 = st.columns([1, 3])
        with col_u1: user_id = st.text_input("닉네임", value="장동진")
        doc_ref = db.collection('portfolios').document(user_id)
        try:
            doc = doc_ref.get()
            pf_data = doc.to_dict().get('stocks', []) if doc.exists else []
        except: pf_data = []

        with st.container():
            st.markdown("#### ➕ 종목 추가")
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1: selected_item = st.selectbox("종목 검색", ["선택하세요"] + SEARCH_LIST)
            with c2: input_price = st.number_input("내 평단가", min_value=0.0, format="%.2f")
            with c3: input_qty = st.number_input("보유 수량(주)", min_value=0, value=1)
            with c4:
                st.write("")
                st.write("")
                if st.button("추가하기", type="primary"):
                    if selected_item != "선택하세요":
                        target_code = SEARCH_MAP[selected_item]
                        new_pf_data = [p for p in pf_data if p['ticker'] != target_code]
                        new_pf_data.append({"ticker": target_code, "price": input_price, "qty": input_qty})
                        doc_ref.set({'stocks': new_pf_data})
                        st.success("추가 완료!")
                        time.sleep(0.5)
                        st.rerun()

        st.divider()

        if pf_data:
            st.markdown("#### ✏️ 보유 종목 수정")
            edit_options = [f"{TICKER_MAP.get(p['ticker'], p['ticker'])} ({p['ticker']})" for p in pf_data]
            selected_edit = st.selectbox("수정할 종목 선택", options=["선택하세요"] + edit_options, key="edit_select")

            if selected_edit != "선택하세요":
                edit_ticker = selected_edit.split("(")[-1].rstrip(")")
                target = next((p for p in pf_data if p["ticker"] == edit_ticker), None)
                if target:
                    new_avg = st.number_input("새 평단가", min_value=0.0, value=float(target["price"]), format="%.4f", key="edit_avg_price")
                    new_qty = st.number_input("새 보유 수량(주)", min_value=0, value=int(target.get("qty", 1)), key="edit_qty")
                    if st.button("변경 내용 저장", type="primary", key="edit_save"):
                        new_pf_data = []
                        for p in pf_data:
                            if p["ticker"] == edit_ticker: new_pf_data.append({"ticker": edit_ticker, "price": new_avg, "qty": new_qty})
                            else: new_pf_data.append(p)
                        doc_ref.set({"stocks": new_pf_data})
                        st.success("수정 완료!")
                        time.sleep(0.5)
                        st.rerun()
            st.divider()

        if pf_data:
            st.subheader(f"{user_id}님의 보유 종목 진단 (After-Market)")
            my_tickers = [p['ticker'] for p in pf_data]
            with st.spinner("초정밀 실시간 데이터 수집 중..."):
                raw_data_dict, realtime_map = get_precise_data(my_tickers)

            display_list = []
            for item in pf_data:
                tk = item['ticker']
                avg = item['price']
                qty = item.get('qty', 1)
                name = TICKER_MAP.get(tk, tk)
                curr = 0.0
                df_indi = None

                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                    if not df_tk.empty:
                        curr_price = realtime_map.get(tk)
                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)

                if df_indi is not None and not df_indi.empty:
                    curr = float(df_indi['Close_Calc'].iloc[-1])
                    cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)
                else:
                    curr = avg 
                    cat, col_name, reasoning, score = "로딩 중", "gray", "대기", 0.0

                if curr > 0:
                    res = calculate_total_profit(tk, avg, curr, qty)
                    display_list.append({
                        "name": name, "tk": tk, "avg": avg, "curr": curr, "qty": qty,
                        "cat": cat, "col_name": col_name, "reasoning": reasoning,
                        "profit_pct": res['pct'], "profit_amt": res['profit_amt'],
                        "eval_amt": res['net_eval_amt'], "currency": res['currency'],
                        "score": score
                    })

            display_list.sort(key=lambda x: x['score'], reverse=True)

            for item in display_list:
                with st.container():
                    c1, c2, c3 = st.columns([1.5, 1.5, 4])
                    sym = item['currency']
                    safe_sym = sym if sym != "$" else "&#36;"

                    with c1:
                        st.markdown(f"### {item['name']}")
                        st.caption(f"{item['tk']} | 보유: {item['qty']}주")

                    with c2:
                        fmt_curr = f"{item['curr']:,.0f}" if sym == "₩" else f"{item['curr']:,.2f}"
                        fmt_avg = f"{item['avg']:,.0f}" if sym == "₩" else f"{item['avg']:,.2f}"
                        fmt_eval = f"{item['eval_amt']:,.0f}" if sym == "₩" else f"{item['eval_amt']:,.2f}"
                        
                        # 수익금 색상 (한국형: 빨강=수익)
                        profit_color = "red" if item['profit_amt'] >= 0 else "blue"
                        
                        st.markdown(f"""
                        <div style='font-size: 24px; font-weight: bold; color: {profit_color};'>
                        {item['profit_pct']:.2f}% <br>
                        <span style='font-size: 16px;'>{safe_sym}{item['profit_amt']:,.0f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"실현예상금: {safe_sym}{fmt_eval}")
                        st.markdown(f"<small style='color: gray'>평단: {safe_sym}{fmt_avg} / 현재: {safe_sym}{fmt_curr}</small>", unsafe_allow_html=True)

                    with c3:
                        st.markdown(f"**AI 점수: {item['score']:.1f}점**")
                        st.markdown(f"**판단:** :{item['col_name']}[{item['cat']}]")
                        st.info(f"💡 {item['reasoning']}")
                    st.divider()

            if st.button("🗑️ 포트폴리오 전체 삭제"):
                doc_ref.delete()
                st.rerun()

with tab3:
    st.markdown("## 📘 AI 투자 전략 (Updated)")
    st.markdown("토스증권의 실제 수수료(국내 약 0.03%, 해외 약 0.2%)와 세금(국내 0.15%)을 반영하여 순수익을 계산합니다.")
