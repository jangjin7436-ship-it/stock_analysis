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
# 2. 데이터 수집 (수정됨: 단일/다중 종목 완벽 호환)
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_bulk_us_data(us_tickers):
    """미국 주식 데이터 수집"""
    if not us_tickers:
        return {}, {}
    
    hist_map = {}
    realtime_map = {}

    # 1개일 때
    if len(us_tickers) == 1:
        ticker = us_tickers[0]
        try:
            df_hist = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            if not df_hist.empty:
                if 'Close' in df_hist.columns:
                    hist_map[ticker] = df_hist

            df_real = yf.download(ticker, period="5d", interval="1m", progress=False, prepost=True)
            if not df_real.empty:
                if 'Close' in df_real.columns:
                    last_p = float(df_real['Close'].iloc[-1])
                    realtime_map[ticker] = last_p
        except:
            pass
        return hist_map, realtime_map

    # 여러 개일 때 (Bulk)
    try:
        df_hist = yf.download(us_tickers, period="2y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
        df_real = yf.download(us_tickers, period="5d", interval="1m", progress=False, group_by='ticker', prepost=True)

        for t in us_tickers:
            # History
            try:
                sub_df = df_hist[t]
                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                    sub_df = sub_df.dropna(how='all') 
                    if 'Close' in sub_df.columns:
                        hist_map[t] = sub_df
            except: pass

            # Realtime
            try:
                sub_real = df_real[t]
                if isinstance(sub_real, pd.DataFrame) and not sub_real.empty:
                      sub_real = sub_real.dropna(how='all')
                      if 'Close' in sub_real.columns:
                        valid_closes = sub_real['Close'].dropna()
                        if not valid_closes.empty:
                            realtime_map[t] = float(valid_closes.iloc[-1])
            except: pass
    except:
        pass

    return hist_map, realtime_map

def fetch_kr_polling(ticker):
    """국내 주식 실시간 (네이버)"""
    code = ticker.split('.')[0]
    try:
        url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=2)
        data = res.json()
        item = data['datas'][0]
        
        close = float(str(item['closePrice']).replace(',', ''))
        
        over_info = item.get('overMarketPriceInfo', {})
        over_price_str = str(over_info.get('overPrice', '')).replace(',', '').strip()
        if over_price_str and over_price_str != '0':
            return (ticker, float(over_price_str))
            
        return (ticker, close)
    except:
        return (ticker, None)

def fetch_kr_history(ticker):
    try:
        df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        return (ticker, df)
    except:
        return (ticker, None)

@st.cache_data(ttl=0)
def get_precise_data(tickers_list):
    """통합 데이터 수집기"""
    if not tickers_list:
        return {}, {}
        
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]

    # 1. 미국 주식
    hist_map, realtime_map = get_bulk_us_data(us_tickers)

    # 2. 국내 주식
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fut_real = [executor.submit(fetch_kr_polling, t) for t in kr_tickers]
        fut_hist = [executor.submit(fetch_kr_history, t) for t in kr_tickers]

        for f in concurrent.futures.as_completed(fut_real):
            try:
                tk, p = f.result()
                if p: realtime_map[tk] = p
            except: pass
            
        for f in concurrent.futures.as_completed(fut_hist):
            try:
                tk, df = f.result()
                if df is not None and not df.empty:
                    hist_map[tk] = df
            except: pass

    return hist_map, realtime_map

# ---------------------------------------------------------
# 3. 분석 엔진 (점수 로직 대폭 수정됨)
# ---------------------------------------------------------

def calculate_indicators(df, realtime_price=None):
    if df is None or len(df) < 30: return None
    df = df.copy()

    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    
    if 'Close' not in df.columns: return None
    
    close = df['Close']
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    
    # 🌟 [중요] 실시간 가격 주입
    if realtime_price is not None and realtime_price > 0:
        close.iloc[-1] = realtime_price

    df['Close_Calc'] = close

    # 지표 계산
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA10'] = df['Close_Calc'].rolling(10).mean()
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()
    
    # RSI
    delta = df['Close_Calc'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # MOM
    df['MOM10'] = df['Close_Calc'].pct_change(10)

    # Volume
    df['STD20'] = df['Close_Calc'].rolling(20).std()
    
    return df.dropna()

def analyze_advanced_strategy(df):
    """
    수정된 상세 채점 로직 (Granular Scoring System)
    - 단순 가산(+10)이 아닌, 지표의 강도와 근접도에 따라 소수점 점수 부여
    - 중복 점수 최소화
    """
    if df is None or df.empty:
        return "분석 불가", "gray", "데이터 부족", 0

    try:
        curr = float(df['Close_Calc'].iloc[-1])
        ma5 = float(df['MA5'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        ma60 = float(df['MA60'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        sig = float(df['Signal_Line'].iloc[-1])
        std20 = float(df['STD20'].iloc[-1])
        mom10 = float(df['MOM10'].iloc[-1])
    except:
        return "오류", "gray", "계산 실패", 0

    # 🟢 기본 점수: 50.0 (Float로 시작)
    score = 50.0
    reasons = []

    # -----------------------------------------------------------
    # 1. 추세 (Trend) - 정밀 평가 (Max +35점)
    # -----------------------------------------------------------
    # 60일선 위 (기본 추세)
    if curr > ma60:
        score += 10
        # 추세의 강도: 이격도가 너무 크지 않은 선에서 가산점 (최대 +5점)
        divergence_60 = (curr - ma60) / ma60
        if 0 < divergence_60 < 0.15:
            score += divergence_60 * 33  # 0.15 * 33 ≈ 5점
        else:
            score += 2 # 너무 멀어지면 조금만
    else:
        score -= 20 # 역배열 감점
        
    # 정배열 보너스 (5 > 20 > 60)
    if ma5 > ma20 > ma60:
        score += 10
        reasons.append("⚡ 정배열 상승세")
    elif ma20 > ma60:
        score += 5
        
    # -----------------------------------------------------------
    # 2. 위치 & 눌림목 (Position) - 거리 기반 가변 점수 (Max +30점)
    # -----------------------------------------------------------
    # 20일선과의 거리 계산 (비율)
    dist_ma20 = (curr - ma20) / ma20 
    abs_dist = abs(dist_ma20)

    # 황금 눌림목: 60일선 위에 있으면서, 20일선 근처(+/- 3%)에 붙어있을 때
    if curr > ma60 and abs_dist <= 0.03:
        # 거리가 0에 가까울수록 점수가 높음 (최대 20점)
        # 공식: 20 * (1 - (현재거리 / 허용거리))
        proximity_score = 20 * (1 - (abs_dist / 0.03))
        score += proximity_score
        
        if dist_ma20 >= 0:
            reasons.append(f"💎 황금 눌림목 (20일선 +{dist_ma20*100:.1f}%)")
        else:
            reasons.append(f"🛒 저점 매수 기회 (20일선 {dist_ma20*100:.1f}%)")
            
    # 상승 지속형 (20일선 위 3%~8%)
    elif curr > ma60 and 0.03 < dist_ma20 <= 0.08:
        score += 5
        
    # 과열 주의 (20일선 10% 이상 이격)
    elif dist_ma20 > 0.10:
        score -= 15
        reasons.append("🔥 단기 과열 (이격 과다)")

    # -----------------------------------------------------------
    # 3. RSI 정밀 평가 - 곡선형 점수 (Max +15점)
    # -----------------------------------------------------------
    # 40~60: 안정적 상승 구간 (가장 선호)
    if 40 <= rsi <= 60:
        # 50을 중심으로 대칭 점수 부여 (중립일 때 +10, 60에 가까우면 +12)
        score += 10 + ((rsi - 40) * 0.1)
        reasons.append(f"⚖ 안정적 RSI ({rsi:.1f})")
    # 30~40: 반등 준비 구간
    elif 30 <= rsi < 40:
        score += 5 + ((40 - rsi) * 0.5) # 30에 가까울수록 점수 높게
        reasons.append("반등 준비")
    # 60~70: 강한 모멘텀
    elif 60 < rsi <= 70:
        score += 8
    # 과매도 (30 미만) - 역발상
    elif rsi < 30:
        score += 15
        reasons.append("💧 과매도 (기술적 반등 기대)")
    # 과매수 (70 초과)
    elif rsi > 70:
        score -= 15
        reasons.append("🚨 RSI 과열")

    # -----------------------------------------------------------
    # 4. MACD & 모멘텀 (Max +20점)
    # -----------------------------------------------------------
    # MACD 오실레이터(히스토그램) 크기에 비례한 점수
    macd_hist = macd - sig
    
    if macd > sig:
        score += 5
        # 히스토그램이 양수이면서 커질수록 가산점 (최대 5점)
        hist_bonus = min(5.0, (macd_hist / curr) * 1000)
        score += hist_bonus
        if macd_hist > 0 and macd_hist > float(df['MACD'].iloc[-2] - df['Signal_Line'].iloc[-2]):
             reasons.append("🚀 상승 에너지 확대")
    else:
        score -= 5

    # -----------------------------------------------------------
    # 5. 최종 보정
    # -----------------------------------------------------------
    # 변동성 페널티 (너무 등락폭이 크면 감점)
    vol_ratio = std20 / curr if curr > 0 else 0
    if vol_ratio > 0.05:
        score -= (vol_ratio * 100) # 변동성이 클수록 점수 깎임
        
    score = max(0.0, min(100.0, score)) # 0~100 사이로 클램핑

    # 등급 결정
    if score >= 80: cat, col = "🚀 강력 매수", "green"
    elif score >= 65: cat, col = "📈 매수 우위", "blue"
    elif score >= 45: cat, col = "👀 관망", "gray"
    elif score >= 25: cat, col = "📉 비중 축소", "orange"
    else: cat, col = "💥 매도", "red"

    if not reasons: reasons.append("중립/관망")
    
    # 팁: 리턴할 때 소수점 1자리까지 포함하여 리턴 -> 순위 정렬 시 동점자 방지
    return cat, col, " / ".join(reasons[:3]), round(score, 1)

def calculate_total_profit(ticker, avg_price, current_price, quantity):
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
# 4. UI
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2, tab3 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)", "📘 알고리즘 백서"])

# TAB 1: 스캐너
with tab1:
    st.markdown("### 📋 AI 정밀 스캐너")
    st.caption("초정밀 실시간/AfterMarket 데이터 기반 AI 분석")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 분석 새로고침", type="primary"):
            st.session_state['scan_result_df'] = None 
            st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 리스트 정밀 분석 시작"):
            with st.spinner('초정밀 데이터 수집 및 분석 중... (15~20초 소요)'):
                raw_data_dict, realtime_map = get_precise_data(USER_WATCHLIST)
                scan_results = []
                progress_bar = st.progress(0)
                
                for i, ticker_code in enumerate(USER_WATCHLIST):
                    if ticker_code not in raw_data_dict: continue
                    try:
                        df_tk = raw_data_dict[ticker_code].dropna(how='all')
                        if df_tk.empty: continue
                        
                        curr_price = realtime_map.get(ticker_code)
                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)
                        
                        if df_indi is None: continue

                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)

                        final_price = float(df_indi['Close_Calc'].iloc[-1])
                        rsi_val = float(df_indi['RSI'].iloc[-1])
                        name = TICKER_MAP.get(ticker_code, ticker_code)
                        is_kr = ticker_code.endswith(".KS") or ticker_code.endswith(".KQ")
                        sym = "₩" if is_kr else "$"
                        fmt_price = f"{sym}{final_price:,.0f}" if is_kr else f"{sym}{final_price:,.2f}"

                        scan_results.append({
                            "종목명": f"{name} ({ticker_code})",
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
                    df_res = df_res.sort_values('점수', ascending=False)
                    st.session_state['scan_result_df'] = df_res
                    st.success("완료! (결과는 '분석 새로고침' 전까지 고정됩니다)")
                    st.rerun()
                else:
                    st.error("데이터 수집 실패.")
    
    if st.session_state['scan_result_df'] is not None:
        st.dataframe(
            st.session_state['scan_result_df'],
            use_container_width=True,
            height=700,
            column_config={
                "종목명": st.column_config.TextColumn("종목명 (코드)", width="medium"),
                "점수": st.column_config.ProgressColumn("AI 점수", format="%.1f점", min_value=0, max_value=100),
                "현재가": st.column_config.TextColumn("현재가"), 
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "AI 등급": st.column_config.TextColumn("AI 판단"),
                "핵심 요약": st.column_config.TextColumn("분석 내용", width="large"),
            },
            hide_index=True
        )

# TAB 2: 포트폴리오
with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오")
    st.caption("네이버페이(국내) / 1분봉(해외) 실시간 기반 | 세후 순수익 계산")
    
    db = get_db()
    if not db:
        st.warning("⚠️ Firebase 설정 필요")
    else:
        col_u1, col_u2 = st.columns([1, 3])
        with col_u1:
            user_id = st.text_input("닉네임", value="장동진")
        doc_ref = db.collection('portfolios').document(user_id)
        try:
            doc = doc_ref.get()
            pf_data = doc.to_dict().get('stocks', []) if doc.exists else []
        except: pf_data = []

        with st.container():
            st.markdown("#### ➕ 종목 추가")
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                selected_item = st.selectbox("종목 검색", ["선택하세요"] + SEARCH_LIST)
            with c2:
                input_price = st.number_input("내 평단가", min_value=0.0, format="%.2f")
            with c3:
                input_qty = st.number_input("보유 수량(주)", min_value=0, value=1)
            with c4:
                st.write("")
                st.write("")
                if st.button("추가하기", type="primary"):
                    if selected_item != "선택하세요":
                        target_code = SEARCH_MAP[selected_item]
                        new_pf_data = [p for p in pf_data if p['ticker'] != target_code]
                        new_pf_data.append({
                            "ticker": target_code, 
                            "price": input_price,
                            "qty": input_qty
                        })
                        doc_ref.set({'stocks': new_pf_data})
                        st.success("추가 완료!")
                        time.sleep(0.5)
                        st.rerun()

        st.divider()

        if pf_data:
            # 수정 섹션
            st.markdown("#### ✏️ 보유 종목 정보 수정")
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
                            if p["ticker"] == edit_ticker:
                                new_pf_data.append({"ticker": edit_ticker, "price": new_avg, "qty": new_qty})
                            else:
                                new_pf_data.append(p)
                        doc_ref.set({"stocks": new_pf_data})
                        st.success("수정 완료!")
                        time.sleep(0.5)
                        st.rerun()

            st.divider()
        
        if pf_data:
            st.subheader(f"{user_id}님의 보유 종목 진단")
            my_tickers = [p['ticker'] for p in pf_data]
            with st.spinner("초정밀 실시간 데이터 수집 중..."):
                raw_data_dict, realtime_map = get_precise_data(my_tickers)
            
            display_list = []
            for item in pf_data:
                tk = item['ticker']
                avg = item['price']
                qty = item.get('qty', 1)
                name = TICKER_MAP.get(tk, tk)
                
                curr = 0
                df_indi = None
                
                # 데이터 유효성 검사 및 추출
                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                    if not df_tk.empty:
                        curr_price = realtime_map.get(tk)
                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)
                
                if df_indi is not None:
                      curr = float(df_indi['Close_Calc'].iloc[-1])
                
                cat, col_name, reasoning, score = "데이터 로딩 중", "gray", "잠시 후 다시 시도", 0

                if df_indi is not None:
                    cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)
                
                if curr > 0:
                    res = calculate_total_profit(tk, avg, curr, qty)
                    display_list.append({
                        "name": name, "tk": tk, "avg": avg, "curr": curr, "qty": qty,
                        "cat": cat, "col_name": col_name, "reasoning": reasoning,
                        "profit_pct": res['pct'], "profit_amt": res['profit_amt'],
                        "eval_amt": res['net_eval_amt'], "currency": res['currency'], "score": score
                    })
                else:
                    display_list.append({
                        "name": name, "tk": tk, "avg": avg, "curr": avg, "qty": qty,
                        "cat": "로딩 실패", "col_name": "gray", "reasoning": "데이터 수신 불가",
                        "profit_pct": 0.0, "profit_amt": 0.0, "eval_amt": 0.0,
                        "currency": "$" if not tk.endswith(".KS") else "₩", "score": 0
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
                        fmt_curr = f"{item['curr']:,.0f}" if item['currency'] == "₩" else f"{item['curr']:,.2f}"
                        fmt_avg  = f"{item['avg']:,.0f}"  if item['currency'] == "₩" else f"{item['avg']:,.2f}"
                        fmt_eval = f"{item['eval_amt']:,.0f}"   if item['currency'] == "₩" else f"{item['eval_amt']:,.2f}"
                        
                        st.metric("총 순수익 (수수료 제)", f"{item['profit_pct']:.2f}%", delta=f"{sym}{item['profit_amt']:,.0f}" if sym=="₩" else f"{sym}{item['profit_amt']:,.2f}")
                        st.markdown(f"**세후 총 평가금:** {safe_sym}{fmt_eval}", unsafe_allow_html=True)
                        st.markdown(f"<small style='color: gray'>평단: {safe_sym}{fmt_avg} / 현재: {safe_sym}{fmt_curr}</small>", unsafe_allow_html=True)
                        
                    with c3:
                        st.markdown(f"**AI 점수: {item['score']}점**")
                        st.markdown(f"**판단:** :{item['col_name']}[{item['cat']}]")
                        st.info(f"💡 {item['reasoning']}")
                    st.divider()

            if st.button("🗑️ 포트폴리오 전체 삭제"):
                doc_ref.delete()
                st.rerun()

with tab3:
    st.markdown("## 📘 AI 투자 전략 알고리즘 백서 (Whitepaper v2.0)")
    st.markdown("""
    본 서비스에 탑재된 AI 알고리즘은 월가(Wall St)의 퀀트 트레이딩에서 검증된 **'추세 추종(Trend Following)'** 전략과 
    단기 과매도 구간을 포착하는 **'평균 회귀(Mean Reversion)'** 이론을 정밀하게 결합한 하이브리드 모델입니다.
    
    모든 점수는 **0점(강력 매도) ~ 100점(강력 매수)** 사이의 실수(float)로 계산되며, 
    단순한 조건 매칭이 아닌 **지표의 강도(Strength)와 이격도(Divergence)**를 미분적으로 분석하여 산출됩니다.
    """)
    
    st.divider()
    
    st.subheader("1. 🎯 AI 종합 점수 가이드 (Scoring Guide)")
    score_guide_data = [
        {"점수 구간": "80점 ~ 100점", "등급": "🚀 강력 매수 (Strong Buy)", "설명": "추세, 수급, 모멘텀이 모두 완벽한 상태. 적극 진입 추천 구간."},
        {"점수 구간": "65점 ~ 80점", "등급": "📈 매수 우위 (Buy)", "설명": "상승 추세가 확연하며 진입 근거가 충분함. 분할 매수 유효."},
        {"점수 구간": "45점 ~ 65점", "등급": "👀 관망 (Hold/Neutral)", "설명": "방향성이 불분명하거나 상승 후 쉬어가는 구간. 신규 진입 자제."},
        {"점수 구간": "25점 ~ 45점", "등급": "📉 비중 축소 (Sell)", "설명": "하락 전환 신호 발생 또는 단기 과열 징후. 이익 실현 권장."},
        {"점수 구간": "0점 ~ 25점", "등급": "💥 강력 매도 (Strong Sell)", "설명": "위험 신호 감지. 역배열 하락 추세 또는 극심한 과매수."}
    ]
    st.table(score_guide_data)

    st.header("2. 🧠 핵심 평가 로직 (5-Factor Deep Dive)")
    st.markdown("AI는 다음 5가지 핵심 요소를 수치화하여 미세한 점수 차이를 만들어냅니다.")

    with st.expander("① 추세 (Trend Hierarchy) - 주가의 '생명선'", expanded=True):
        st.markdown("""
        **"추세는 당신의 친구입니다 (Trend is your friend)."**
        
        AI는 이동평균선(Moving Average)의 배열 상태를 통해 주가의 현재 위치를 파악합니다.
        
        * **장기 추세 (60일선):** 주가의 '계절'을 의미합니다. 60일선 위에 있다는 것은 현재가 '여름(상승장)'임을 뜻합니다. 
            * 로직: 주가가 60일선보다 높을수록 점수가 상승하지만, 이격도(거리)가 15%를 넘어가면 가산점이 제한됩니다.
        * **정배열 가산점 (Golden Alignment):** `5일선 > 20일선 > 60일선` 순서로 정렬된 경우, 상승 에너지가 가장 강한 상태로 판단하여 추가 점수(+10점)를 부여합니다.
        * **역배열 감점 (Death Alignment):** 모든 이평선 아래에 주가가 위치하면 '하락장'으로 간주하여 강력한 페널티(-20점)를 부과합니다.
        """)
        [Image of stock market moving average chart]

    with st.expander("② 황금 눌림목 (The Golden Dip) - 고수익의 비밀", expanded=True):
        st.markdown("""
        **"무릎에 사서 어깨에 팔아라."**
        
        가장 높은 점수가 부여되는 핵심 구간입니다. 상승 추세(60일선 위)에 있는 종목이 
        일시적인 차익 실현 매물로 인해 **20일 이동평균선(생명선)** 근처까지 내려왔을 때를 포착합니다.
        
        * **초정밀 거리 계산:** 현재 주가와 20일선 사이의 거리가 `0`에 가까울수록 점수는 기하급수적으로 상승합니다.
        * **매수 타이밍:** 20일선 위 `+3%` 이내에 접근했을 때가 가장 이상적인 매수 타이밍입니다. (+20점 만점)
        * **과열 경고:** 반대로 20일선과 `10%` 이상 벌어지면 '단기 과열'로 판단하여 점수를 깎습니다.
        """)

    with st.expander("③ RSI (상대강도지수) - 투자 심리 역이용", expanded=True):
        st.markdown("""
        **"공포에 사고 탐욕에 팔아라."**
        
        RSI는 현재 시장의 과열/침체 정도를 0~100 사이 숫자로 나타냅니다.
        
        * **안정적 상승 (RSI 40~60):** 가장 건전한 상승 구간입니다. 이 구간에서는 RSI가 60에 가까워질수록 가산점을 줍니다.
        * **과매도 역발상 (RSI < 30):** 남들이 공포에 질려 투매할 때입니다. AI는 이를 '기술적 반등 기회'로 포착하여 큰 가산점(+15점)을 줍니다.
        * **과매수 경고 (RSI > 70):** 매수세가 너무 강해 곧 조정이 올 수 있습니다. AI는 이를 위험 신호로 보고 감점합니다.
        """)

    with st.expander("④ MACD & 모멘텀 - 상승의 속도", expanded=True):
        st.markdown("""
        이동평균선이 '방향'을 알려준다면, MACD는 '속도'를 알려줍니다.
        
        * **MACD 오실레이터:** 막대그래프(Histogram)가 양수(+)이면서 점점 커지고 있다면 상승 가속도가 붙고 있다는 뜻입니다.
        * **골든크로스:** MACD 선이 시그널 선을 돌파하는 순간은 강력한 매수 신호로 간주합니다.
        * **모멘텀 가속:** 최근 10일간의 주가 상승률(Momentum)을 분석하여, 너무 급격하게 오르지도, 내리지도 않는 적절한 상승 각도를 찾습니다.
        """)

    with st.expander("⑤ 변동성 (Volatility) - 위험 관리", expanded=True):
        st.markdown("""
        변동성이 너무 큰 주식은 '도박'에 가깝습니다.
        
        * **변동성 페널티:** 표준편차(STD)를 분석하여, 하루 등락폭이 지나치게 큰 종목은 점수를 깎아 안정성을 확보합니다.
        * 이는 급등주나 작전주에 잘못 진입하여 큰 손실을 보는 것을 방지하는 안전장치입니다.
        """)
        
    st.divider()
    st.info("💡 **Tip:** 본 알고리즘은 '추세가 살아있는 종목'이 '일시적으로 눌렸을 때' 잡는 것을 최우선 목표로 합니다. 점수가 높더라도 본인의 투자 원칙과 병행하여 사용하시기 바랍니다.")
