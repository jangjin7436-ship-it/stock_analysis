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
st.set_page_config(page_title="AI 스나이퍼 스캐너", page_icon="🎯", layout="wide")

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
# 2. 데이터 수집 (최적화: 단일/다중 종목 호환)
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
            except: 
                pass

            # Realtime
            try:
                sub_real = df_real[t]
                if isinstance(sub_real, pd.DataFrame) and not sub_real.empty:
                    sub_real = sub_real.dropna(how='all')
                    if 'Close' in sub_real.columns:
                        valid_closes = sub_real['Close'].dropna()
                        if not valid_closes.empty:
                            realtime_map[t] = float(valid_closes.iloc[-1])
            except: 
                pass
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
                if p: 
                    realtime_map[tk] = p
            except: 
                pass
            
        for f in concurrent.futures.as_completed(fut_hist):
            try:
                tk, df = f.result()
                if df is not None and not df.empty:
                    hist_map[tk] = df
            except: 
                pass

    return hist_map, realtime_map

# ---------------------------------------------------------
# 3. 분석 엔진 (AI Sniper Logic: 2주 스윙 최적화)
# ---------------------------------------------------------

def calculate_indicators(df, realtime_price=None):
    """
    [최적화] 2주 단기 스윙용 지표 (MA10, 볼린저밴드, 거래량 추가)
    """
    if df is None or len(df) < 60:
        return None

    df = df.copy()

    # 컬럼 통일
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    if 'Close' not in df.columns:
        return None

    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # 실시간 가격 주입
    if realtime_price is not None and realtime_price > 0:
        try:
            close.iloc[-1] = realtime_price
        except Exception:
            pass

    df['Close_Calc'] = close

    # 1. 이동평균 (MA10 - 스윙 생명선 추가)
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA10'] = df['Close_Calc'].rolling(10).mean() 
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()
    
    # 2. 볼린저 밴드 (변동성)
    std = df['Close_Calc'].rolling(20).std()
    df['Upper_Band'] = df['MA20'] + (std * 2)
    df['Lower_Band'] = df['MA20'] - (std * 2)
    df['Band_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
    
    # 3. RSI
    delta = df['Close_Calc'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. MACD
    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)
    
    # 5. 거래량
    if 'Volume' in df.columns:
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    else:
        df['Vol_Ratio'] = 1.0 

    df['STD20'] = std
    return df.dropna()

def get_ai_score_row(row):
    """
    [AI 스나이퍼 스코어링]
    - MA10 지지 & 모멘텀 가속 & 밴드 돌파 시 고득점
    """
    try:
        score = 50.0
        
        curr = row['Close_Calc']
        ma5, ma10, ma20, ma60 = row['MA5'], row['MA10'], row['MA20'], row['MA60']
        rsi = row['RSI']
        
        # 1. 추세 (10일선 생명선)
        if curr > ma10:
            score += 15.0
            if ma5 > ma10 > ma20: # 정배열
                score += 5.0
        else:
            score -= 10.0 # 탄력 둔화
            
        if curr > ma60:
            score += 5.0
        else:
            score -= 5.0

        # 2. 모멘텀 (MACD 가속)
        if row['MACD_Hist'] > 0:
            score += 5.0
            if row['MACD_Hist'] > row['Prev_MACD_Hist']:
                score += 5.0 # 상승 가속
        elif row['MACD_Hist'] > row['Prev_MACD_Hist'] and row['MACD_Hist'] > -0.5:
             score += 5.0 # 반등 시도

        # 3. RSI (스윙 적정 구간 50~70)
        if 50 <= rsi <= 70:
            score += 10.0
        elif rsi > 75:
            score -= 5.0 # 과열
        elif rsi < 35:
            score += 5.0 # 낙폭 과대

        # 4. 볼린저 밴드 (변동성)
        u_band = row['Upper_Band']
        if curr >= u_band * 0.98: # 밴드 상단 돌파 시도
            score += 10.0
            
        # 스퀴즈 후 발산
        if row['Band_Width'] < 0.15 and ma5 > ma10:
            score += 5.0

        # 5. 거래량
        if row['Vol_Ratio'] >= 1.2 and curr > row['MA5']:
             score += 5.0

        # 6. 안정성 페널티 (변동성 클수록 감점)
        vol_ratio = row['STD20'] / curr if curr > 0 else 0
        score -= (vol_ratio * 100.0)

        return max(0.0, min(100.0, score))
    except:
        return 0.0

def analyze_advanced_strategy(df):
    """
    [AI 스나이퍼 전략 등급 분류]
    - 조건 만족 전부 매수 모드: 70점 이상이면 모두 '매수' 등급 부여
    """
    if df is None or df.empty:
        return "분석 불가", "gray", "데이터 부족", 0.0

    try:
        row = df.iloc[-1]
        score = float(get_ai_score_row(row))
        curr = float(row['Close_Calc'])
        ma10 = float(row['MA10'])
        ma60 = float(row['MA60'])
        rsi = float(row['RSI'])
        macd_hist = float(row['MACD_Hist'])
        u_band = float(row['Upper_Band'])
    except Exception:
        return "오류", "gray", "계산 실패", 0.0

    reasons = []

    # 1. 10일선(생명선) 기준 판단
    if curr > ma10:
        reasons.append("10일선 위(강세)")
    else:
        reasons.append("10일선 이탈(약세)")

    # 2. 밴드/추세
    if curr >= u_band * 0.99:
        reasons.append("밴드 돌파(급등)")
    elif curr > ma60:
        reasons.append("장기 정배열")
    
    # 3. RSI
    if rsi > 75:
        reasons.append(f"RSI 과열({rsi:.0f})")
    elif rsi < 35:
        reasons.append(f"과매도({rsi:.0f})")
    
    # 4. MACD
    if macd_hist > 0 and macd_hist > row['Prev_MACD_Hist']:
        reasons.append("에너지 가속")

    # -----------------------------------------------------
    # [스나이퍼 전략] 점수 구간 → 매수/매도 등급 매핑
    # -----------------------------------------------------
    if score >= 70:
        # 70점 이상이면 무조건 매수 후보 (분산 투자 권장)
        if rsi > 75:
            cat = "🔥 매수 주의 (과열권)"
            col = "orange"
            reasons.insert(0, "단기 고점 위험")
        else:
            cat = "🎯 스나이퍼 매수 (진입 타점)"
            col = "green"

    elif score < 40:
        cat = "💥 매도/손절 (추세 이탈)"
        col = "red"
        
    elif 40 <= score < 50:
        cat = "📉 비중 축소 (약세)"
        col = "orange"
        
    else: # 50 ~ 69
        cat = "👀 관망 (Hold)"
        col = "blue"

    reasoning = " / ".join(reasons[:3])
    return cat, col, reasoning, round(score, 3)

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
st.title("🎯 AI 스나이퍼 스캐너 Pro")

tab1, tab2, tab3 = st.tabs(["🚀 AI 스나이퍼 스캔", "💼 내 포트폴리오", "📘 전략 백서"])

# TAB 1: 스캐너
with tab1:
    st.markdown("### 📋 AI 스나이퍼 종목 발굴")
    st.caption("전략: 2주 단기 스윙 | 선정: 조건 만족 종목 전부 매수 (분산 투자 권장)")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 시장 정밀 스캔", type="primary"):
            st.session_state['scan_result_df'] = None 
            st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 유니버스 분석 시작"):
            with st.spinner('AI 스나이퍼 알고리즘 가동 중... (10일선/볼린저/모멘텀 분석)'):
                raw_data_dict, realtime_map = get_precise_data(USER_WATCHLIST)
                scan_results = []
                progress_bar = st.progress(0)
                
                for i, ticker_code in enumerate(USER_WATCHLIST):
                    if ticker_code not in raw_data_dict: 
                        continue
                    try:
                        df_tk = raw_data_dict[ticker_code].dropna(how='all')
                        if df_tk.empty: 
                            continue
                        
                        curr_price = realtime_map.get(ticker_code)
                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)
                        
                        if df_indi is None: 
                            continue

                        # AI 스나이퍼 분석 실행
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
                    except: 
                        continue
                    progress_bar.progress((i + 1) / len(USER_WATCHLIST))
                
                if scan_results:
                    df_res = pd.DataFrame(scan_results)
                    # 점수 내림차순 정렬 (사용자가 보기 편하게)
                    df_res = df_res.sort_values('점수', ascending=False)
                    st.session_state['scan_result_df'] = df_res
                    st.success("스캔 완료! 70점 이상인 종목들을 확인하세요.")
                    st.rerun()
                else:
                    st.error("데이터 수집 실패.")
    
    if st.session_state['scan_result_df'] is not None:
        # 70점 이상 종목 개수 파악
        high_score_df = st.session_state['scan_result_df'][st.session_state['scan_result_df']['점수'] >= 70]
        count = len(high_score_df)
        
        if count > 0:
            st.info(f"✨ **매수 조건 만족 종목: {count}개** (AI 스나이퍼 기준 70점 이상)")
        else:
            st.warning("현재 매수 조건을 만족하는 종목이 없습니다. (관망 권장)")
        
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
        except: 
            pf_data = []

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
                    new_avg = st.number_input(
                        "새 평단가", 
                        min_value=0.0, 
                        value=float(target["price"]), 
                        format="%.4f", 
                        key="edit_avg_price"
                    )
                    new_qty = st.number_input(
                        "새 보유 수량(주)", 
                        min_value=0, 
                        value=int(target.get("qty", 1)), 
                        key="edit_qty"
                    )

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
                
                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                    if not df_tk.empty:
                        curr_price = realtime_map.get(tk)
                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)
                
                if df_indi is not None:
                    curr = float(df_indi['Close_Calc'].iloc[-1])
                    cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)
                else:
                    cat, col_name, reasoning, score = "데이터 로딩 중", "gray", "잠시 후 다시 시도", 0

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
            
            # 점수 순으로 정렬
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
                        
                        st.metric(
                            "총 순수익 (수수료 제)", 
                            f"{item['profit_pct']:.2f}%", 
                            delta=f"{sym}{item['profit_amt']:,.0f}" if sym=="₩" else f"{sym}{item['profit_amt']:,.2f}"
                        )
                        st.markdown(f"**세후 총 평가금:** {safe_sym}{fmt_eval}", unsafe_allow_html=True)
                        st.markdown(
                            f"<small style='color: gray'>평단: {safe_sym}{fmt_avg} / 현재: {safe_sym}{fmt_curr}</small>",
                            unsafe_allow_html=True
                        )
                        
                    with c3:
                        st.markdown(f"**AI 점수: {item['score']}점**")
                        st.markdown(f"**판단:** :{item['col_name']}[{item['cat']}]")
                        st.info(f"💡 {item['reasoning']}")
                    st.divider()

            if st.button("🗑️ 포트폴리오 전체 삭제"):
                doc_ref.delete()
                st.rerun()

with tab3:
    st.markdown("## 📘 AI 스나이퍼 전략 백서 (Sniper Mode v2.0)")
    st.markdown("""
    **'2주 단기 스윙'**에 최적화된 AI 스나이퍼 전략은 지지부진한 흐름을 배제하고, 
    **상승 탄력(Momentum)이 붙는 시점**과 **변동성이 폭발(Volatility Breakout)하는 시점**을 정밀 타격합니다.
    """)
    
    st.divider()
    
    st.subheader("1. 🎯 점수 & 등급 가이드")
    score_guide_data = [
        {"점수": "70점 이상", "등급": "🎯 스나이퍼 매수", "설명": "10일선 지지 + 거래량 실린 상승 + 밴드 돌파. 분산 매수 추천."},
        {"점수": "50~69점", "등급": "👀 관망 (Hold)", "설명": "추세는 살아있으나 모멘텀이 약함. 기존 보유자는 홀딩, 신규는 대기."},
        {"점수": "40점 미만", "등급": "💥 매도/손절", "설명": "10일선 붕괴 또는 추세 이탈. 리스크 관리(손절) 필수."},
        {"특이사항": "RSI > 75", "등급": "🔥 과열 경고", "설명": "단기 급등으로 인한 조정 가능성. 분할 익절 권장."}
    ]
    st.table(score_guide_data)

    st.header("2. 🧠 스나이퍼 핵심 3요소")
    
    with st.expander("① 10일선 생명선 매매 (Trend Line)", expanded=True):
        st.markdown("""
        스윙 트레이딩에서 20일선은 너무 느리고, 5일선은 너무 빠릅니다.
        **10일 이동평균선**은 2주간의 평균 가격으로, 단기 추세의 '생명선'입니다.
        * 주가가 10일선 위에 있어야만 점수를 부여하며, 깨지는 순간 매도 신호로 간주합니다.
        """)

    with st.expander("② 볼린저 밴드 스퀴즈 & 돌파 (Volatility)", expanded=True):
        st.markdown("""
        주가는 '휴식(수렴)' 후에 '폭발(발산)'합니다.
        * **스퀴즈(Squeeze):** 볼린저 밴드 폭이 좁아지며 에너지를 응축하는 구간에 가산점을 줍니다.
        * **돌파(Breakout):** 밴드 상단을 강하게 돌파할 때 추격 매수 신호를 포착합니다.
        """)

    with st.expander("③ MACD 가속도 (Momentum Acceleration)", expanded=True):
        st.markdown("""
        단순히 MACD가 양수라고 좋은 것이 아닙니다.
        * 어제보다 오늘의 막대(Histogram)가 더 커져야 **'상승 가속도'**가 붙은 것입니다.
        * 상승 힘이 줄어들면(막대가 작아지면) 선제적으로 점수를 차감하여 고점 판독을 돕습니다.
        """)
