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
# 2. 데이터 수집 (단일/다중 종목 호환)
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
            if not df_hist.empty and 'Close' in df_hist.columns:
                hist_map[ticker] = df_hist

            df_real = yf.download(ticker, period="5d", interval="1m", progress=False, prepost=True)
            if not df_real.empty and 'Close' in df_real.columns:
                last_p = float(df_real['Close'].iloc[-1])
                realtime_map[ticker] = last_p
        except Exception:
            pass
        return hist_map, realtime_map

    # 여러 개일 때 (Bulk)
    try:
        df_hist = yf.download(us_tickers, period="2y", interval="1d",
                              progress=False, group_by='ticker', auto_adjust=True)
        df_real = yf.download(us_tickers, period="5d", interval="1m",
                              progress=False, group_by='ticker', prepost=True)

        for t in us_tickers:
            # History
            try:
                sub_df = df_hist[t]
                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                    sub_df = sub_df.dropna(how='all')
                    if 'Close' in sub_df.columns:
                        hist_map[t] = sub_df
            except Exception:
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
            except Exception:
                pass
    except Exception:
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
    except Exception:
        return (ticker, None)

def fetch_kr_history(ticker):
    try:
        df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        return (ticker, df)
    except Exception:
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
            except Exception:
                pass
            
        for f in concurrent.futures.as_completed(fut_hist):
            try:
                tk, df = f.result()
                if df is not None and not df.empty:
                    hist_map[tk] = df
            except Exception:
                pass

    return hist_map, realtime_map

# ---------------------------------------------------------
# 3. 분석 엔진 (2주 스윙 백테스트 로직 이식)
# ---------------------------------------------------------

def calculate_indicators(df, realtime_price=None):
    """
    2주 스윙 백테스트 엔진과 동일한 방식의 지표 계산
    - Close/Adj Close 중 하나 사용
    - 실시간가가 들어오면 마지막 캔들에 반영 후 지표 계산
    """
    if df is None or len(df) < 60:
        return None

    df = df.copy()

    # yfinance & FDR 양쪽 호환: 우선 Adj Close, 없으면 Close 사용
    base_close = None
    if 'Adj Close' in df.columns:
        base_close = df['Adj Close']
    elif 'Close' in df.columns:
        base_close = df['Close']

    if base_close is None:
        return None

    # 멀티컬럼일 경우 첫 컬럼 사용
    if isinstance(base_close, pd.DataFrame):
        base_close = base_close.iloc[:, 0]

    # 실시간 가격 주입
    if realtime_price is not None and realtime_price > 0:
        try:
            base_close.iloc[-1] = realtime_price
        except Exception:
            pass

    df['Close_Calc'] = base_close

    # 이동평균
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()

    # RSI (14일)
    delta = df['Close_Calc'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12-26-9)
    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)

    # 20일 변동성
    df['STD20'] = df['Close_Calc'].rolling(20).std()

    # 2주(10영업일) 관점용 최근 5일 수익률
    df['Ret5'] = df['Close_Calc'].pct_change(5)

    return df.dropna()


def get_ai_score_row(row: pd.Series) -> float:
    """
    2주 스윙 기준 AI 점수:
    - 상승 추세 + 20일선 근처 눌림
    - 적당한 RSI 구간
    - 최근 5일 모멘텀
    - MACD 방향
    - 변동성 페널티
    (포트폴리오 백테스트 코드에서 그대로 가져온 버전)
    """
    try:
        curr = row['Close_Calc']
        ma5 = row['MA5']
        ma20 = row['MA20']
        ma60 = row['MA60']
        rsi = row['RSI']
        macd = row['MACD']
        sig = row['Signal_Line']
        macd_hist = row['MACD_Hist']
        prev_hist = row['Prev_MACD_Hist']
        std20 = row['STD20']
        ret5 = row.get('Ret5', 0.0)

        if curr <= 0 or ma20 <= 0 or ma60 <= 0:
            return 0.0

        score = 50.0

        # 1) 중·장기 추세 (MA20, MA60 기준)
        if curr > ma60 and ma20 > ma60:
            score += 15.0
            if ma5 > ma20:
                score += 5.0  # 5-20-60 정배열이면 가산
        else:
            score -= 15.0
            if curr < ma60:
                score -= 10.0

        # 2) 20일선과의 거리 (눌림 구간)
        dist20 = (curr - ma20) / ma20  # 비율
        abs_d20 = abs(dist20)

        # -2% ~ +3%: 최적 매수 존, 20점까지 가산 (0에 가까울수록 가장 좋음)
        if -0.02 <= dist20 <= 0.03:
            score += 20.0 * (1.0 - abs_d20 / 0.03)
        # -5% ~ -2%: 조금 깊은 눌림, 소폭 가산
        elif -0.05 <= dist20 < -0.02:
            score += 5.0
        # +8% 이상 이격: 단기 과열
        elif dist20 > 0.08:
            score -= min(20.0, (dist20 - 0.08) * 400)

        # 3) RSI (모멘텀 밸런스)
        if 40 <= rsi <= 60:
            score += 10.0
        elif 30 <= rsi < 40:
            score += 7.0
        elif 60 < rsi <= 70:
            score += 5.0
        elif rsi < 25 or rsi > 75:
            score -= 10.0

        # 4) 최근 5일 수익률 (2주 스윙용 단기 모멘텀)
        if ret5 is None:
            ret5 = 0.0
        if ret5 > 0:
            # 5일 +3%면 약 +6점
            score += min(7.0, float(ret5) * 100 * 2.0)
        else:
            # 하락이면 약하게 감점
            score += float(ret5) * 100.0 * 0.5

        # 5) MACD 방향 (상승 + 에너지 증가)
        if macd > sig and macd_hist > 0:
            score += 8.0
            if macd_hist > prev_hist:
                score += 4.0
        else:
            score -= 5.0

        # 6) 변동성 (안정성)
        vol_ratio = std20 / curr if curr > 0 else 0.0
        if vol_ratio > 0:
            if vol_ratio < 0.015:
                # 너무 안 움직이면(박스) 약간 감점
                score -= 2.0
            elif 0.015 <= vol_ratio <= 0.05:
                # 일간 1.5%~5% 정도를 이상적인 스윙 변동성으로 봄
                score += (0.05 - vol_ratio) * 200.0
            else:
                # 5% 이상은 리스크 크므로 강하게 감점
                score -= (vol_ratio - 0.05) * 300.0

        return max(0.0, min(100.0, float(score)))
    except Exception:
        return 0.0


def analyze_advanced_strategy(df):
    """
    스캐너에서 사용하는 '등급/설명/점수' 인터페이스는 유지하면서
    내부 점수는 2주 스윙 백테스트용 AI_Score와 완전히 동일하게 계산.
    
    + 매매 기준 해석은 'AI 스나이퍼 + 점수 1등만 매수' 기준으로 세팅:
      - Sniper 진입     : 점수 >= 70, Ret5 >= -2%
      - Basic 진입 하한 : 점수 >= 65
      - 기본 방어 매도  : 점수 < 45
      - 스나이퍼 추세 이탈: 점수 < 40
    """
    if df is None or df.empty:
        return "분석 불가", "gray", "데이터 부족", 0.0

    try:
        row = df.iloc[-1]
        score = float(get_ai_score_row(row))
        curr = float(row['Close_Calc'])
        ma20 = float(row['MA20'])
        ma60 = float(row['MA60'])
        rsi = float(row['RSI'])
        macd = float(row['MACD'])
        sig = float(row['Signal_Line'])
        ret5 = float(row.get('Ret5', 0.0))
    except Exception:
        return "오류", "gray", "계산 실패", 0.0

    reasons = []

    # 추세 설명
    if curr > ma60:
        reasons.append("상승 추세(60일선 위)")
    else:
        reasons.append("하락/조정 추세(60일선 아래)")

    # 20일선과의 거리(눌림목/과열)
    dist_ma20 = (curr - ma20) / ma20 if ma20 != 0 else 0.0
    if curr > ma60 and -0.03 <= dist_ma20 <= 0.03:
        reasons.append(f"20일선 근처 눌림목({dist_ma20*100:.1f}%)")
    elif dist_ma20 > 0.10:
        reasons.append("20일선 대비 과열(10%↑)")

    # RSI 상태
    if rsi < 30:
        reasons.append(f"RSI 과매도({rsi:.1f})")
    elif rsi > 70:
        reasons.append(f"RSI 과매수({rsi:.1f})")
    elif 40 <= rsi <= 60:
        reasons.append(f"안정적 RSI({rsi:.1f})")

    # MACD 방향
    if macd > sig:
        reasons.append("MACD 상승 에너지")
    else:
        reasons.append("MACD 약세/조정")

    # 최근 5일 모멘텀
    reasons.append(f"최근 5일 수익률 {ret5*100:.2f}%")

    # -----------------------------------------------------
    # 점수 구간 → 매수/매도 해석 (AI 스나이퍼 기준)
    # -----------------------------------------------------
    if score >= 80:
        cat = "🚀 강력 매수 (슈퍼 락킹 진입 구간)"
        col = "green"
    elif score >= 70:
        cat = "📈 매수 우위 (스나이퍼·기본 진입)"
        col = "blue"
    elif score >= 65:
        cat = "📈 약한 매수 (기본 진입 하한선)"
        col = "blue"
    elif score >= 45:
        cat = "👀 관망 (진입·청산 모두 보류)"
        col = "gray"
    elif score >= 40:
        cat = "📉 비중 축소 (기본 매도 후보)"
        col = "orange"
    else:
        cat = "💥 매도 (스나이퍼 추세 이탈)"
        col = "red"

    reasoning = " / ".join(reasons[:4])
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
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2, tab3 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)", "📘 알고리즘 백서"])

# TAB 1: 스캐너
with tab1:
    st.markdown("### 📋 AI 정밀 스캐너")
    st.caption("초정밀 실시간/AfterMarket 데이터 기반 AI 분석 (2주 스윙 · AI 스나이퍼 / 점수 1등 매수 기준)")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 분석 새로고침", type="primary"):
            st.session_state['scan_result_df'] = None 
            st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 리스트 정밀 분석 시작"):
            with st.spinner('초정밀 데이터 수집 및 분석 중...'):
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

                        # 🔥 2주 스윙 백테스트용 AI_Score/등급 사용
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
                    except Exception:
                        continue
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
        st.warning("⚠️ Firebase 설정 필요 (firebase_key 시크릿)")
    else:
        col_u1, col_u2 = st.columns([1, 3])
        with col_u1:
            user_id = st.text_input("닉네임", value="장동진")
        doc_ref = db.collection('portfolios').document(user_id)
        try:
            doc = doc_ref.get()
            pf_data = doc.to_dict().get('stocks', []) if doc.exists else []
        except Exception:
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
                
                # 데이터 유효성 검사 및 추출
                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                    if not df_tk.empty:
                        curr_price = realtime_map.get(tk)
                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)
                
                if df_indi is not None:
                    curr = float(df_indi['Close_Calc'].iloc[-1])
                
                # 🔥 여기서도 2주 스윙 백테스트용 AI_Score/등급 사용
                if df_indi is not None:
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
            
            # 점수 기준 정렬 (백테스트와 동일한 스코어 기반)
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

# TAB 3: 알고리즘 설명
with tab3:
    st.markdown("## 📘 AI 투자 전략 알고리즘 백서 (Whitepaper v2.0)")
    st.markdown("""
    본 서비스에 탑재된 AI 알고리즘은 월가(Wall St)의 퀀트 트레이딩에서 검증된 **'추세 추종(Trend Following)'** 전략과  
    단기 과매도 구간을 포착하는 **'평균 회귀(Mean Reversion)'** 이론을 정밀하게 결합한 하이브리드 모델입니다.
    
    현재 버전은 **2주 스윙 트레이딩용 AI 스나이퍼 전략**에 맞춰 튜닝되어 있으며,  
    모든 점수는 **0점(강력 매도) ~ 100점(강력 매수)** 사이의 실수(float)로 계산됩니다.
    """)

    st.divider()
    
    st.subheader("1. 🎯 AI 종합 점수 가이드 (Scoring Guide)")
    score_guide_data = [
        {"점수 구간": "80점 ~ 100점", "등급": "🚀 강력 매수 (Strong Buy)", "설명": "추세, 수급, 모멘텀 등이 모두 우수한 상태. 적극 진입 추천 구간."},
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
          단, 이격도가 너무 커지면(과열) 가산점이 줄어듭니다.
        * **정배열 강도:** `5일선 > 20일선 > 60일선` 순서로 정렬된 경우, 상승 에너지가 강한 상태로 판단하여 추가 점수를 부여합니다.
        * **역배열 감점:** 모든 이평선 아래에 주가가 위치하면 '하락장'으로 간주하여 강력한 페널티를 부과합니다.
        """)

    with st.expander("② 황금 눌림목 (The Golden Dip) - 고수익의 비밀", expanded=True):
        st.markdown("""
        **"무릎에 사서 어깨에 팔아라."**
        
        가장 높은 점수가 부여되는 핵심 구간입니다. 상승 추세(60일선 위)에 있는 종목이  
        일시적인 조정으로 **20일 이동평균선** 근처까지 눌렸을 때를 포착합니다.
        
        * **20일선 ±2~3% 이내:** 최적 매수 존, 최대 +20점 가산  
        * **20일선 대비 8~10% 이상 이격:** 단기 과열로 판단하여 강한 감점
        """)

    with st.expander("③ RSI (상대강도지수) - 투자 심리 역이용", expanded=True):
        st.markdown("""
        **"공포에 사고 탐욕에 팔아라."**
        
        RSI는 현재 시장의 과열/침체 정도를 0~100 사이 숫자로 나타냅니다.
        
        * **RSI 40~60:** 가장 건전한 상승 구간 → 가산점  
        * **RSI < 25:** 과매도, 기술적 반등 가능성 → 상황에 따라 소폭 가산 또는 관망  
        * **RSI > 75:** 과매수, 조정 가능성 → 감점
        """)

    with st.expander("④ MACD & 모멘텀 - 상승의 속도", expanded=True):
        st.markdown("""
        이동평균선이 '방향'을 알려준다면, MACD는 '속도'를 알려줍니다.
        
        * **MACD > Signal & 히스토그램 > 0:** 상승 에너지 유입 → 가산점  
        * **히스토그램 증가:** 상승 가속도 증가 → 추가 가산  
        * **반대로 MACD가 시그널 아래로 내려가거나 음수 전환:** 하락/조정으로 감점
        """)

    with st.expander("⑤ 변동성 (Volatility) - 위험 관리", expanded=True):
        st.markdown("""
        변동성이 너무 큰 주식은 '도박'에 가깝습니다.
        
        * **일간 표준편차 / 가격 비율(STD20 / Close):**  
          - 1.5%~5%: 이상적인 스윙 변동성 → 가산  
          - 5%↑: 고위험 존 → 강한 감점  
          - 너무 안 움직이는 종목(1.5% 미만)은 '박스권'으로 소폭 감점
        """)

    st.divider()
    st.info("💡 **Tip:** 현재 스캐너는 'AI 스나이퍼 + 점수 1등만 매수' 시나리오를 기준으로 설계되었습니다. "
            "실제 매매에서는 상위 1~3개 종목만 골라 차트/호가/뉴스를 함께 검토하는 것을 추천합니다.")  
