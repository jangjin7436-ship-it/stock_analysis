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
# 2. 데이터 수집 (수정됨: 단일/다중 종목 완벽 호환)
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_bulk_us_data(us_tickers):
    """미국 주식 데이터 수집 (단일/다중 모두 동일 로직으로 처리)"""
    if not us_tickers:
        return {}, {}

    hist_map = {}
    realtime_map = {}

    try:
        # tickers가 1개여도 리스트로 주면 MultiIndex 구조가 나오므로
        # 항상 같은 경로를 타게 한다.
        df_hist = yf.download(
            us_tickers,
            period="2y",
            interval="1d",
            progress=False,
            group_by="ticker",
            auto_adjust=True,
        )
        df_real = yf.download(
            us_tickers,
            period="5d",
            interval="1m",
            progress=False,
            group_by="ticker",
            prepost=True,
        )

        # 컬럼이 MultiIndex인지(여러 종목) 아닌지에 따라 안전하게 처리
        hist_is_multi = isinstance(df_hist.columns, pd.MultiIndex)
        real_is_multi = isinstance(df_real.columns, pd.MultiIndex)

        for t in us_tickers:
            # ---------- 일봉 히스토리 ----------
            try:
                sub_df = df_hist[t] if hist_is_multi else df_hist
                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                    sub_df = sub_df.dropna(how="all")
                    # Close 또는 Adj Close 중 하나만 있어도 히스토리로 인정
                    if "Close" in sub_df.columns or "Adj Close" in sub_df.columns:
                        hist_map[t] = sub_df
            except Exception:
                pass

            # ---------- 실시간/최근 체결가 ----------
            try:
                sub_real = df_real[t] if real_is_multi else df_real
                if isinstance(sub_real, pd.DataFrame) and not sub_real.empty:
                    sub_real = sub_real.dropna(how="all")

                    price_series = None
                    if "Close" in sub_real.columns:
                        price_series = sub_real["Close"]
                    elif "Adj Close" in sub_real.columns:
                        price_series = sub_real["Adj Close"]

                    if price_series is not None:
                        valid_closes = price_series.dropna()
                        if not valid_closes.empty:
                            realtime_map[t] = float(valid_closes.iloc[-1])
            except Exception:
                pass

    except Exception:
        # yfinance 쪽 에러는 조용히 무시 (호출 쪽에서 로딩 실패 처리)
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
            return ticker, float(over_price_str)

        return ticker, close
    except Exception:
        return ticker, None


def fetch_kr_history(ticker):
    try:
        df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        return ticker, df
    except Exception:
        return ticker, None


@st.cache_data(ttl=0)
def get_precise_data(tickers_list):
    """통합 데이터 수집기"""
    if not tickers_list:
        return {}, {}

    # 국내 / 해외 분리
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]

    # 1. 미국(및 yfinance로 받는 나머지) 주식 히스토리 + 1분 데이터
    hist_map, realtime_map = get_bulk_us_data(us_tickers)

    # 2. 국내 주식 (네이버 실시간 + FDR 히스토리)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        fut_real = [executor.submit(fetch_kr_polling, t) for t in kr_tickers]
        fut_hist = [executor.submit(fetch_kr_history, t) for t in kr_tickers]

        # ----- (1) 네이버 실시간: 3초 타임아웃 -----
        for f in fut_real:
            try:
                tk, p = f.result(timeout=3)  # ★ 여기서 오래 걸리면 3초 후 버리고 넘어감
                if p:
                    realtime_map[tk] = p
            except concurrent.futures.TimeoutError:
                # 이 티커는 네이버 응답 너무 느림 → 그냥 무시
                continue
            except Exception:
                continue

        # ----- (2) FDR 히스토리: 5초 타임아웃 -----
        for f in fut_hist:
            try:
                tk, df = f.result(timeout=5)  # ★ 여기도 5초 후 포기
                if df is not None and not df.empty:
                    hist_map[tk] = df
            except concurrent.futures.TimeoutError:
                # 특정 종목 DataReader가 안 끝나는 경우 → 이 종목만 스킵
                continue
            except Exception:
                continue

    return hist_map, realtime_map

# ---------------------------------------------------------
# 3. 분석 엔진 (백테스트와 완전 동일한 지표/점수 로직)
# ---------------------------------------------------------

def calculate_indicators(df, realtime_price=None):
    """
    2주 스윙 백테스트에서 쓰는 지표 계산 로직 그대로 사용
    - Close/Adj Close 정리 (대소문자/형태 상관 없이 탐색)
    - 실시간가(realtime_price)가 들어오면 마지막 캔들 가격만 교체
    - MA5/20/60, RSI, MACD, MACD_Hist, Prev_MACD_Hist, STD20, Ret5 모두 계산
    """
    if df is None or len(df) < 60:
        return None

    # 혹시 Series가 들어오면 DataFrame으로 변환
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.copy()

    # --- 1) Close / Adj Close 컬럼 찾기 (모든 타입 안전 처리) ---
    close_col = None
    adj_close_col = None

    for c in df.columns:
        name = str(c).lower()
        if name == "close":
            close_col = c
        elif name in ("adj close", "adjclose", "adj_close"):
            adj_close_col = c

    if close_col is None and adj_close_col is not None:
        close_col = adj_close_col

    if close_col is None:
        # Close 계열 컬럼을 못 찾으면 지표 계산 불가
        return None

    close = df[close_col]
    if isinstance(close, pd.DataFrame):
        # yfinance 멀티컬럼 방지: 첫 컬럼만 사용
        close = close.iloc[:, 0]

    close = close.astype(float)

    # --- 2) 실시간 가격 주입 (가능하면 마지막 캔들 교체) ---
    if realtime_price is not None:
        try:
            rp = float(realtime_price)
            if rp > 0:
                close = close.copy()
                close.iloc[-1] = rp
        except Exception:
            pass

    df["Close_Calc"] = close

    # --- 3) 이동평균 ---
    df["MA5"] = df["Close_Calc"].rolling(5).mean()
    df["MA20"] = df["Close_Calc"].rolling(20).mean()
    df["MA60"] = df["Close_Calc"].rolling(60).mean()

    # --- 4) RSI (14) ---
    delta = df["Close_Calc"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- 5) MACD (12-26-9) ---
    exp12 = df["Close_Calc"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close_Calc"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]
    df["Prev_MACD_Hist"] = df["MACD_Hist"].shift(1)

    # --- 6) 20일 변동성 ---
    df["STD20"] = df["Close_Calc"].rolling(20).std()

    # --- 7) 최근 5일 수익률 (단기 모멘텀) ---
    df["Ret5"] = df["Close_Calc"].pct_change(5)

    return df.dropna()


def get_ai_score_row(row: pd.Series) -> float:
    """
    2주 스윙 백테스트에서 쓰던 AI_Score 로직 100% 동일 이식
    - 상승 추세 + 20일선 근처 눌림
    - RSI 구간
    - 최근 5일 모멘텀
    - MACD 방향/가속
    - 변동성 페널티
    """
    try:
        curr = float(row['Close_Calc'])
        ma5 = float(row['MA5'])
        ma20 = float(row['MA20'])
        ma60 = float(row['MA60'])
        rsi = float(row['RSI'])
        macd = float(row['MACD'])
        sig = float(row['Signal_Line'])
        macd_hist = float(row['MACD_Hist'])
        prev_hist = float(row['Prev_MACD_Hist'])
        std20 = float(row['STD20'])
        ret5 = float(row.get('Ret5', 0.0) or 0.0)

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
        dist20 = (curr - ma20) / ma20
        abs_d20 = abs(dist20)

        # -2% ~ +3%: 최적 매수 존, 20점까지 가산
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

        # 4) 최근 5일 수익률 (단기 모멘텀)
        if ret5 > 0:
            score += min(7.0, ret5 * 100.0 * 2.0)
        else:
            score += ret5 * 100.0 * 0.5

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
    스캐너 / 마이포트폴리오 / (원하면 백테스트) 에서 공통으로 쓰는 해석 함수
    - 점수는 위 get_ai_score_row로만 계산
    - 스나이퍼 진입 조건도 백테스트 로직 그대로:
      * 추세(60일선 위 & 20>60)
      * 20일선 ±3% 눌림
      * RSI 35~65
      * MACD 상승 (MACD>시그널, 히스토그램>0)
      * AI_Score >= 70
      * 최근 5일 수익률 Ret5 >= -2%
    """
    if df is None or df.empty or len(df) < 60:
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
        hist = float(row['MACD_Hist'])
        std20 = float(row['STD20'])
        ret5 = float(row.get('Ret5', 0.0) or 0.0)
    except Exception:
        return "오류", "gray", "계산 실패", 0.0

    reasons = []

    # 1) 추세
    if curr > ma60 and ma20 > ma60:
        reasons.append("상승 추세(20·60일선 위 정배열)")
        trend_ok = True
    elif curr > ma60:
        reasons.append("60일선 위지만 20일선이 약함")
        trend_ok = False
    else:
        reasons.append("60일선 아래(조정/하락 추세)")
        trend_ok = False

    # 2) 20일선과의 거리
    dist_ma20 = (curr - ma20) / ma20 if ma20 != 0 else 0.0
    if curr > ma60 and abs(dist_ma20) <= 0.03:
        reasons.append(f"20일선 근처 눌림({dist_ma20*100:.1f}%)")
    elif dist_ma20 > 0.10:
        reasons.append("20일선 대비 과열(10%↑)")

    # 3) RSI 상태
    if rsi < 30:
        reasons.append(f"RSI 과매도({rsi:.1f})")
    elif rsi > 70:
        reasons.append(f"RSI 과매수({rsi:.1f})")
    elif 40 <= rsi <= 60:
        reasons.append(f"안정적 RSI({rsi:.1f})")

    # 4) 변동성 정보
    vol_ratio = std20 / curr if curr > 0 else 0.0
    if vol_ratio > 0.05:
        reasons.append(f"변동성 높음({vol_ratio*100:.1f}%)")
    elif 0.015 <= vol_ratio <= 0.05:
        reasons.append(f"적정 변동성({vol_ratio*100:.1f}%)")

    # 5) 최근 5일 수익률
    if ret5 > 0:
        reasons.append(f"최근 5일 +{ret5*100:.1f}%")
    elif ret5 < 0:
        reasons.append(f"최근 5일 {ret5*100:.1f}%")

    # 6) MACD 방향
    macd_ok = (macd > sig and hist > 0)
    if macd_ok:
        reasons.append("MACD 상승 에너지")
    else:
        reasons.append("MACD 약세/조정")

    # ---- 스나이퍼 진입 조건 (백테스트와 동일) ----
    pullback_ok = (-0.03 <= dist_ma20 <= 0.03)
    rsi_ok = (35 <= rsi <= 65)
    base_entry = trend_ok and pullback_ok and rsi_ok and macd_ok
    entry_signal = base_entry and (score >= 70.0) and (ret5 >= -0.02)

    # ---- 최종 카테고리 ----
    if entry_signal:
        cat = "🚀 AI 스나이퍼 매수 신호 (조건 만족 전부 매수)"
        col = "green"
    else:
        if score >= 80:
            cat = "📈 강한 상승 추세 (매수 우위)"
            col = "blue"
        elif score >= 65:
            cat = "📈 매수 우위 (조건 일부 부족)"
            col = "blue"
        elif score >= 45:
            cat = "👀 관망 구간 (중립)"
            col = "gray"
        elif score >= 30:
            cat = "📉 비중 축소/부분 매도 구간"
            col = "orange"
        else:
            cat = "💥 스나이퍼 매도/회피 구간 (추세 이탈/위험)"
            col = "red"

    reasoning = " / ".join(reasons[:3]) if reasons else "지표 정보 부족"
    return cat, col, reasoning, round(score, 2)


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
        "pct": pct,
        "profit_amt": net_profit,
        "net_eval_amt": net_eval,
        "currency": "₩" if is_kr else "$",
    }


# ---------------------------------------------------------
# 4. UI
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2, tab3 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)", "📘 알고리즘 백서"])

# TAB 1: 스캐너
with tab1:
    st.markdown("### 📋 AI 정밀 스캐너")
    st.caption("초정밀 실시간/AfterMarket 데이터 기반 AI 분석 (AI 스나이퍼 기준)")

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

                        if df_indi is None or df_indi.empty:
                            continue

                        # 🔥 거래량 비율(Volume Ratio) 계산: 최근 거래량 / 20일 평균 거래량
                        vol_ratio = np.nan
                        try:
                            if "Volume" in df_tk.columns:
                                vol_series = df_tk["Volume"].astype(float).dropna()
                                if len(vol_series) >= 20:
                                    last_vol = float(vol_series.iloc[-1])
                                    mean_vol20 = float(vol_series.tail(20).mean())
                                    if mean_vol20 > 0:
                                        vol_ratio = last_vol / mean_vol20
                        except Exception:
                            vol_ratio = np.nan

                        # 🔥 백테스트와 동일한 AI_Score/스나이퍼 기준으로 매수/매도 해석
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
                            "핵심 요약": reasoning,
                            "거래량비율": vol_ratio,  # 🔥 추가 필드
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
        df_scan = st.session_state['scan_result_df']

        # 🔥 AI 점수 100점 종목이 5개 초과일 때, 거래량 비율 상위 5개 추천
        try:
            if "점수" in df_scan.columns:
                df_100 = df_scan[df_scan["점수"] == 100.0]
                if len(df_100) > 5 and "거래량비율" in df_100.columns:
                    df_100_valid = df_100.dropna(subset=["거래량비율"])
                    if not df_100_valid.empty:
                        top5 = df_100_valid.sort_values("거래량비율", ascending=False).head(5)
                        st.markdown("#### 🔥 AI 점수 100점 + 거래량 비율 상위 5 종목 추천")
                        st.dataframe(
                            top5[["종목명", "점수", "현재가", "RSI", "AI 등급", "핵심 요약", "거래량비율"]],
                            use_container_width=True,
                            hide_index=True,
                        )
        except Exception:
            pass

        st.dataframe(
            df_scan,
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
            hide_index=True,
        )

# TAB 2: 포트폴리오
with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오")
    st.caption("네이버페이(국내) / 1분봉(해외) 실시간 기반 | 세후 순수익 계산 + AI 스나이퍼 진단")

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
                            "qty": input_qty,
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
                        key="edit_avg_price",
                    )
                    new_qty = st.number_input(
                        "새 보유 수량(주)",
                        min_value=0,
                        value=int(target.get("qty", 1)),
                        key="edit_qty",
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
            st.subheader(f"{user_id}님의 보유 종목 진단 (AI 스나이퍼 기준)")
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

                # 데이터 유효성 검사 및 추출
                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                    if not df_tk.empty:
                        curr_price = realtime_map.get(tk)
                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)

                if df_indi is not None and not df_indi.empty:
                    curr = float(df_indi['Close_Calc'].iloc[-1])

                # 🔥 여기서도 백테스트와 동일한 AI_Score/스나이퍼 기준 사용
                if df_indi is not None and not df_indi.empty:
                    cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)
                else:
                    cat, col_name, reasoning, score = "데이터 로딩 중", "gray", "잠시 후 다시 시도", 0.0

                if curr > 0:
                    res = calculate_total_profit(tk, avg, curr, qty)
                    display_list.append({
                        "name": name,
                        "tk": tk,
                        "avg": avg,
                        "curr": curr,
                        "qty": qty,
                        "cat": cat,
                        "col_name": col_name,
                        "reasoning": reasoning,
                        "profit_pct": res['pct'],
                        "profit_amt": res['profit_amt'],
                        "eval_amt": res['net_eval_amt'],
                        "currency": res['currency'],
                        "score": score,
                    })
                else:
                    display_list.append({
                        "name": name,
                        "tk": tk,
                        "avg": avg,
                        "curr": avg,
                        "qty": qty,
                        "cat": "로딩 실패",
                        "col_name": "gray",
                        "reasoning": "데이터 수신 불가",
                        "profit_pct": 0.0,
                        "profit_amt": 0.0,
                        "eval_amt": 0.0,
                        "currency": "₩" if tk.endswith(".KS") or tk.endswith(".KQ") else "$",
                        "score": 0.0,
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
                        fmt_avg = f"{item['avg']:,.0f}" if item['currency'] == "₩" else f"{item['avg']:,.2f}"
                        fmt_eval = f"{item['eval_amt']:,.0f}" if item['currency'] == "₩" else f"{item['eval_amt']:,.2f}"

                        st.metric(
                            "총 순수익 (수수료·세금 차감 후)",
                            f"{item['profit_pct']:.2f}%",
                            delta=f"{sym}{item['profit_amt']:,.0f}" if sym == "₩"
                            else f"{sym}{item['profit_amt']:,.2f}",
                        )
                        st.markdown(f"**세후 총 평가금:** {safe_sym}{fmt_eval}", unsafe_allow_html=True)
                        st.markdown(
                            f"<small style='color: gray'>평단: {safe_sym}{fmt_avg} / 현재: {safe_sym}{fmt_curr}</small>",
                            unsafe_allow_html=True,
                        )

                    with c3:
                        st.markdown(f"**AI 점수: {item['score']:.1f}점**")
                        st.markdown(f"**판단:** :{item['col_name']}[{item['cat']}]")
                        st.info(f"💡 {item['reasoning']}")
                    st.divider()

            if st.button("🗑️ 포트폴리오 전체 삭제"):
                doc_ref.delete()
                st.rerun()

# TAB 3: 알고리즘 백서
with tab3:
    st.markdown("## 📘 AI 투자 전략 알고리즘 백서 (Whitepaper v2.0)")
    st.markdown("""
본 서비스에 탑재된 AI 알고리즘은 월가(Wall St)의 퀀트 트레이딩에서 검증된 **'추세 추종(Trend Following)'** 전략과  
단기 과매도 구간을 포착하는 **'평균 회귀(Mean Reversion)'** 이론을 정밀하게 결합한 하이브리드 모델입니다.

모든 점수는 **0점(강력 매도) ~ 100점(강력 매수)** 사이의 실수(float)로 계산되며,  
단순한 조건 매칭이 아닌 **지표의 강도(Strength)와 이격도(Divergence)**를 미분적으로 분석하여 산출됩니다.

현재 실전 스캐너/포트폴리오 진단 엔진은 **AI 스나이퍼 전략**을 기준으로 매수/매도 구간을 해석합니다.
""")

    st.divider()

    st.subheader("1. 🎯 AI 종합 점수 가이드 (Scoring Guide)")
    score_guide_data = [
        {"점수 구간": "80점 ~ 100점", "등급": "🚀 강력 매수 (Strong Buy)", "설명": "추세, 눌림, 모멘텀이 모두 이상적인 상태. AI 스나이퍼 관점에서 최상급 진입 후보."},
        {"점수 구간": "70점 ~ 80점", "등급": "📈 매수 우위 (Buy)", "설명": "상승 추세가 확연하며, 스나이퍼 진입 조건에 근접한 구간. 눌림 상태에 따라 전체 매수 트리거 가능."},
        {"점수 구간": "60점 ~ 70점", "등급": "👍 강한 종목 (Strong Trend)", "설명": "추세는 강하지만 눌림·RSI 조건이 덜 맞을 수 있음. 기존 보유자는 홀딩, 신규 진입은 보수적 접근 권장."},
        {"점수 구간": "45점 ~ 60점", "등급": "👀 관망 (Hold/Neutral)", "설명": "방향성이 뚜렷하지 않거나, 상승 후 쉬어가는 구간. 스나이퍼 기준으로는 대기/관망 영역."},
        {"점수 구간": "0점 ~ 45점", "등급": "💥 매도/회피 (Exit/Avoid)", "설명": "역배열 또는 추세 이탈 가능성이 큰 구간. 스나이퍼 전략에서는 비중 축소 또는 관망 권장."},
    ]
    st.table(score_guide_data)

    st.header("2. 🧠 핵심 평가 로직 (5-Factor Deep Dive)")
    st.markdown("AI는 다음 5가지 핵심 요소를 수치화하여 미세한 점수 차이를 만들어냅니다.")

    with st.expander("① 추세 (Trend Hierarchy) - 주가의 '생명선'", expanded=True):
        st.markdown("""
**"추세는 당신의 친구입니다 (Trend is your friend)."**

AI는 이동평균선(Moving Average)의 배열 상태를 통해 주가의 현재 위치를 파악합니다.

* **장기 추세 (60일선):** 주가의 '계절'을 의미합니다. 60일선 위에 있다는 것은 현재가 '여름(상승장)'임을 뜻합니다.  
* **20·60일선 정배열:** `MA20 > MA60` 이면서 가격이 60일선 위에 있을 때, 상승 추세가 살아있는 것으로 판단하여 가산점을 부여합니다.  
* **5·20·60 정배열:** `MA5 > MA20 > MA60` 인 경우, 단기·중기·장기 추세가 모두 한 방향으로 정렬된 것으로 보고 추가 가산점을 부여합니다.  
* **역배열 감점:** 반대로 모든 이동평균선 아래에 주가가 위치하면 '하락장'으로 간주하여 강한 페널티를 줍니다.
""")

    with st.expander("② 황금 눌림목 (The Golden Dip) - 고수익의 비밀", expanded=True):
        st.markdown("""
**"무릎에 사서 어깨에 팔아라."**

가장 높은 점수가 부여되는 핵심 구간입니다. 상승 추세(60일선 위)에 있는 종목이  
일시적인 조정으로 **20일 이동평균선(생명선)** 근처까지 눌렸을 때를 포착합니다.

* **초정밀 거리 계산:** 현재 주가와 20일선 사이의 거리가 `-2% ~ +3%` 이내에 있을 때 가장 높은 가산점(+20점)을 줍니다.  
* **살짝 깊은 눌림(-5% ~ -2%):** 기술적 반등 가능성이 커지는 구간으로, 소폭의 추가 점수를 부여합니다.  
* **과열 경고 (20일선 대비 +8%↑):** 단기 급등으로 해석하여 점수를 강하게 깎습니다.  
* 스나이퍼 전략의 실제 진입은 **"상승 추세 + 20일선 ±3% 이내"** 조건을 동시에 만족할 때만 활성화됩니다.
""")

    with st.expander("③ RSI (상대강도지수) - 투자 심리 역이용", expanded=True):
        st.markdown("""
**"공포에 사고 탐욕에 팔아라."**

RSI는 현재 시장의 과열/침체 정도를 0~100 사이 숫자로 나타냅니다.

* **안정적 상승 (RSI 40~60):** 가장 건강한 상승 구간으로 판단되어 가산점이 부여됩니다.  
* **완만한 눌림 (RSI 30~40):** 과도한 공포는 아니지만, 조정 국면으로 해석하여 소폭 가산합니다.  
* **살짝 과열 (RSI 60~70):** 추세는 좋지만 단기 상단부에 근접해 있는 상황으로, 적당한 가산점을 유지합니다.  
* **과매도/과매수 극단 (RSI <25 또는 >75):** 급락/급등 구간으로, 추세 붕괴 또는 과열 위험 구간으로 인식하여 감점합니다.  

스나이퍼 전략에서는 **RSI 35~65** 구간을 선호하며, 이 범위를 벗어나면 진입 신호를 보수적으로 봅니다.
""")

    with st.expander("④ MACD & 모멘텀 - 상승의 속도", expanded=True):
        st.markdown("""
이동평균선이 '방향'을 알려준다면, MACD는 '속도'를 알려줍니다.

* **MACD > Signal & Histogram > 0:** 상승 에너지가 켜진 상태로, AI 점수를 추가로 끌어올립니다.  
* **Histogram 증가:** 바로 전날보다 막대 높이가 커지고 있다면, 상승 가속도가 붙는 중으로 해석하여 가산점을 줍니다.  
* **스나이퍼 진입 필수 조건:** MACD가 시그널 위에 있고, 히스토그램이 양수일 때만 진입 후보로 인정합니다.  
* **단기 모멘텀 (Ret5):** 최근 5일 수익률이 -2% 이하로 너무 약하면 스나이퍼 진입을 보류합니다.
""")

    with st.expander("⑤ 변동성 (Volatility) - 위험 관리", expanded=True):
        st.markdown("""
변동성이 너무 큰 종목은 '도박'에 가깝습니다.

* **20일 표준편차 / 현재가 비율(STD20 / Price)** 을 통해 일간 등락 폭을 추정합니다.  
* **1.5% ~ 5% 사이:** 이상적인 스윙 변동성 구간으로 보고 가산점을 부여합니다.  
* **5% 이상:** 급등락이 심한 종목으로 간주하여 강한 감점을 적용, 리스크를 통제합니다.  
* **너무 안 움직이는 종목:** 변동성이 지나치게 낮아도 기회 비용 측면에서 소폭 감점합니다.
""")

    st.divider()
    st.info(
        "💡 **Tip:** 본 알고리즘은 '상승 추세가 살아있는 종목'이 '20일선 근처로 눌렸을 때'를 포착하여, "
        "**AI 스나이퍼 전략(조건 만족 전부 매수 & 보수적 손절)** 기준으로 해석합니다. "
        "점수가 높더라도 본인의 투자 원칙과 병행하여 사용하시기 바랍니다."
    )
