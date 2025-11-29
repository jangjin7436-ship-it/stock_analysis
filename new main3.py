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
        except Exception:
            pass
        return hist_map, realtime_map

    # 여러 개일 때 (Bulk)
    try:
        df_hist = yf.download(
            us_tickers,
            period="2y",
            interval="1d",
            progress=False,
            group_by='ticker',
            auto_adjust=True
        )
        df_real = yf.download(
            us_tickers,
            period="5d",
            interval="1m",
            progress=False,
            group_by='ticker',
            prepost=True
        )

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
# 3. 분석 엔진 (백테스트 점수/매매 기준 그대로 적용 - AI 스나이퍼)
# ---------------------------------------------------------
def get_ai_score_row(row):
    """
    2주 스윙 기준 AI 점수:
    - 상승 추세 + 20일선 근처 눌림
    - 적당한 RSI 구간
    - 최근 5일 모멘텀
    - MACD 방향
    - 변동성 페널티
    (백테스트 엔진과 완전히 동일)
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


def calculate_indicators(df, realtime_price=None):
    """
    백테스트 엔진과 동일한 방식의 지표 계산 + Ret5/AI_Score 포함
    - Close/Adj Close 중 하나 사용
    - 실시간가가 들어오면 마지막 캔들에 반영 후 지표 계산
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

    # 실시간 가격 주입 (스캐너 특화)
    if realtime_price is not None and realtime_price > 0:
        try:
            close.iloc[-1] = realtime_price
        except Exception:
            pass

    df['Close_Calc'] = close

    # 이동평균 (5/20/60)
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()

    # RSI (14일)
    delta = df['Close_Calc'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
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

    # 최근 5일 수익률 (2주 스윙용 단기 모멘텀)
    df['Ret5'] = df['Close_Calc'].pct_change(5)

    df = df.dropna()

    # AI 점수 계산 (백테스트와 완전히 동일)
    df['AI_Score'] = df.apply(get_ai_score_row, axis=1)

    return df


def analyze_advanced_strategy(df):
    """
    스캐너/포트폴리오용 AI 해석 엔진
    - 백테스트의 AI_Score/필터를_
