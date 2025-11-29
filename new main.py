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
import re  # ✅ 국내 애프터마켓 가격 파싱용

# ---------------------------------------------------------
# 0. 파이어베이스(DB) 설정
# ---------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, firestore

def _now_kst():
    """UTC 기준 현재 시간을 KST(UTC+9)로 변환."""
    now_utc = datetime.datetime.utcnow()
    return now_utc + datetime.timedelta(hours=9)

def _is_kr_regular_session():
    """
    한국 정규장(09:00~15:30) 여부 판별.
    정확한 초 단위까지 필요 없으니 대략 시간 만으로 판단.
    """
    t = _now_kst().time()
    return datetime.time(9, 0) <= t <= datetime.time(15, 30)

def _is_kr_after_session():
    """
    시간외 단일가(16:00~18:00) 구간 여부.
    """
    t = _now_kst().time()
    return datetime.time(16, 0) <= t <= datetime.time(18, 0)

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

# 스캔 결과 영구 보존을 위한 세션 초기화
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
# 2. 데이터 수집 혁신 (New Method)
# ---------------------------------------------------------

def fetch_kr_polling(ticker):
    """
    국내 주식 실시간 가격

    네이버 domestic realtime API에서
    - 정규장 가격(closePrice)와
    - 시간외 단일가(overMarketPriceInfo.overPrice)를 함께 받아서

    두 가격의 localTradedAt(체결 시각)을 비교해
    **가장 최근에 체결된 가격**을 현재가로 사용한다.
    """
    code = ticker.split('.')[0]  # "005930.KS" -> "005930"

    try:
        url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Referer": "https://finance.naver.com/"
        }
        res = requests.get(url, headers=headers, timeout=3)
        res.raise_for_status()
        data = res.json()

        datas = data.get("datas", [])
        if not datas:
            raise ValueError("no datas in naver realtime response")

        item = datas[0]

        # ---- 1) 가격 문자열 정리 ----
        over_info = item.get("overMarketPriceInfo") or {}

        over_price_str  = str(over_info.get("overPrice", "")).replace(",", "").strip()
        close_price_str = str(item.get("closePrice", "")).replace(",", "").strip()

        over_price  = float(over_price_str)  if over_price_str  not in ("", "0") else None
        close_price = float(close_price_str) if close_price_str not in ("", "0") else None

        # ---- 2) 체결 시각 파싱 (정규장 / 시간외) ----
        def _parse_dt(s: str):
            if not s:
                return None
            try:
                # 예: "2025-11-28T20:00:00.000000+09:00"
                return datetime.datetime.fromisoformat(s)
            except Exception:
                return None

        base_time_str = item.get("localTradedAt", "")
        over_time_str = over_info.get("localTradedAt", "")

        base_time = _parse_dt(base_time_str)
        over_time = _parse_dt(over_time_str)

        # ---- 3) 가장 최근에 체결된 가격 선택 ----
        chosen_price = None
        chosen_time = None

        if close_price is not None:
            chosen_price = close_price
            chosen_time = base_time

        if over_price is not None:
            if over_time is not None and chosen_time is not None:
                # 둘 다 시간이 있으면 더 최근 시각 쪽 선택
                if over_time > chosen_time:
                    chosen_price = over_price
                    chosen_time = over_time
            else:
                # 한쪽만 시간이 있으면, 그냥 가격 있는 쪽 사용
                if chosen_price is None:
                    chosen_price = over_price
                    chosen_time = over_time

        if chosen_price is not None:
            return (ticker, float(chosen_price))

        # usable price가 없으면 FDR 폴백
        raise ValueError("no usable price in naver realtime response")

    except Exception:
        # 네이버 API 실패 시 FDR 종가로 폴백 (애프터마켓은 반영 안 됨)
        try:
            df = fdr.DataReader(code, "2023-01-01")
            if not df.empty:
                return (ticker, float(df["Close"].iloc[-1]))
        except Exception:
            pass
        return (ticker, None)


def fetch_us_1m_candle(ticker):
    """
    [New Method] 미국 주식 1분봉(장전/장후 포함) 조회
    가장 마지막에 찍힌 캔들의 Close 가격을 가져옴. 이것이 진정한 애프터마켓 가격.
    """
    try:
        # period='5d'로 넉넉히 잡고, interval='1m', prepost=True(장외거래 포함)
        df = yf.download(ticker, period="5d", interval="1m", prepost=True, progress=False)
        if not df.empty:
            # 가장 마지막 줄의 종가(Close)
            last_price = float(df['Close'].iloc[-1])
            return (ticker, last_price)
        return (ticker, None)
    except:
        return (ticker, None)

def fetch_history_data(ticker):
    """지표 분석용 일봉 데이터 (2년치, 정규장 종가 기준)"""
    try:
        if ticker.endswith('.KS') or ticker.endswith('.KQ'):
            # 국내는 FDR 일봉 그대로 사용 (정규장 종가)
            df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        else:
            # 해외: 2년 치 일봉, 정규장만
            df = yf.download(
                ticker,
                period="2y",
                interval="1d",
                progress=False,
                prepost=False  # 🔑 장외 제외 (일봉은 정규장만)
            )

            # 컬럼 정리
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]

            # 많은 차트가 Adjusted Close 기준으로 이동평균/RSI 계산 → 있으면 사용
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']

        return (ticker, df)
    except Exception:
        return (ticker, None)

@st.cache_data(ttl=0) # 캐시 0초 (항상 실행)
def get_precise_data(tickers_list):
    """
    1) 지표 계산용 일봉 데이터 (정규장 종가 기준만 사용)
    2) 실시간/애프터마켓 가격은 별도 딕셔너리로 관리
       → 지표엔 절대 섞지 않음
    """
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]

    realtime_prices = {}
    hist_map = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # A. 실시간 가격 (국내: 네이버, 해외: 1분봉)
        fut_real = []
        for t in kr_tickers:
            fut_real.append(executor.submit(fetch_kr_polling, t))
        for t in us_tickers:
            fut_real.append(executor.submit(fetch_us_1m_candle, t))

        # B. 히스토리 데이터 (일봉)
        fut_hist = [executor.submit(fetch_history_data, t) for t in tickers_list]

        # 실시간 수집
        for f in concurrent.futures.as_completed(fut_real):
            tk, p = f.result()
            if p is not None:
                realtime_prices[tk] = p

        # 히스토리 수집
        for f in concurrent.futures.as_completed(fut_hist):
            tk, df = f.result()
            if df is not None and not df.empty:
                hist_map[tk] = df

    # C. 최종 일봉 데이터 정리 (지표용) - ❗ 실시간 가격 덮어쓰기 금지
    final_dfs = {}
    for t in tickers_list:
        if t in hist_map:
            df = hist_map[t].copy()

            # MultiIndex 방지 + 정렬
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.sort_index()

            final_dfs[t] = df

    return final_dfs, realtime_prices

def calculate_indicators(df):
    """
    스윙(2~4주) 전략용 지표 세트 계산

    - Close_Calc : 종가(결측 보정)
    - MA5 / MA10 / MA20 / MA60 : 단·중기 이동평균
    - STD20 : 20일 표준편차 (변동성)
    - VolMA20 : 20일 평균 거래량
    - RSI(14), MACD(12,26,9)
    - BB_Upper / BB_Lower : 볼린저 밴드
    - MOM10 : 10일 모멘텀 (약 2주 수익률)
    """
    if len(df) < 60:
        return None

    df = df.copy()

    # 단일 Close 시리즈 확보
    if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close_series = close.iloc[:, 0]
        else:
            close_series = close
    else:
        return None

    close_series = close_series.ffill()
    df['Close_Calc'] = close_series

    # 📌 이동평균 (단기/중기)
    df['MA5']  = df['Close_Calc'].rolling(window=5).mean()
    df['MA10'] = df['Close_Calc'].rolling(window=10).mean()
    df['MA20'] = df['Close_Calc'].rolling(window=20).mean()
    df['MA60'] = df['Close_Calc'].rolling(window=60).mean()

    # 📌 변동성 / 모멘텀
    df['STD20'] = df['Close_Calc'].rolling(window=20).std()
    df['RET1']  = df['Close_Calc'].pct_change()
    df['MOM10'] = df['Close_Calc'] / df['Close_Calc'].shift(10) - 1  # 최근 10영업일 수익률

    # 📌 거래량 관련
    if 'Volume' in df.columns:
        vol = df['Volume']
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        df['Volume_Calc'] = vol
        df['VolMA20'] = vol.rolling(window=20).mean()
    else:
        df['Volume_Calc'] = 0
        df['VolMA20'] = 0

    # 📌 RSI(14)
    delta = df['Close_Calc'].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 📌 MACD(12,26,9)
    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 📌 볼린저 밴드 (20, 2σ)
    df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)

    # NaN 행 제거
    return df.dropna()

def calculate_total_profit(ticker, avg_price, current_price, quantity):
    """
    토스증권 방식에 최대한 맞춘 순수익/수익률 계산

    - avg_price: 토스 '1주 평균금액' 그대로 입력했다고 가정 (매수 수수료 이미 포함)
    - current_price: 우리가 실시간으로 가져온 현재가 (애프터마켓 포함)
    - quantity: 보유 주식 수

    국내주식(KS/KQ):
        • 매도 수수료 ≈ 0.0295%
        • 증권거래세   = 0.15%
        → 평가금 = 현재가*수량 - (수수료 + 세금)

    해외주식(그 외):
        • 매도 수수료 ≈ 0.1965%
        • 세금 없음 (토스 화면 기준)
    """
    # 1) 기본 값 계산
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")

    qty = float(quantity)
    avg_price = float(avg_price)
    current_price = float(current_price)

    total_buy = avg_price * qty              # 원금 (이미 매수 수수료 포함된 평단이라고 가정)
    gross_eval = current_price * qty         # 세전 평가금 (현재가 * 수량)

    # 2) 시장별 수수료/세금율 설정 (토스 캡처 기반 튜닝)
    if is_kr:
        fee_rate = 0.000295   # ≈ 0.0295%
        tax_rate = 0.0015     # 0.15% 증권거래세
    else:
        fee_rate = 0.001965   # ≈ 0.1965% (TQQQ 예시 기준)
        tax_rate = 0.0        # 해외주식은 세금 컬럼 '-' 기준

    sell_fee = gross_eval * fee_rate
    sell_tax = gross_eval * tax_rate

    # 3) 세후 평가금 & 순수익
    net_eval = gross_eval - sell_fee - sell_tax       # 세후 총 평가금
    net_profit_amt = net_eval - total_buy             # 총 순수익 (수수료·세금 반영)

    if total_buy > 0:
        net_profit_pct = (net_profit_amt / total_buy) * 100
    else:
        net_profit_pct = 0.0

    currency = "₩" if is_kr else "$"

    return {
        "pct": net_profit_pct,       # 총 수익률 (%)
        "profit_amt": net_profit_amt,  # 총 순수익 (수수료·세금 차감 후)
        "net_eval_amt": net_eval,      # 세후 총 평가금
        "currency": currency
    }

def analyze_advanced_strategy(df, curr_override=None):
    """
    [2~4주 스윙 전용] 매수 매력도 스코어링 엔진
    """
    if df is None or df.empty:
        return "분석 불가", "gray", "데이터 부족", 0

    try:
        # 기본값은 일봉 종가
        curr = float(df['Close_Calc'].iloc[-1])

        # 🔑 실시간 가격이 들어오면 그걸로 덮어쓰기
        if curr_override is not None:
            try:
                curr = float(curr_override)
            except:
                pass

        ma5  = float(df['MA5'].iloc[-1])
        ma10 = float(df['MA10'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        ma60 = float(df['MA60'].iloc[-1])

        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        sig  = float(df['Signal_Line'].iloc[-1])
        prev_macd = float(df['MACD'].iloc[-2])
        prev_sig  = float(df['Signal_Line'].iloc[-2])

        bb_upper = float(df['BB_Upper'].iloc[-1])
        bb_lower = float(df['BB_Lower'].iloc[-1])

        vol     = float(df['Volume_Calc'].iloc[-1]) if 'Volume_Calc' in df.columns else 0.0
        vol_ma  = float(df['VolMA20'].iloc[-1]) if 'VolMA20' in df.columns else 0.0
        open_px = float(df['Open'].iloc[-1]) if 'Open' in df.columns else curr
        prev_close = float(df['Close_Calc'].iloc[-2])

        std20 = float(df['STD20'].iloc[-1])
        mom10 = float(df['MOM10'].iloc[-1]) if 'MOM10' in df.columns else 0.0
    except Exception:
        return "데이터 오류", "gray", "지표 계산 실패", 0

    score = 50  # 기준점
    reasons = []

    # 1) 📈 중기 추세 필터 (4주 이상)
    if curr > ma60 and ma20 > ma60:
        score += 20
        reasons.append("📈 중기 상승 추세(60일선 위, 20일선도 우상향)")
    elif curr > ma60:
        score += 5
        reasons.append("↗ 60일선 위이지만 추세는 다소 애매")
    else:
        score -= 25
        reasons.append("⚠ 하락 추세(60일선 아래) - 2~4주 스윙에 부적합")

    # 2) 💎 단기 위치 (2주 안 진입/청산 시, 너무 고점·너무 저점 피하기)
    dist_ma10 = (curr - ma10) / ma10 if ma10 > 0 else 0
    dist_ma20 = (curr - ma20) / ma20 if ma20 > 0 else 0

    # 상승장 + 20일선 인근 눌림목 → 최우선 가산
    if (curr >= ma20) and (curr >= ma60) and (-0.03 <= dist_ma20 <= 0.02):
        score += 20
        reasons.append("💎 상승장 속 20일선 부근 눌림목 (2~4주 스윙 최적 구간)")
    # 약간 위에 있지만 과열은 아닌 구간
    elif 0.02 < dist_ma20 <= 0.07:
        score += 5
        reasons.append("🙂 20일선 위이지만 과열은 아님 (추세 상단)")
    # 20일선에서 너무 멀리 이탈 → 단기 과열
    elif dist_ma20 > 0.07:
        score -= 15
        reasons.append("🔥 20일선 대비 과도한 이격(>7%) - 단기 조정 위험")

    # 3) ⚖ RSI - 과열/과매도 관리
    if 40 <= rsi <= 60:
        score += 15
        reasons.append("⚖ RSI 40~60, 단기 균형 구간 (스윙 진입 적당)")
    elif 30 <= rsi < 40:
        score += 5
        reasons.append("가벼운 과매도, 단기 반등 여지")
    elif rsi < 30:
        score += 5
        reasons.append("심한 과매도 - 반등 여지는 있으나 변동성 큼")
    elif 60 < rsi <= 70:
        score -= 5
        reasons.append("과열 초입 - 보수적 접근 필요")
    else:  # rsi > 70
        score -= 20
        reasons.append("🚨 RSI 과열 (단기 조정 가능성↑)")

    # 4) 📊 10일 모멘텀 (≈ 최근 2주 수익률)
    if 0.03 <= mom10 <= 0.15:
        score += 10
        reasons.append("📊 최근 2주간 3~15% 상승, 건강한 모멘텀")
    elif -0.05 <= mom10 < 0.03:
        score += 5
        reasons.append("횡보/소폭 조정 후 구간 (재출발 가능)")
    elif mom10 > 0.25:
        score -= 15
        reasons.append("급등(>25%) 이후 구간 - 차익 실현 구간일 가능성")
    elif mom10 < -0.10:
        score -= 10
        reasons.append("최근 2주간 10% 이상 급락 - 리스크 큰 반등 매매")

    # 5) MACD 크로스 - 타이밍
    if macd > sig and prev_macd <= prev_sig:
        score += 15
        reasons.append("⚡ MACD 골든크로스 - 단기 매수 시그널")
    elif macd > sig:
        score += 5
        reasons.append("MACD 상방 유지")
    elif macd < sig and prev_macd >= prev_sig:
        score -= 10
        reasons.append("💧 MACD 데드크로스 - 단기 하락 전환 가능")
    else:
        score -= 3  # 방향 모호

    # 6) 🎢 변동성 + 거래량 (리스크 & 신뢰도)
    vol_ratio = std20 / curr if curr > 0 else 0
    if vol_ratio > 0.08:
        score -= 15
        reasons.append("🎢 변동성 매우 큼(20일 표준편차>8%) - 2~4주 보유 리스크↑")
    elif vol_ratio < 0.03:
        score += 5
        reasons.append("⚙ 변동성 낮음 - 비교적 안정적인 스윙 환경")

    if vol_ma > 0 and vol > vol_ma * 1.5 and curr > prev_close:
        score += 10
        reasons.append("🔥 거래량 동반 상승 - 추세 신뢰도↑")

    # 점수 범위 제한
    score = max(0, min(100, score))

    # 등급 매핑 (스윙용 문구로 변경)
    if score >= 80:
        category = "🚀 단기 강력 매수 (2~4주 스윙)"
        color_name = "green"
    elif score >= 65:
        category = "📈 매수 우위 (스윙 진입 가능)"
        color_name = "blue"
    elif score >= 45:
        category = "👀 관망 (명확한 엣지 부족)"
        color_name = "gray"
    elif score >= 25:
        category = "📉 매도/비중 축소 권장"
        color_name = "red"
    else:
        category = "💥 강력 매도 또는 관심 제외"
        color_name = "red"

    if not reasons:
        reasons.append("명확한 시그널 부족 - 관망 권장")

    # 너무 길어지지 않게 상위 4개 이유만 요약
    summary = " / ".join(reasons[:4])

    return category, color_name, summary, score

# ---------------------------------------------------------
# 4. UI
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2, tab3 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)", "📘 알고리즘 설명서"])

with tab1:
    st.markdown("### 📋 AI 정밀 스캐너")
    st.caption("초정밀 실시간/AfterMarket 데이터 기반 AI 분석")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 분석 새로고침", type="primary"):
            st.session_state['scan_result_df'] = None 
            st.rerun()

    # 결과가 없으면 새로 분석, 있으면 기존 것 유지 (결과 고정)
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
                        
                        df_indi = calculate_indicators(df_tk)
                        if df_indi is None:
                            continue

                        # 🔑 실시간 가격 (없으면 일봉 종가 사용)
                        base_close = float(df_indi['Close_Calc'].iloc[-1])
                        curr_price = realtime_map.get(ticker_code, base_close)

                        # 🔑 추세 판단에 실시간 가격 반영
                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi, curr_override=curr_price)

                        rsi_val = float(df_indi['RSI'].iloc[-1])
                        name = TICKER_MAP.get(ticker_code, ticker_code)

                        is_kr = ticker_code.endswith(".KS") or ticker_code.endswith(".KQ")
                        sym = "₩" if is_kr else "$"
                        fmt_price = f"{sym}{curr_price:,.0f}" if is_kr else f"{sym}{curr_price:,.2f}"

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
                "점수": st.column_config.ProgressColumn("AI 점수", format="%d점", min_value=0, max_value=100),
                "현재가": st.column_config.TextColumn("현재가"), 
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "AI 등급": st.column_config.TextColumn("AI 판단"),
                "핵심 요약": st.column_config.TextColumn("분석 내용", width="large"),
            },
            hide_index=True
        )

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
            # ✏️ 보유 종목 정보 수정 섹션
            st.markdown("#### ✏️ 보유 종목 정보 수정")

            edit_options = [
                f"{TICKER_MAP.get(p['ticker'], p['ticker'])} ({p['ticker']})"
                for p in pf_data
            ]
            selected_edit = st.selectbox(
                "수정할 종목 선택",
                options=["선택하세요"] + edit_options,
                key="edit_select"
            )

            if selected_edit != "선택하세요":
                # "삼성전자 (005930.KS)" -> "005930.KS"
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
                                new_pf_data.append(
                                    {"ticker": edit_ticker, "price": new_avg, "qty": new_qty}
                                )
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
                
                df_indi = None
                curr = 0

                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                    if not df_tk.empty:
                        df_indi = calculate_indicators(df_tk)

                # 🔑 현재가 (실시간 우선)
                if tk in realtime_map:
                    curr = float(realtime_map[tk])
                elif df_tk is not None and not df_tk.empty:
                    curr = float(df_tk['Close'].iloc[-1])

                cat, col_name, reasoning, score = "데이터 로딩 중", "gray", "잠시 후 다시 시도", 0

                if df_indi is not None:
                    # 🔑 추세 분석에 실시간 가격 반영
                    cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi, curr_override=curr)
                
                # 표시는 실시간 맵 기준 (가장 정확)
                if tk in realtime_map:
                    curr = realtime_map[tk]
                elif df_tk is not None and not df_tk.empty:
                    curr = float(df_tk['Close'].iloc[-1])

                if curr > 0:
                    res = calculate_total_profit(tk, avg, curr, qty)
                    
                    display_list.append({
                        "name": TICKER_MAP.get(tk, tk), "tk": tk, 
                        "avg": avg, "curr": curr, "qty": qty,
                        "cat": cat, "col_name": col_name, "reasoning": reasoning,
                        "profit_pct": res['pct'], 
                        "profit_amt": res['profit_amt'],
                        "eval_amt": res['net_eval_amt'],
                        "currency": res['currency'], 
                        "score": score
                    })
                else:
                    display_list.append({
                        "name": TICKER_MAP.get(tk, tk), "tk": tk, 
                        "avg": avg, "curr": avg, "qty": qty,
                        "cat": "로딩 실패", "col_name": "gray", "reasoning": "데이터 수신 불가",
                        "profit_pct": 0.0, "profit_amt": 0.0, "eval_amt": 0.0,
                        "currency": "$" if not tk.endswith(".KS") else "₩", "score": 0
                    })
            
            display_list.sort(key=lambda x: x['score'], reverse=True)

            for item in display_list:
                with st.container():
                    c1, c2, c3 = st.columns([1.5, 1.5, 4])
                    sym = item['currency'] 
                    
                    with c1:
                        st.markdown(f"### {item['name']}")
                        st.caption(f"{item['tk']} | 보유: {item['qty']}주")
                        
                    with c2:
                        fmt_curr = f"{item['curr']:,.0f}" if item['currency'] == "₩" else f"{item['curr']:,.2f}"
                        fmt_avg  = f"{item['avg']:,.0f}"  if item['currency'] == "₩" else f"{item['avg']:,.2f}"
                        fmt_profit = f"{item['profit_amt']:,.0f}" if item['currency'] == "₩" else f"{item['profit_amt']:,.2f}"
                        fmt_eval   = f"{item['eval_amt']:,.0f}"   if item['currency'] == "₩" else f"{item['eval_amt']:,.2f}"

                        # ✅ 달러 기호가 수식으로 인식되지 않도록 HTML 엔티티로 변환
                        #    - 원: sym = "₩" 또는 "$"
                        #    - 표시용: safe_sym = "₩" 또는 "&#36;" (둘 다 화면에는 ₩ / $로 보임)
                        safe_sym = sym if sym != "$" else "&#36;"

                        # metric은 markdown이 아니라서 굳이 바꿀 필요 없음
                        st.metric("총 순수익 (수수료 제)", f"{item['profit_pct']:.2f}%", delta=f"{sym}{fmt_profit}")

                        # 아래 두 줄만 safe_sym 사용 + HTML 허용
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
    st.markdown("## 📘 AI 투자 전략 알고리즘 상세 백서 (Whitepaper)")
    st.markdown("""
    본 서비스에 탑재된 AI 알고리즘은 **'추세 추종(Trend Following)'** 전략과 **'평균 회귀(Mean Reversion)'** 이론을 
    결합하여 설계되었습니다. 단순히 감에 의존하는 투자가 아닌, 철저한 **데이터와 통계적 확률**에 기반하여 
    0점부터 100점까지의 '매수 매력도'를 산출합니다.
    """)
    
    st.divider()
    
    st.header("1. 🧠 핵심 평가 로직 (5-Factor Model)")
    st.markdown("AI는 다음 5가지 핵심 요소를 종합적으로 분석하여 점수를 계산합니다.")
    
    with st.expander("① 추세 (Trend) - 시장의 흐름을 읽다", expanded=True):
        st.markdown("""
        * **개념:** '달리는 말에 올라타라'는 격언처럼, 주가가 상승세일 때 매수하는 것이 승률이 높습니다.
        * **판단 기준:**
            * **장기 추세 (60일 이동평균선):** 주가가 60일선 위에 있으면 '상승장'으로 판단합니다. (+15점)
            * **단기 추세 (20일 이동평균선):** 주가가 20일선 위에 있으면 단기 탄력이 좋다고 판단합니다. (+10점)
            * **역배열:** 주가가 이동평균선 아래에 있으면 하락 추세로 간주하여 감점합니다. (-10~20점)
        """)

    with st.expander("② 지지 & 저점 (Support) - 싸게 사는 기술", expanded=True):
        st.markdown("""
        * **개념:** 아무리 좋은 주식도 비싸게 사면 의미가 없습니다. 상승 추세 속에서 일시적으로 가격이 하락했을 때(조정)가 기회입니다.
        * **판단 기준:**
            * **황금 눌림목 (Golden Dip):** 주가가 상승 추세(60일선 위)에 있으면서, 단기적으로 하락해 **20일선(-2% ~ +2%)**에 근접할 때. 가장 높은 가산점을 부여합니다. (+25점)
            * **볼린저 밴드 하단:** 주가가 통계적 하단 밴드를 터치하면 '과매도' 상태로 보아 기술적 반등을 기대합니다. (+15점)
            * **볼린저 밴드 상단:** 주가가 상단 밴드를 뚫으면 '단기 고점'으로 보아 감점합니다. (-10점)
        """)

    with st.expander("③ 모멘텀 (Momentum) - 상승 에너지", expanded=True):
        st.markdown("""
        * **개념:** 주가가 상승하려고 하는 '가속도'를 측정합니다.
        * **판단 기준 (MACD):**
            * **골든크로스:** 단기 이평선이 장기 이평선을 뚫고 올라갈 때 강력한 매수 신호로 봅니다. (+15점)
            * **상승 추세 유지:** MACD가 시그널선 위에 머물러 있으면 상승 에너지가 지속되는 것으로 봅니다. (+5점)
            * **데드크로스:** 반대로 하락 반전 신호가 뜨면 감점합니다. (-15점)
        """)
        
    with st.expander("④ 심리 (Psychology) - 공포와 탐욕", expanded=True):
        st.markdown("""
        * **개념:** 투자자들의 심리가 과열되었는지, 공포에 질려있는지를 RSI 지표로 판단합니다.
        * **판단 기준 (RSI):**
            * **침체 구간 (RSI < 30):** '공포' 구간입니다. 남들이 팔 때 사는 역발상 전략으로 가산점을 줍니다. (+15점)
            * **과열 구간 (RSI > 75):** '탐욕' 구간입니다. 언제든 차익 실현 매물이 나올 수 있어 감점합니다. (-20점)
        """)
        
    with st.expander("⑤ 거래량 (Volume) - 신뢰의 척도", expanded=True):
        st.markdown("""
        * **개념:** 거래량이 없는 상승은 가짜일 수 있습니다. 거래량이 동반된 상승만이 '진짜'입니다.
        * **판단 기준:**
            * **거래량 폭발:** 평소 거래량(20일 평균)보다 1.5배 이상 터지면서 양봉(상승)이 나오면 '세력 유입'으로 봅니다. (+10점)
        """)

    st.divider()
    
    st.header("2. 🚦 AI 판단 등급표 (Decision Matrix)")
    st.markdown("위 5가지 요소의 합산 점수(0~100점)에 따라 최종 행동 지침을 내립니다.")
    
    grade_data = {
        "점수 구간": ["80점 ~ 100점", "60점 ~ 79점", "41점 ~ 59점", "21점 ~ 40점", "0점 ~ 20점"],
        "등급 (Grade)": ["🚀 강력 매수 (Strong Buy)", "📈 매수 (Buy)", "👀 관망 (Hold)", "📉 매도 (Sell)", "💥 강력 매도 (Strong Sell)"],
        "상세 설명": [
            "모든 지표가 상승을 가리킵니다. 추세는 살아있고 가격은 매력적인 '눌림목' 상태일 확률이 높습니다. 적극적으로 비중을 실을 때입니다.",
            "전반적으로 긍정적입니다. 상승 추세에 있거나, 과매도 구간에서 반등을 시작했습니다. 분할 매수로 진입하기 좋습니다.",
            "방향성이 뚜렷하지 않습니다. 호재와 악재가 섞여있거나 횡보장입니다. 신규 진입보다는 추세를 더 지켜봐야 합니다.",
            "위험 신호가 감지됩니다. 추세가 꺾였거나 단기 과열 상태입니다. 이익 실현을 하거나 비중을 줄이는 것이 현명합니다.",
            "매우 위험합니다. 역배열 하락 추세가 가속화되고 있습니다. 가지고 있다면 손절을, 없다면 쳐다보지도 말아야 할 때입니다."
        ]
    }
    st.table(pd.DataFrame(grade_data))
    
    st.divider()
    
    st.header("3. 💸 수수료 및 비용 계산 방식 (토스증권 기준)")
    st.info("이 봇은 단순 등락률이 아닌, 세금과 수수료를 모두 뗀 '실현 손익'을 계산합니다.")
    
    st.markdown("""
    **🇰🇷 국내 주식 (KR)**
    * **증권거래세:** 매도 금액의 `0.15%` (국가 납부)
    * **유관기관 제비용:** 약 `0.03%`
    * **총 비용:** 매도 시 약 **0.18%**가 원금에서 차감됩니다.
    
    **🇺🇸 해외 주식 (US)**
    * **매매 수수료:** 매도 금액의 `0.2%` (토스증권 표준 요율 적용 시)
    * **총 비용:** 매도 시 약 **0.2%**가 원금에서 차감됩니다.
    * *(참고: 환전 수수료는 변동성이 커서 계산에 포함하지 않았습니다)*
    """)
    
    st.warning("⚠️ **면책 조항:** 본 서비스는 투자를 보조하는 도구일 뿐이며, AI의 분석이 100% 정확성을 보장하지 않습니다. 모든 투자 결정의 책임은 본인에게 있습니다.")
