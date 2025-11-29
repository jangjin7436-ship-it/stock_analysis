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
from bs4 import BeautifulSoup

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

SEARCH_LIST = [f"{name} ({code})" for code, name in TICKER_MAP.items()]
SEARCH_MAP = {f"{name} ({code})": code for code, name in TICKER_MAP.items()}
USER_WATCHLIST = list(TICKER_MAP.keys())

# ---------------------------------------------------------
# 2. 데이터 수집 (네이버 금융 크롤링 + YF Fast Info)
# ---------------------------------------------------------
def fetch_kr_realtime(ticker):
    """한국 주식 실시간 가격 크롤링 (네이버 금융)"""
    try:
        code = ticker.split('.')[0]
        url = f"https://finance.naver.com/item/sise.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        price_str = soup.select_one('#_nowVal').text.replace(',', '')
        return (ticker, float(price_str))
    except:
        try:
            df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
            if not df.empty:
                return (ticker, float(df['Close'].iloc[-1]))
        except:
            pass
        return (ticker, None)

def fetch_us_realtime(ticker):
    """미국 주식: 실시간/애프터마켓 가격 (fast_info)"""
    try:
        price = yf.Ticker(ticker).fast_info['last_price']
        return (ticker, price)
    except:
        return (ticker, None)

def fetch_history_data(ticker):
    """지표 분석용 과거 데이터 (2년치) - 안전한 데이터 평탄화 적용"""
    try:
        if ticker.endswith('.KS') or ticker.endswith('.KQ'):
            df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        else:
            df = yf.download(ticker, period="2y", progress=False)
            
            # [안전장치 1] MultiIndex 평탄화 (값 손실 없이 구조만 단순화)
            if isinstance(df.columns, pd.MultiIndex):
                # 레벨 1(Ticker)이 있다면 제거, 없다면 레벨 0 유지
                try:
                    df.columns = df.columns.droplevel(1)
                except:
                    pass
            
            # [안전장치 2] 중복 컬럼 제거 (Close가 두 개 생기는 버그 방지)
            df = df.loc[:, ~df.columns.duplicated()]

            # [안전장치 3] 컬럼명 표준화
            if 'Close' not in df.columns and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
                
        return (ticker, df)
    except:
        return (ticker, None)

@st.cache_data(ttl=5) # 5초 캐시
def get_hybrid_data_v3(tickers_list):
    """실시간 가격(크롤링/FastInfo) + 과거 차트 데이터 병합"""
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]
    
    final_dfs = {} 

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_realtime = []
        for t in kr_tickers:
            future_realtime.append(executor.submit(fetch_kr_realtime, t))
        for t in us_tickers:
            future_realtime.append(executor.submit(fetch_us_realtime, t))
            
        future_history = []
        for t in tickers_list:
            future_history.append(executor.submit(fetch_history_data, t))
            
        realtime_map = {}
        for f in concurrent.futures.as_completed(future_realtime):
            tk, price = f.result()
            if price is not None: realtime_map[tk] = price
            
        history_map = {}
        for f in concurrent.futures.as_completed(future_history):
            tk, df = f.result()
            if df is not None and not df.empty: history_map[tk] = df

    for t in tickers_list:
        if t in history_map:
            df = history_map[t].copy()
            
            # 분석 전 데이터 컬럼 재확인
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if t in realtime_map:
                latest_price = realtime_map[t]
                if 'Close' in df.columns:
                    # 마지막 종가를 실시간 가격으로 덮어씀 (분석 정확도 향상)
                    df.iloc[-1, df.columns.get_loc('Close')] = latest_price
            final_dfs[t] = df

    return final_dfs, realtime_map

def calculate_indicators(df):
    if len(df) < 60: return None
    df = df.copy()
    
    # [안전장치 4] Series인지 DataFrame인지 확인하여 단일 컬럼 보장
    if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
        if isinstance(df['Close'], pd.DataFrame):
            close_series = df['Close'].iloc[:, 0]
        else:
            close_series = df['Close']
    else:
        return None

    close_series = close_series.ffill()
    df['Close_Calc'] = close_series

    df['MA20'] = df['Close_Calc'].rolling(window=20).mean()
    df['MA60'] = df['Close_Calc'].rolling(window=60).mean()
    
    if 'Volume' in df.columns:
        vol = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
        df['VolMA20'] = vol.rolling(window=20).mean()
    else:
        df['VolMA20'] = 0

    delta = df['Close_Calc'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['STD20'] = df['Close_Calc'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)
    
    return df.dropna()

def calculate_total_profit(ticker, avg_price, current_price, quantity):
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
    if is_kr: fee_tax_rate = 0.0018 
    else: fee_tax_rate = 0.002
    
    total_buy = avg_price * quantity
    raw_eval = current_price * quantity
    total_fee = raw_eval * fee_tax_rate
    net_eval = raw_eval - total_fee
    net_profit_amt = net_eval - total_buy
    
    if total_buy > 0:
        net_profit_pct = (net_profit_amt / total_buy) * 100
    else:
        net_profit_pct = 0.0
    
    currency = "₩" if is_kr else "$"
    
    return {
        "pct": net_profit_pct,
        "profit_amt": net_profit_amt,
        "net_eval_amt": net_eval,
        "currency": currency
    }

# ---------------------------------------------------------
# 3. 전략 분석 (안전한 타입 변환 적용)
# ---------------------------------------------------------
def analyze_advanced_strategy(df):
    if df is None or df.empty: return "분석 불가", "gray", "데이터 부족", 0
    
    # [안전장치 5] float() 강제 형변환으로 모호성 제거 (ValueError 해결)
    try:
        curr = float(df['Close_Calc'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        ma60 = float(df['MA60'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        sig = float(df['Signal_Line'].iloc[-1])
        bb_upper = float(df['BB_Upper'].iloc[-1])
        bb_lower = float(df['BB_Lower'].iloc[-1])
        
        prev_macd = float(df['MACD'].iloc[-2])
        prev_sig = float(df['Signal_Line'].iloc[-2])
        
        vol = float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0.0
        vol_ma = float(df['VolMA20'].iloc[-1]) if 'VolMA20' in df.columns else 0.0
        open_price = float(df['Open'].iloc[-1]) if 'Open' in df.columns else curr

    except Exception as e:
        return "데이터 오류", "gray", "지표 계산 실패", 0

    trend_up = curr > ma60
    above_ma20 = curr > ma20
    golden_cross = (macd > sig) and (prev_macd <= prev_sig)
    dead_cross = (macd < sig) and (prev_macd >= prev_sig)
    oversold = rsi < 35
    overbought = rsi > 70
    dist_to_ma20 = (curr - ma20) / ma20
    dip_buy = trend_up and abs(dist_to_ma20) <= 0.02

    score = 50
    reasons = []

    if curr > ma60:
        score += 15
        if curr > ma2
