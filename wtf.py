import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import time

@st.cache_data(show_spinner=False)
def load_price_data(code: str, start_date: str):
    """
    yfinance에서 개별 종목 데이터를 받아오는 함수 (캐시됨)
    같은 code, start_date로 다시 호출하면 네트워크를 다시 안 타고
    이전에 받아온 데이터를 그대로 사용해서 결과가 항상 같게 됨.
    """
    df = yf.download(code, start=start_date, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(show_spinner=False)
def load_fx_series(start_date: str):
    """
    KRW=X 환율 시계열 다운로드 (캐시됨)
    Dynamic 모드에서도 같은 start_date면 항상 같은 환율 시계열 사용.
    """
    ex_df = yf.download("KRW=X", start=start_date, progress=False)
    if isinstance(ex_df.columns, pd.MultiIndex):
        ex_df.columns = ex_df.columns.get_level_values(0)
    return ex_df['Close']


def prepare_stock_data(ticker_info, start_date):
    """
    개별 종목의 데이터를 미리 준비하는 함수
    → 네트워크는 load_price_data에서 캐시되므로
      같은 세션/같은 시작일이면 항상 같은 데이터 사용
    """
    code, name = ticker_info
    try:
        # ★ 캐시된 다운로드 사용
        df_raw = load_price_data(code, start_date)
        if df_raw is None or df_raw.empty or len(df_raw) < 60:
            return None

        df = calculate_indicators_for_backtest(df_raw)
        df['AI_Score'] = df.apply(get_ai_score_row, axis=1)
        df['Ticker'] = code
        df['Name'] = name
        
        # ★ STD20까지 돌려줘서 포지션 사이징에 사용
        return df[['Close_Calc', 'AI_Score', 'STD20', 'Ticker', 'Name']]
    except Exception as e:
        # 원하면 로그 찍기
        # st.write(f"{code} 데이터 오류: {e}")
        return None

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

# =========================================================
# 1. 백테스트용 로직 분리 (기존 로직을 Row 단위로 변환)
# =========================================================

def calculate_indicators_for_backtest(df):
    """지표 계산 (기존 함수 재활용 및 최적화)"""
    df = df.copy()
    
    # 수정 종가 사용
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['Close_Calc'] = df[col]

    # 이동평균
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()
    
    # RSI (정밀도 유지)
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
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)
    
    # 변동성 (표준편차)
    df['STD20'] = df['Close_Calc'].rolling(20).std()
    
    return df.dropna()

def get_ai_score_row(row):
    """
    [업그레이드] 초정밀 점수 산정 로직
    - 단순 가산(+10)이 아니라, 이격도와 강도를 소수점 단위로 반영하여
    - 동점자가 나올 확률을 수학적으로 제거함.
    """
    try:
        curr = row['Close_Calc']
        ma5, ma20, ma60 = row['MA5'], row['MA20'], row['MA60']
        rsi = row['RSI']
        macd, sig = row['MACD'], row['Signal_Line']
        std20 = row['STD20']
        macd_hist = row['MACD_Hist']
        
        # 기본 점수 50점에서 시작
        score = 50.0

        # ---------------------------------------------------------
        # 1. 추세 (Trend) - 거리 비례 점수
        # ---------------------------------------------------------
        # 60일선 위에 있으면 기본 +10점이지만,
        # 60일선보다 얼마나 더 위에 있는지(이격도)를 0.001 단위로 더함
        if curr > ma60:
            score += 10.0
            # 이격도 보너스: (가격 - 60일선) / 60일선 * 100
            # 예: 5% 높으면 +5점, 5.1% 높으면 +5.1점
            divergence = (curr - ma60) / ma60 * 100
            # 너무 높으면(15% 이상) 과열이므로 최대 5점까지만 반영
            score += min(5.0, divergence)
        else:
            score -= 20.0
            # 하락폭이 클수록 더 깎음 (미세 조정)
            divergence = (ma60 - curr) / ma60 * 100
            score -= min(5.0, divergence * 0.1)

        # 정배열 강도 체크 (미세 점수)
        # 단순히 정배열이다(+10)가 아니라, 5일선과 20일선의 간격만큼 가산
        if ma5 > ma20 > ma60:
            score += 10.0
            gap_5_20 = (ma5 - ma20) / ma20 * 100 # 간격 %
            score += min(3.0, gap_5_20) # 간격이 넓을수록(상승세가 가파를수록) 최대 3점 추가
        elif ma20 > ma60:
            score += 5.0

        # ---------------------------------------------------------
        # 2. 눌림목 (Pullback) - 근접도 미분
        # ---------------------------------------------------------
        dist_ma20 = (curr - ma20) / ma20
        abs_dist = abs(dist_ma20)

        # 60일선 위 상승 추세에서 20일선에 붙을수록 점수 급증
        if curr > ma60:
            if abs_dist <= 0.03: # 3% 이내
                # 거리가 0에 가까울수록 20점에 수렴 (소수점 반영)
                # 예: 거리 1%면 +13.3점, 거리 0.1%면 +19.3점
                proximity_score = 20.0 * (1.0 - (abs_dist / 0.03))
                score += proximity_score
            elif 0.03 < dist_ma20 <= 0.08:
                score += 5.0
            
            # 여기서 미세 조정: 20일선 위에 있는게 아래 있는것보다 0.1점이라도 유리하게
            if dist_ma20 > 0: score += 0.1

        # 과열 페널티 (10% 이상 이격)
        if dist_ma20 > 0.10:
            # 많이 벌어질수록 더 많이 깎음
            overheat = (dist_ma20 - 0.10) * 100
            score -= (15.0 + overheat)

        # ---------------------------------------------------------
        # 3. RSI - 소수점 반영
        # ---------------------------------------------------------
        # RSI는 그 자체로 소수점이므로 그대로 공식에 대입
        if 40 <= rsi <= 60:
            # 50을 기준으로 점수 부여 (50 -> +10, 60 -> +12)
            score += 10.0 + ((rsi - 40) * 0.1)
        elif 30 <= rsi < 40:
            score += 5.0 + ((40 - rsi) * 0.5)
        elif 60 < rsi <= 70:
            score += 8.0 + ((rsi - 60) * 0.1)
        elif rsi < 30:
            score += 15.0 + ((30 - rsi) * 0.2) # 과매도 심할수록 점수 더 줌
        elif rsi > 70:
            score -= 15.0

        # ---------------------------------------------------------
        # 4. MACD - 에너지 강도
        # ---------------------------------------------------------
        if macd > sig:
            score += 5.0
            # 히스토그램의 크기(에너지)를 점수에 반영 (소수점)
            # 주가 대비 히스토그램 비율 사용
            hist_ratio = (macd_hist / curr) * 1000 
            score += min(3.0, hist_ratio) 
            
            # 상승 가속도 (어제보다 오늘 막대가 더 큰가?)
            if macd_hist > row['Prev_MACD_Hist']:
                score += 2.0
                # 얼마나 더 커졌는지 반영
                growth = (macd_hist - row['Prev_MACD_Hist']) / curr * 10000
                score += min(1.0, growth)
        else:
            score -= 5.0

        # ---------------------------------------------------------
        # 5. 변동성 페널티 (Tie Breaker 역할)
        # ---------------------------------------------------------
        # 변동성이 적은(안정적인) 종목이 우세하도록 세팅
        vol_ratio = std20 / curr if curr > 0 else 0
        
        # 변동성 비율만큼 점수를 미세하게 깎음
        # 예: 변동성 2%면 -2점, 2.1%면 -2.1점
        # -> 점수가 완벽히 같을 때 변동성이 적은 종목이 0.001점이라도 높게 됨
        score -= (vol_ratio * 100.0)

        # 최종 클램핑 (0~100)
        return max(0.0, min(100.0, score))
    except:
        return 0.0

# =========================================================
# 2. 개별 종목 백테스트 엔진 (정리된 최종 버전)
# =========================================================

def prepare_stock_data(ticker_info, start_date):
    """
    개별 종목의 데이터를 미리 준비하는 함수
    → 네트워크는 load_price_data에서 캐시되므로
      같은 세션/같은 시작일이면 항상 같은 데이터 사용
    """
    code, name = ticker_info
    try:
        # ★ 캐시된 다운로드 사용
        df_raw = load_price_data(code, start_date)
        if df_raw is None or df_raw.empty or len(df_raw) < 60:
            return None

        df = calculate_indicators_for_backtest(df_raw)
        df['AI_Score'] = df.apply(get_ai_score_row, axis=1)
        df['Ticker'] = code
        df['Name'] = name
        
        # ★ STD20까지 돌려줘서 포지션 사이징에 사용
        return df[['Close_Calc', 'AI_Score', 'STD20', 'Ticker', 'Name']]
    except Exception as e:
        # 원하면 로그 찍기
        # st.write(f"{code} 데이터 오류: {e}")
        return None


def run_portfolio_backtest(targets, start_date, initial_capital, strategy_mode,
                           max_hold_days, exchange_data, use_compound, selection_mode):
    """
    [수정됨] 현실성 + 재현성 강화:
    - 데이터: 단일 스레드 + 캐시 사용 (항상 같은 유니버스)
    - 매수 우선순위: 점수 내림차순, 동점 시 티커 사전순
    """

    # 1. 전 종목 데이터 준비 (단일 스레드, 순서 고정)
    all_dfs = []
    # 정렬까지 해서 완전 고정하고 싶으면 아래처럼:
    # for t in sorted(targets, key=lambda x: x[0]):
    for t in targets:
        res = prepare_stock_data(t, start_date)
        if res is not None:
            all_dfs.append(res)

    # st.write(f"Loaded Tickers: {len(all_dfs)} / {len(targets)}")

    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame()

    # 2. Market Data 통합 (날짜별로 종목 리스트 모으기)
    market_data = {}
    for df in all_dfs:
        for date, row in df.iterrows():
            if date not in market_data:
                market_data[date] = []
            market_data[date].append(row)

    sorted_dates = sorted(market_data.keys())

    # 3. 환율 데이터 준비
    if isinstance(exchange_data, (float, int)):
        get_rate = lambda d: float(exchange_data)
    else:
        rate_dict = exchange_data.to_dict()

        def get_rate(d):
            ts = pd.Timestamp(d)
            # 해당 날짜가 없으면 1430.0으로 fallback (항상 동일)
            return rate_dict.get(ts, 1430.0)

    # 4. 시뮬레이션 상태 변수
    balance = initial_capital
    portfolio = {}
    trades_log = []
    equity_curve = []

    max_slots = 1 if selection_mode == 'TOP1' else 10

    # --- 날짜별 루프 ---
    for date in sorted_dates:
        daily_stocks = market_data[date]
        current_rate = get_rate(date)

        # A. 매도 (Sell Check)
        sell_list = []

        for ticker in sorted(portfolio.keys()):  # 순서 고정
            info = portfolio[ticker]

            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            if stock_row is None:
                continue

            curr_price_raw = stock_row['Close_Calc']
            curr_price_krw = curr_price_raw * (1.0 if ".KS" in ticker else current_rate)
            score = stock_row['AI_Score']

            fee_sell = 0.003 if ".KS" in ticker else 0.001

            should_sell = False
            sell_reason = ""

            profit_pct = (curr_price_krw - info['avg_price']) / info['avg_price'] * 100
            profit_ratio = (curr_price_krw - info['avg_price']) / info['avg_price']

            # 1) 타임 컷
            if max_hold_days > 0:
                held_days = (date - info['buy_date']).days
                if held_days >= max_hold_days:
                    should_sell = True
                    sell_reason = f"⏱️ TimeCut({held_days}일)"

            # 2) 전략별 매도 로직
            if not should_sell:
                if strategy_mode == "Basic":
                    if score <= 45:
                        should_sell = True
                        sell_reason = "AI 45↓"

                elif strategy_mode == "SuperLocking":
                    if not info['mode_active'] and profit_ratio >= 0.03:
                        portfolio[ticker]['mode_active'] = True
                        portfolio[ticker]['max_price'] = curr_price_krw

                    if info['mode_active']:
                        if curr_price_krw > portfolio[ticker]['max_price']:
                            portfolio[ticker]['max_price'] = curr_price_krw
                        if curr_price_krw <= portfolio[ticker]['max_price'] * 0.98:
                            should_sell = True
                            sell_reason = "💎 락킹 익절"
                    else:
                        if score <= 45:
                            should_sell = True
                            sell_reason = "방어(45↓)"

                elif strategy_mode == "Sniper":
                    if profit_ratio <= -0.03:
                        should_sell = True
                        sell_reason = "⚡ 칼손절(-3%)"
                    elif not info['mode_active'] and profit_ratio >= 0.05:
                        portfolio[ticker]['mode_active'] = True
                        portfolio[ticker]['max_price'] = curr_price_krw

                    if info['mode_active']:
                        if curr_price_krw > portfolio[ticker]['max_price']:
                            portfolio[ticker]['max_price'] = curr_price_krw
                        if curr_price_krw <= portfolio[ticker]['max_price'] * 0.97:
                            should_sell = True
                            sell_reason = "🎯 스나이퍼 익절"

                    if not should_sell and score < 40:
                        should_sell = True
                        sell_reason = "추세 이탈(40↓)"

            if should_sell:
                return_amt = info['shares'] * curr_price_krw * (1 - fee_sell)
                balance += return_amt

                trades_log.append({
                    'ticker': ticker,
                    'name': info['name'],
                    'date': date,
                    'type': 'sell',
                    'price': curr_price_raw,
                    'score': score,
                    'profit': profit_pct,
                    'reason': sell_reason,
                    'balance': balance
                })
                sell_list.append(ticker)

        for t in sell_list:
            del portfolio[t]

        # B. 신규 매수 (Buy Logic)
        if len(portfolio) < max_slots:
            candidates = []

            for row in daily_stocks:
                ticker = row['Ticker']
                if ticker in portfolio:
                    continue

                score = row['AI_Score']
                price_raw = row['Close_Calc']
                price_krw = price_raw * (1.0 if ".KS" in ticker else current_rate)

                entry_signal = False
                reason = ""

                if strategy_mode == "Basic" and score >= 65:
                    entry_signal = True
                    reason = "AI 65↑"
                elif strategy_mode == "SuperLocking" and score >= 80:
                    entry_signal = True
                    reason = "강력매수(80↑)"
                elif strategy_mode == "Sniper" and score >= 70:
                    entry_signal = True
                    reason = "스나이퍼(70↑)"

                if entry_signal:
                    # 변동성 비율(20일 표준편차 / 가격) 계산
                    std20 = row.get('STD20', np.nan)
                    if pd.notna(std20) and price_raw > 0:
                        vol_ratio = float(std20 / price_raw)  # 일간 변동성 %
                    else:
                        vol_ratio = np.nan

                    candidates.append({
                        'ticker': ticker,
                        'name': row['Name'],
                        'price_raw': price_raw,
                        'price_krw': price_krw,
                        'score': score,
                        'vol_ratio': vol_ratio,
                        'reason': reason
                    })

            # 점수 내림차순, 동점 시 티커 사전순
            candidates.sort(key=lambda x: (x['score'], x['ticker']), reverse=True)

            open_slots = max_slots - len(portfolio)
            buy_targets = candidates[:open_slots]

            if buy_targets:
                # -------------------------------------------------
                # ① 기존 방식으로 "총 투자 예산" 먼저 결정
                #    - use_compound=True  : 남은 현금 balance 기준
                #    - use_compound=False : 초기자본 / 슬롯 기준
                # -------------------------------------------------
                if use_compound:
                    base_per_stock_budget = balance / max(open_slots, 1)
                else:
                    base_per_stock_budget = min(balance, initial_capital / max_slots)

                # 예전엔 per_stock_budget * len(buy_targets) 만큼 투자했으니,
                # 총 예산도 그 수준에 맞춰서 유지
                total_budget = min(balance, base_per_stock_budget * len(buy_targets))

                # -------------------------------------------------
                # ② 각 후보별 "위험-보상 가중치" 계산
                #    weight = (점수 - 50) / 변동성
                #    → 점수 높고, 변동성 낮을수록 더 많이 배정
                # -------------------------------------------------
                weights = []
                for target in buy_targets:
                    # 점수 50점을 기준으로, 그 이상만 강점으로 사용
                    score_component = max(1.0, target['score'] - 50.0)

                    vol = target.get('vol_ratio', None)
                    if vol is None or not np.isfinite(vol) or vol <= 0:
                        vol = 0.03  # 기본 3% 변동성 가정

                    vol = float(vol)
                    # 말도 안 되게 작거나 큰 값 방지 (0.5% ~ 10% 사이로 자름)
                    vol = max(0.005, min(vol, 0.10))

                    # 점수 ↑, 변동성 ↓ → weight 커짐
                    weight = score_component / vol
                    weights.append(weight)

                weight_sum = float(sum(weights))
                if weight_sum <= 0:
                    # 혹시 모를 예외: 전부 0이면 균등 배분
                    weights = [1.0 for _ in buy_targets]
                    weight_sum = float(len(buy_targets))

                # -------------------------------------------------
                # ③ 가중치 비율대로 총 예산을 나눠서 "몇 주 살지" 결정
                # -------------------------------------------------
                for target, w in zip(buy_targets, weights):
                    if total_budget <= 0 or balance <= 0:
                        break

                    # 이 종목에 배정된 이론상 예산
                    target_budget = total_budget * (w / weight_sum)

                    # 실제 사용 가능한 현금 한도 내에서만 사용
                    budget = min(balance, target_budget)
                    fee_buy = 0.00015 if ".KS" in target['ticker'] else 0.001

                    if target['price_krw'] > 0:
                        shares = int(budget / (target['price_krw'] * (1 + fee_buy)))
                    else:
                        shares = 0

                    if shares > 0:
                        cost = shares * target['price_krw'] * (1 + fee_buy)
                        balance -= cost
                        total_budget -= cost  # 전체 예산에서도 차감

                        portfolio[target['ticker']] = {
                            'name': target['name'],
                            'shares': shares,
                            'avg_price': target['price_krw'],
                            'buy_date': date,
                            'mode_active': False,
                            'max_price': 0
                        }

                        trades_log.append({
                            'ticker': target['ticker'],
                            'name': target['name'],
                            'date': date,
                            'type': 'buy',
                            'price': target['price_raw'],
                            'score': target['score'],
                            'profit': 0,
                            'reason': target['reason'],
                            'balance': balance
                        })

        # C. 자산 평가
        current_equity = balance
        for ticker in sorted(portfolio.keys()):  # 순서 고정
            info = portfolio[ticker]
            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            if stock_row is not None:
                p_raw = stock_row['Close_Calc']
                p_krw = p_raw * (1.0 if ".KS" in ticker else current_rate)
                current_equity += info['shares'] * p_krw
            else:
                current_equity += info['shares'] * info['avg_price']

        equity_curve.append({'date': date, 'equity': current_equity})

    return pd.DataFrame(trades_log), pd.DataFrame(equity_curve)
                               
# =========================================================
# 3. UI 통합 (탭 추가)
# =========================================================
# (기존 코드의 tab1, tab2, tab3 정의 아래에 tab4를 추가한다고 가정)

tab4 = st.tabs(["📊 전체 백테스트 시뮬레이션"])[0] # 기존 tabs 리스트에 추가 필요

with tab4:
    st.markdown("### 🧪 포트폴리오 유니버스 백테스트")
    st.caption("AI 전략 시뮬레이터 Final Ver. (일봉 종가 기준 / 동시 호가 반영)")
    
    # --------------------------------------------------------------------------------
    # 1. 설정 패널
    # --------------------------------------------------------------------------------
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        bt_start_date = st.date_input("시작일", value=pd.to_datetime("2024-01-01"))
        max_hold_days = st.slider("⏱️ 타임 컷 (일)", 0, 60, 0, help="매수 후 N일 지나면 강제 매도")
    with r1_c2:
        initial_cap_input = st.number_input("💰 초기 자본금", value=10000000, step=1000000, format="%d")
        sel_mode = st.selectbox("🎯 종목 선정", ["조건 만족 전부 매수 (분산)", "점수 1등만 매수 (집중)"])
        selection_code = "TOP1" if "집중" in sel_mode else "ALL"
    with r1_c3:
        ex_method = st.radio("💱 환율 방식", ["실시간 변동 (Dynamic)", "고정 환율 (Fixed)"])
        if "고정" in ex_method:
            fixed_exchange_rate = st.number_input("환율 (원/$)", value=1430.0, step=10.0, format="%.1f")
            exchange_arg_val = fixed_exchange_rate
        else:
            exchange_arg_val = "DYNAMIC"

    st.divider()
    
    # 전략 및 옵션
    c_strat, c_opt, c_btn = st.columns([2, 1, 1])
    with c_strat:
        selected_strategy = st.radio(
            "⚔️ 매매 전략 선택", 
            ["AI 스나이퍼 (추천)", "슈퍼 락킹 (안전)", "기본 모드 (장투)"],
            captions=[
                "70점 진입 / -3% 손절 / +5% 후 트레일링", 
                "80점 진입 / +3% 후 타이트 익절", 
                "65점 진입 / 45점 이탈 시 매도"
            ],
            horizontal=True
        )
        if "스나이퍼" in selected_strategy: strat_code = "Sniper"
        elif "슈퍼" in selected_strategy: strat_code = "SuperLocking"
        else: strat_code = "Basic"
        
    with c_opt:
        comp_mode = st.checkbox("복리 투자 (재투자)", value=True)
    with c_btn:
        st.write("")
        start_btn = st.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True)

    # --------------------------------------------------------------------------------
    # 2. 실행 로직
    # --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
    # 2. 실행 로직 (세션 스테이트를 사용하여 결과 저장)
    # --------------------------------------------------------------------------------
    
    # 세션 상태 초기화 (결과 저장용 변수가 없으면 만듦)
    if 'bt_result_trade' not in st.session_state:
        st.session_state['bt_result_trade'] = pd.DataFrame()
    if 'bt_result_equity' not in st.session_state:
        st.session_state['bt_result_equity'] = pd.DataFrame()

    if start_btn:
        # 환율 준비
        if exchange_arg_val == "DYNAMIC":
            with st.spinner("💱 환율 데이터 수집 중..."):
                exchange_data_payload = load_fx_series(str(bt_start_date))
        else:
            exchange_data_payload = float(exchange_arg_val)

        # 시뮬레이션 실행
        with st.spinner(f"🔄 [{selected_strategy}] 전략으로 전체 시장 스캔 중..."):
            targets = list(TICKER_MAP.items())
            
            # 백테스트 함수 실행
            t_df, e_df = run_portfolio_backtest(
                targets, str(bt_start_date), initial_cap_input, strat_code, 
                max_hold_days, exchange_data_payload, comp_mode, selection_code
            )
            
            # ★ 핵심: 결과를 세션 스테이트에 저장 (화면이 리로드되어도 안 사라짐)
            st.session_state['bt_result_trade'] = t_df
            st.session_state['bt_result_equity'] = e_df
            
            # 완료 메시지 (잠깐 떴다 사라짐)
            st.success("백테스트 완료! 결과를 확인하세요.")

    # --------------------------------------------------------------------------------
    # 3. 결과 대시보드 (저장된 데이터가 있으면 출력)
    # --------------------------------------------------------------------------------
    
    # 버튼을 눌렀든 안 눌렀든, 저장된 결과가 있으면 변수에 할당하여 화면에 표시
    trade_df = st.session_state['bt_result_trade']
    equity_df = st.session_state['bt_result_equity']

    # 데이터가 비어있지 않을 때만 대시보드 렌더링
    if not trade_df.empty and not equity_df.empty:
            # --- 추가 지표 계산 ---
            equity_df['max_equity'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['max_equity']) / equity_df['max_equity'] * 100
            mdd = equity_df['drawdown'].min()

            final_equity = equity_df.iloc[-1]['equity']
            total_return = (final_equity - initial_cap_input) / initial_cap_input * 100
            profit_amt = final_equity - initial_cap_input
            
            sells = trade_df[trade_df['type'] == 'sell']
            win_count = len(sells[sells['profit'] > 0])
            total_sells = len(sells)
            win_rate = (win_count / total_sells * 100) if total_sells > 0 else 0.0

            # ---------------------------
            # [섹션 A] 핵심 성과 지표 (KPI)
            # ---------------------------
            st.markdown("#### 🚀 백테스트 요약 리포트")
            
            with st.container(border=True):
                k1, k2, k3, k4, k5 = st.columns(5)
                
                k1.metric("최종 자산", f"{final_equity/10000:,.0f}만원", 
                          delta=f"{profit_amt/10000:,.0f}만원", delta_color="normal")
                
                k2.metric("총 수익률", f"{total_return:,.2f}%", 
                          delta="복리 적용" if comp_mode else "단리 적용")
                
                k3.metric("승률 (Win Rate)", f"{win_rate:.1f}%", 
                          f"{win_count}승 {total_sells-win_count}패")
                
                k4.metric("MDD (최대낙폭)", f"{mdd:.2f}%", 
                          "Risk Level", delta_color="off")
                
                k5.metric("총 매매 횟수", f"{len(trade_df)//2}회", 
                          f"평균 {len(trade_df)//2 / len(equity_df) * 5:.1f}회/주")

            # ---------------------------
            # [섹션 B] 자산 성장 그래프 (테마 적응형)
            # ---------------------------
            st.markdown("#### 📈 자산 성장 & MDD 추이")
            
            tab_g1, tab_g2 = st.tabs(["💰 자산 커브 (Equity)", "💧 낙폭 (Drawdown)"])
            
            # 공통 레이아웃 설정 (투명 배경 + 반투명 그리드)
            common_layout = dict(
                paper_bgcolor='rgba(0,0,0,0)',  # 전체 배경 투명
                plot_bgcolor='rgba(0,0,0,0)',   # 차트 영역 투명
                font=dict(color=None),          # 폰트색: None으로 두면 Streamlit 테마 자동 추적
                xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'), # 그리드: 연한 회색 (양쪽 모드 호환)
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                hovermode="x unified"
            )

            with tab_g1:
                fig = px.line(equity_df, x='date', y='equity', title=None, height=350)
                fig.add_hline(y=initial_cap_input, line_dash="dash", line_color="gray", annotation_text="원금")
                
                # 라인 색상: 민트색 (다크/라이트 모두 잘 보임)
                fig.update_traces(line=dict(color='#00CC96', width=2), fill='tozeroy') 
                fig.update_layout(xaxis_title="", yaxis_title="평가 금액 (원)", **common_layout)
                st.plotly_chart(fig, use_container_width=True)
                
            with tab_g2:
                fig_dd = px.area(equity_df, x='date', y='drawdown', title=None, height=350)
                # 낙폭 색상: 붉은 계열 (경고 의미)
                fig_dd.update_traces(line=dict(color='#EF553B'), fillcolor='rgba(239, 85, 59, 0.2)')
                
                y_min = mdd * 1.2 if mdd < 0 else -5.0
                fig_dd.update_layout(xaxis_title="", yaxis_title="낙폭 (%)", yaxis_range=[y_min, 1], **common_layout)
                st.plotly_chart(fig_dd, use_container_width=True)

            st.divider()

            # ---------------------------
            # [섹션 C] 매매 상세 분석
            # ---------------------------
            c_left, c_right = st.columns([1, 1.5])
            
            with c_left:
                st.markdown("#### 🏆 Best & Worst")
                if not sells.empty:
                    best_trade = sells.loc[sells['profit'].idxmax()]
                    worst_trade = sells.loc[sells['profit'].idxmin()]
                    
                    with st.container(border=True):
                        st.caption("🔥 최고의 매매")
                        st.markdown(f"**{best_trade['name']}**")
                        # 빨간색/파란색 텍스트 대신 Streamlit 기본 컬러 사용 (가독성 확보)
                        st.metric("수익률", f"{best_trade['profit']:.2f}%", best_trade['reason'])
                        
                    with st.container(border=True):
                        st.caption("💧 최악의 매매")
                        st.markdown(f"**{worst_trade['name']}**")
                        st.metric("수익률", f"{worst_trade['profit']:.2f}%", worst_trade['reason'], delta_color="inverse") # inverse: 하락이 빨강(나쁨) 표시
                else:
                    st.info("매도 완료된 거래가 없습니다.")

            with c_right:
                st.markdown("#### 🔍 종목별 타점 복기")
                traded_tickers = trade_df['ticker'].unique()
                ticker_options = [f"{TICKER_MAP.get(t, t)} ({t})" for t in traded_tickers]
                
                if len(ticker_options) > 0:
                    selected_option = st.selectbox("종목 선택", ticker_options, label_visibility="collapsed")
                    selected_ticker = selected_option.split('(')[-1].replace(')', '')
                    
                    # 데이터 로딩
                    my_trades = trade_df[trade_df['ticker'] == selected_ticker].sort_values('date')
                    with st.spinner("차트 로딩..."):
                        chart_data = yf.download(selected_ticker, start=str(bt_start_date), progress=False, auto_adjust=True)
                        if isinstance(chart_data.columns, pd.MultiIndex):
                            chart_data.columns = chart_data.columns.get_level_values(0)
                        chart_data = chart_data.loc[:, ~chart_data.columns.duplicated()]

                    if not chart_data.empty:
                        fig_d = go.Figure()
                        
                        # 주가 라인: 테마에 따라 자동 조정되도록 회색 계열 사용하되 약간 진하게
                        fig_d.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'], 
                                                   mode='lines', name='주가', 
                                                   line=dict(color='#888888', width=1.5)))
                        
                        # 매수: 빨강 (표준)
                        buys = my_trades[my_trades['type'] == 'buy']
                        if not buys.empty:
                            fig_d.add_trace(go.Scatter(x=buys['date'], y=buys['price'], mode='markers', name='매수', 
                                                       marker=dict(symbol='triangle-up', color='#FF4B4B', size=11), # 가시성 높은 빨강
                                                       hovertemplate='매수: %{y:,.0f}<br>날짜: %{x}'))
                        # 매도: 파랑 (표준)
                        sells_sub = my_trades[my_trades['type'] == 'sell']
                        if not sells_sub.empty:
                            fig_d.add_trace(go.Scatter(x=sells_sub['date'], y=sells_sub['price'], mode='markers', name='매도', 
                                                       marker=dict(symbol='triangle-down', color='#1C83E1', size=11), # 가시성 높은 파랑
                                                       text=[f"{p:.1f}%" for p in sells_sub['profit']], 
                                                       hovertemplate='매도: %{y:,.0f}<br>수익: %{text}'))
                        
                        fig_d.update_layout(
                            title=dict(text=f"{selected_option} 매매 타점", font=dict(size=15)),
                            height=350, 
                            margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            **common_layout # 위에서 정의한 공통 투명 레이아웃 적용
                        )
                        st.plotly_chart(fig_d, use_container_width=True)
                    else:
                        st.warning("차트 데이터를 불러올 수 없습니다.")
                else:
                    st.info("거래 내역이 없습니다.")

            st.divider()

            # ---------------------------
            # [섹션 D] 전체 거래 일지
            # ---------------------------
            st.subheader("📝 전체 거래 로그")
            
            with st.expander("전체 거래 내역 (펼치기/접기)", expanded=True):
                log_df = trade_df.copy()
                log_df['date'] = log_df['date'].dt.date
                log_df = log_df[['date', 'name', 'type', 'price', 'profit', 'score', 'reason']]
                log_df.columns = ['날짜', '종목명', '구분', '가격', '수익률', 'AI점수', '매매사유']

                st.dataframe(
                    log_df.sort_values('날짜', ascending=False),
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                    column_config={
                        "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
                        "가격": st.column_config.NumberColumn("체결가", format="%.0f"), # 원화 기준이므로 소수점 제거
                        "AI점수": st.column_config.ProgressColumn("AI Score", format="%.0f점", min_value=0, max_value=100),
                        "수익률": st.column_config.NumberColumn("수익률(%)", format="%.2f%%"),
                        "구분": st.column_config.TextColumn("Type", width="small")
                    }
                )
    else:
            st.warning("⚠️ 매매 신호가 발생하지 않았습니다. 전략 조건을 완화하거나 기간을 늘려보세요.")
