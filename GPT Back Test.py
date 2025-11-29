import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import time

st.set_page_config(page_title="AI 전략 스윙 백테스터", layout="wide")

# =========================================================
# 0. 데이터 로딩 (캐시)
# =========================================================

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


# =========================================================
# 티커 맵
# =========================================================

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
# 1. 지표 계산 (2주 스윙 기준)
# =========================================================

def calculate_indicators_for_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """지표 계산 (2주 스윙 기준 Ret5 추가)"""
    df = df.copy()
    
    # 수정 종가 사용
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['Close_Calc'] = df[col]

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

    # MACD
    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)
    
    # 변동성 (표준편차)
    df['STD20'] = df['Close_Calc'].rolling(20).std()
    
    # 2주(10영업일) 관점용 단기 모멘텀 (최근 5일 수익률)
    df['Ret5'] = df['Close_Calc'].pct_change(5)

    # 거래량 비율 (20일 평균 대비 Volume Ratio)
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Vol_MA20']
    else:
        df['Volume_Ratio'] = np.nan
    
    return df.dropna()


# =========================================================
# 2. AI 점수 (2주 스윙 최적화)
# =========================================================

def get_ai_score_row(row: pd.Series) -> float:
    """
    2주 스윙 기준 AI 점수:
    - 상승 추세 + 20일선 근처 눌림
    - 적당한 RSI 구간
    - 최근 5일 모멘텀
    - MACD 방향
    - 변동성 페널티
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


# =========================================================
# 3. 개별 종목 데이터 준비
# =========================================================

def prepare_stock_data(ticker_info, start_date: str):
    """
    개별 종목 데이터 준비
    - 캐시된 load_price_data 사용
    - AI_Score 포함 각종 지표를 한 번에 계산
    """
    code, name = ticker_info
    try:
        df_raw = load_price_data(code, start_date)
        if df_raw is None or df_raw.empty or len(df_raw) < 60:
            return None

        df = calculate_indicators_for_backtest(df_raw)
        df['AI_Score'] = df.apply(get_ai_score_row, axis=1)
        df['Ticker'] = code
        df['Name'] = name
        
        # 백테스트에서 사용할 컬럼들만 반환
        return df[[
            'Close_Calc', 'MA5', 'MA20', 'MA60',
            'RSI', 'MACD', 'Signal_Line', 'MACD_Hist', 'Prev_MACD_Hist',
            'STD20', 'Ret5', 'Volume_Ratio', 'AI_Score', 'Ticker', 'Name'
        ]]
    except Exception:
        return None


# =========================================================
# 4. 포트폴리오 백테스트 (2주 스윙)
# =========================================================

def run_portfolio_backtest(targets, start_date, initial_capital, strategy_mode,
                           max_hold_days, exchange_data, use_compound, selection_mode):
    """
    2주 이내 스윙 트레이딩 기준 포트폴리오 백테스트
    - 보유일: 최대 14일 (슬라이더가 더 길어도 캡)
    - 추세 + 눌림 + 모멘텀 + 변동성 기반 AI 점수 사용
    """
    # 1. 전 종목 데이터 준비
    all_dfs = []
    for t in targets:
        res = prepare_stock_data(t, start_date)
        if res is not None:
            all_dfs.append(res)

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
            # 해당 날짜가 없으면 1430.0으로 fallback
            return rate_dict.get(ts, 1430.0)

    # 4. 시뮬레이션 상태 변수
    balance = float(initial_capital)
    portfolio = {}
    trades_log = []
    equity_curve = []

    max_slots = 1 if selection_mode == 'TOP1' else 10
    max_hold_cap = 14  # 보유일 상한 (2주, 캘린더 기준)

    # --- 날짜별 루프 ---
    for date in sorted_dates:
        daily_stocks = market_data[date]
        current_rate = get_rate(date)

        # A. 매도 (Sell Check)
        sell_list = []

        for ticker in sorted(portfolio.keys()):
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

            # 1) 시간 제한 (최대 2주, 슬라이더가 더 길어도 캡)
            held_days = (date - info['buy_date']).days
            if max_hold_days > 0:
                effective_hold = min(max_hold_days, max_hold_cap)
            else:
                effective_hold = max_hold_cap

            if held_days >= effective_hold:
                should_sell = True
                sell_reason = f"⏱️ TimeCut({held_days}일)"

            # 2) 전략별 빠른 손절 (Sniper만 -3% 우선 적용)
            if (not should_sell) and strategy_mode == "Sniper" and profit_ratio <= -0.03:
                should_sell = True
                sell_reason = "⚡ 스나이퍼 손절(-3%)"

            # 3) 공통 손절 (-5%) : 모든 전략 공통
            if (not should_sell) and profit_ratio <= -0.05:
                should_sell = True
                sell_reason = "🛑 공통 손절(-5%)"

            # 4) 전략별 추가 매도 규칙
            if not should_sell:
                if strategy_mode == "Basic":
                    # 2주 안에 +10% 정도면 익절, 점수 급락 시 방어 매도
                    if profit_ratio >= 0.10:
                        should_sell = True
                        sell_reason = "기본 익절(+10%)"
                    elif score <= 48:
                        should_sell = True
                        sell_reason = "AI 48↓(추세 약화)"

                elif strategy_mode == "SuperLocking":
                    # +4% 이상 수익 시 락 모드 진입 → 이후 3% 역행 시 익절
                    if not info.get('mode_active', False) and profit_ratio >= 0.04:
                        info['mode_active'] = True
                        info['max_price'] = curr_price_krw

                    if info.get('mode_active', False):
                        if curr_price_krw > info.get('max_price', curr_price_krw):
                            info['max_price'] = curr_price_krw
                        if curr_price_krw <= info['max_price'] * 0.97:
                            should_sell = True
                            sell_reason = "💎 락킹 익절(-3% 트레일링)"
                    else:
                        # 아직 수익이 많이 나지 않았는데 점수가 꺾이면 보수적으로 정리
                        if score <= 50 and profit_ratio >= 0.0:
                            should_sell = True
                            sell_reason = "방어(점수 하락)"

                elif strategy_mode == "Sniper":
                    # +5% 이상 이익 나면 더 타이트한 트레일링 (3.5%) 시작
                    if not info.get('mode_active', False) and profit_ratio >= 0.05:
                        info['mode_active'] = True
                        info['max_price'] = curr_price_krw

                    if info.get('mode_active', False):
                        if curr_price_krw > info.get('max_price', curr_price_krw):
                            info['max_price'] = curr_price_krw
                        if curr_price_krw <= info['max_price'] * 0.965:
                            should_sell = True
                            sell_reason = "🎯 스나이퍼 익절(-3.5% 트레일링)"

                    # 점수가 너무 떨어지면 추세 이탈로 보고 정리
                    if (not should_sell) and score < 45:
                        should_sell = True
                        sell_reason = "추세 이탈(45↓)"

            if should_sell:
                return_amt = info['shares'] * curr_price_krw * (1 - fee_sell)
                balance += return_amt

                trades_log.append({
                    'ticker': ticker,
                    'name': info['name'],
                    'date': date,
                    'type': 'sell',
                    'price': curr_price_raw,
                    'shares': info['shares'],
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

                ma20 = row['MA20']
                ma60 = row['MA60']
                rsi = row['RSI']
                macd = row['MACD']
                sig = row['Signal_Line']
                hist = row['MACD_Hist']

                # 추세 필터: 60일선 위, 20일선도 60일선 위
                dist20 = (price_raw - ma20) / ma20 if ma20 > 0 else 0.0
                trend_ok = (price_raw > ma60) and (ma20 > ma60)
                pullback_ok = (-0.03 <= dist20 <= 0.03)  # 20일선 ±3% 근처
                rsi_ok = (35 <= rsi <= 65)               # 과매수/과매도 피함
                macd_ok = (macd > sig and hist > 0)      # 시그널 상향 돌파 후 양의 히스토그램

                base_entry = trend_ok and pullback_ok and rsi_ok and macd_ok

                entry_signal = False
                reason = ""

                if strategy_mode == "Basic":
                    if base_entry and score >= 65:
                        entry_signal = True
                        reason = "기본 진입(65↑ & 추세 양호)"

                elif strategy_mode == "SuperLocking":
                    if base_entry and score >= 72:
                        entry_signal = True
                        reason = "안전 진입(72↑ & 추세)"

                elif strategy_mode == "Sniper":
                    # 스나이퍼는 단기 모멘텀도 체크
                    ret5 = row.get('Ret5', 0.0)
                    if base_entry and score >= 70 and ret5 >= -0.02:  # 최근 5일 -2% 이내
                        entry_signal = True
                        reason = "스나이퍼 진입(70↑ & 단기 모멘텀)"

                if entry_signal:
                    # 변동성 비율(20일 표준편차 / 가격) 계산
                    std20 = row.get('STD20', np.nan)
                    if pd.notna(std20) and price_raw > 0:
                        vol_ratio = float(std20 / price_raw)
                    else:
                        vol_ratio = np.nan

                    # 거래량 비율 (Volume_Ratio) 가져오기
                    volume_ratio = row.get('Volume_Ratio', np.nan)
                    try:
                        volume_ratio = float(volume_ratio)
                    except (TypeError, ValueError):
                        volume_ratio = np.nan

                    candidates.append({
                        'ticker': ticker,
                        'name': row['Name'],
                        'price_raw': price_raw,
                        'price_krw': price_krw,
                        'score': score,
                        'vol_ratio': vol_ratio,
                        'volume_ratio': volume_ratio,
                        'reason': reason
                    })

            # AI 점수 100점 종목이 5개 초과인 경우: 거래량 비율 상위 5개만 후보로 사용
            if len(candidates) > 0:
                ai100_list = [c for c in candidates if c.get('score', 0) >= 100.0]
                if len(ai100_list) > 5:
                    # Volume Ratio가 없는 경우 0으로 처리
                    for c in ai100_list:
                        vr = c.get('volume_ratio', np.nan)
                        if not (isinstance(vr, (int, float)) and np.isfinite(vr)):
                            c['volume_ratio'] = 0.0
                    ai100_sorted = sorted(ai100_list, key=lambda x: x['volume_ratio'], reverse=True)
                    top5_tickers = {c['ticker'] for c in ai100_sorted[:5]}
                    candidates = [c for c in candidates if c['ticker'] in top5_tickers]

            # 점수 내림차순, 동점 시 티커 사전순
            candidates.sort(key=lambda x: (x['score'], x['ticker']), reverse=True)

            open_slots = max_slots - len(portfolio)
            buy_targets = candidates[:open_slots]

            if buy_targets:
                # ① "총 투자 예산" 먼저 결정
                if use_compound:
                    base_per_stock_budget = balance / max(open_slots, 1)
                else:
                    base_per_stock_budget = min(balance, initial_capital / max_slots)

                total_budget = min(balance, base_per_stock_budget * len(buy_targets))

                # ② 각 후보별 위험-보상 가중치 계산
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

                # ③ 가중치 비율대로 총 예산을 나눠서 "몇 주 살지" 결정
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
                            'price': target['price_raw'],   # 체결 단가
                            'shares': shares,               # 이번에 산 수량
                            'score': target['score'],
                            'profit': 0,
                            'reason': target['reason'],
                            'balance': balance
                        })

        # C. 자산 평가
        current_equity = balance
        for ticker in sorted(portfolio.keys()):
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
# 5. UI (단일 탭: 전체 백테스트 시뮬레이션)
# =========================================================

st.title("📊 AI 스윙 전략 포트폴리오 백테스터")

# 여기서 탭을 직접 생성하고 첫 번째 탭을 tab 변수로 받음
tab = st.tabs(["📊 전체 백테스트 시뮬레이션"])[0]

with tab:
    st.markdown("### 🧪 포트폴리오 유니버스 백테스트")
    st.caption("AI 전략 시뮬레이터 Final Ver. (일봉 종가 기준 / 2주 스윙)")

    # --------------------------------------------------------------------------------
    # 1. 설정 패널
    # --------------------------------------------------------------------------------
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        bt_start_date = st.date_input("시작일", value=pd.to_datetime("2024-01-01"))
        max_hold_days = st.slider("⏱️ 타임 컷 (일)", 0, 60, 14, help="매수 후 N일 지나면 강제 매도")
    with r1_c2:
        initial_cap_input = st.number_input("💰 초기 자본금", value=10_000_000, step=1_000_000, format="%d")
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
            horizontal=True
        )
        if "스나이퍼" in selected_strategy: 
            strat_code = "Sniper"
        elif "슈퍼" in selected_strategy: 
            strat_code = "SuperLocking"
        else: 
            strat_code = "Basic"
        
    with c_opt:
        comp_mode = st.checkbox("복리 투자 (재투자)", value=True)
    with c_btn:
        st.write("")
        start_btn = st.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True)

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
            
            # 결과를 세션 스테이트에 저장
            st.session_state['bt_result_trade'] = t_df
            st.session_state['bt_result_equity'] = e_df
            
            st.success("백테스트 완료! 결과를 확인하세요.")

    # --------------------------------------------------------------------------------
    # 3. 결과 대시보드 (저장된 데이터가 있으면 출력)
    # --------------------------------------------------------------------------------
    
    trade_df = st.session_state['bt_result_trade']
    equity_df = st.session_state['bt_result_equity']

    if not trade_df.empty and not equity_df.empty:
        # --- 추가 지표 계산 ---
        equity_df = equity_df.copy()
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
            
            k1.metric(
                "최종 자산", 
                f"{final_equity/10000:,.0f}만원", 
                delta=f"{profit_amt/10000:,.0f}만원", 
                delta_color="normal"
            )
            
            k2.metric(
                "총 수익률", 
                f"{total_return:,.2f}%", 
                delta="복리 적용" if comp_mode else "단리 적용"
            )
            
            k3.metric(
                "승률 (Win Rate)", 
                f"{win_rate:.1f}%", 
                f"{win_count}승 {total_sells-win_count}패"
            )
            
            k4.metric(
                "MDD (최대낙폭)", 
                f"{mdd:.2f}%", 
                "Risk Level", 
                delta_color="off"
            )
            
            k5.metric(
                "총 매매 횟수", 
                f"{len(trade_df)//2}회", 
                f"{len(trade_df)//2 / max(len(equity_df),1) * 5:.1f}회/주"
            )

        # ---------------------------
        # [섹션 B] 자산 성장 그래프 (테마 적응형)
        # ---------------------------
        st.markdown("#### 📈 자산 성장 & MDD 추이")
        
        tab_g1, tab_g2 = st.tabs(["💰 자산 커브 (Equity)", "💧 낙폭 (Drawdown)"])
        
        # 공통 레이아웃 설정
        common_layout = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=None),
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            hovermode="x unified"
        )

        with tab_g1:
            fig = px.line(equity_df, x='date', y='equity', title=None, height=350)
            fig.add_hline(y=initial_cap_input, line_dash="dash", line_color="gray", annotation_text="원금")
            
            fig.update_traces(line=dict(color='#00CC96', width=2), fill='tozeroy') 
            fig.update_layout(xaxis_title="", yaxis_title="평가 금액 (원)", **common_layout)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab_g2:
            fig_dd = px.area(equity_df, x='date', y='drawdown', title=None, height=350)
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
                    st.metric("수익률", f"{best_trade['profit']:.2f}%", best_trade['reason'])
                    
                with st.container(border=True):
                    st.caption("💧 최악의 매매")
                    st.markdown(f"**{worst_trade['name']}**")
                    st.metric("수익률", f"{worst_trade['profit']:.2f}%", worst_trade['reason'], delta_color="inverse")
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
                with st.spinner("차트 로딩..."):
                    chart_data = yf.download(selected_ticker, start=str(bt_start_date), progress=False, auto_adjust=True)
                    if isinstance(chart_data.columns, pd.MultiIndex):
                        chart_data.columns = chart_data.columns.get_level_values(0)
                    chart_data = chart_data.loc[:, ~chart_data.columns.duplicated()]

                if not chart_data.empty:
                    fig_d = go.Figure()
                    
                    # 주가 라인
                    fig_d.add_trace(go.Scatter(
                        x=chart_data.index, y=chart_data['Close'], 
                        mode='lines', name='주가', 
                        line=dict(color='#888888', width=1.5)
                    ))
                    
                    # 매수: 빨강
                    buys = trade_df[(trade_df['ticker'] == selected_ticker) & (trade_df['type'] == 'buy')]
                    if not buys.empty:
                        fig_d.add_trace(go.Scatter(
                            x=buys['date'], y=buys['price'], mode='markers', name='매수', 
                            marker=dict(symbol='triangle-up', color='#FF4B4B', size=11),
                            hovertemplate='매수: %{y:,.0f}<br>날짜: %{x}'
                        ))
                    # 매도: 파랑
                    sells_sub = trade_df[(trade_df['ticker'] == selected_ticker) & (trade_df['type'] == 'sell')]
                    if not sells_sub.empty:
                        fig_d.add_trace(go.Scatter(
                            x=sells_sub['date'], y=sells_sub['price'], mode='markers', name='매도', 
                            marker=dict(symbol='triangle-down', color='#1C83E1', size=11),
                            text=[f"{p:.1f}%" for p in sells_sub['profit']], 
                            hovertemplate='매도: %{y:,.0f}<br>수익: %{text}'
                        ))
                    
                    fig_d.update_layout(
                        title=dict(text=f"{selected_option} 매매 타점", font=dict(size=15)),
                        height=350, 
                        margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        **common_layout
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
            log_df['date'] = pd.to_datetime(log_df['date']).dt.date

            log_df = log_df[['date', 'name', 'type', 'price', 'shares', 'profit', 'score', 'reason']]
            log_df.columns = ['날짜', '종목명', '구분', '가격', '수량', '수익률', 'AI점수', '매매사유']

            st.dataframe(
                log_df.sort_values('날짜', ascending=False),
                hide_index=True,
                use_container_width=True,
                height=500,
                column_config={
                    "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
                    "가격": st.column_config.NumberColumn("체결가", format="%.0f"),
                    "수량": st.column_config.NumberColumn("수량(주)", format="%d"),
                    "AI점수": st.column_config.ProgressColumn("AI Score", format="%.0f점", min_value=0, max_value=100),
                    "수익률": st.column_config.NumberColumn("수익률(%)", format="%.2f%%"),
                    "구분": st.column_config.TextColumn("Type", width="small")
                }
            )
    else:
        st.warning("⚠️ 매매 신호가 발생하지 않았습니다. 전략 조건을 완화하거나 기간을 늘려보세요.")
