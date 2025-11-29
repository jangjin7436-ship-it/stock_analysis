import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import time

# =========================================================
# 1. 데이터 로딩 함수 (auto_adjust=False 유지)
# =========================================================

@st.cache_data(show_spinner=False)
def load_price_data(code: str, start_date: str):
    """
    yfinance에서 개별 종목 데이터를 받아오는 함수 (캐시됨)
    [유지] auto_adjust=False로 실제 체결가 사용
    """
    try:
        df = yf.download(code, start=start_date, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_fx_series(start_date: str):
    """
    KRW=X 환율 시계열 다운로드
    """
    try:
        ex_df = yf.download("KRW=X", start=start_date, progress=False, auto_adjust=False)
        if isinstance(ex_df.columns, pd.MultiIndex):
            ex_df.columns = ex_df.columns.get_level_values(0)
        return ex_df['Close']
    except Exception:
        return pd.Series()

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
# 2. 지표 계산 로직 (ATR 추가 및 로직 개선)
# =========================================================

def calculate_indicators_for_backtest(df):
    """지표 계산 최적화: ATR 및 추세 강도 지표 추가"""
    df = df.copy()
    
    # [유지] 실제 종가 사용
    df['Close_Calc'] = df['Close']
    
    # 1. 이동평균
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA10'] = df['Close_Calc'].rolling(10).mean()
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()
    df['MA120'] = df['Close_Calc'].rolling(120).mean()

    # [추가] 이격도 (Disparity): 1.1 이상이면 과열
    df['Disparity_20'] = df['Close_Calc'] / df['MA20']
    
    # [추가] 추세 기울기 (Slope): MA가 상승 중인지 확인
    df['MA20_Slope'] = df['MA20'].diff()
    df['MA60_Slope'] = df['MA60'].diff()
    # [개선] MA120 기울기 계산 추가 (장기 추세 확인용)
    df['MA120_Slope'] = df['MA120'].diff()
    
    # 2. 볼린저 밴드
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
    
    # [중요 추가] ATR (Average True Range) - 변동성 지표
    # 고점 매도/저점 손절 방지를 위한 핵심
    prev_close = df['Close_Calc'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - prev_close)
    tr3 = abs(df['Low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # 5. 거래량
    if 'Volume' in df.columns:
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    else:
        df['Vol_Ratio'] = 1.0

    # 6. 변동성 (표준편차)
    df['STD20'] = std
    
    return df.dropna()

def get_ai_score_row(row):
    """
    [개선된 AI 점수 로직]
    기존: 돌파 매매 중심 (고점 매수 위험)
    변경: 추세 내 눌림목(Dip Buying) 및 과열 방지 중심
    """
    try:
        score = 50.0
        curr = row['Close_Calc']
        ma5, ma10, ma20, ma60 = row['MA5'], row['MA10'], row['MA20'], row['MA60']
        rsi = row['RSI']
        atr = row['ATR']
        
        # 1. 추세 판단 (장기 이평선 기울기가 중요)
        # MA60이 우상향이면 기본 점수 부여 (상승장)
        if row['MA60_Slope'] > 0:
            score += 10.0
            if curr > ma60: score += 5.0
        else:
            score -= 5.0

        # 2. 진입 타이밍 (눌림목 우대)
        # 상승 추세(MA20 우상향)인데 가격이 MA5 근처거나 살짝 아래일 때 점수 UP
        if row['MA20_Slope'] > 0:
            if curr > ma20:
                score += 5.0
                # 골든크로스 초입이거나 눌림목일 때 가산점
                if curr < ma5 * 1.01: 
                    score += 5.0  # 눌림목 보너스
        
        # 3. 과열 방지 (이격도 필터)
        # MA20 대비 10% 이상 급등한 상태면 진입 자제 (점수 대폭 삭감)
        disparity = row['Disparity_20']
        if disparity > 1.10: 
            score -= 20.0  # 고점 추격 매수 방지
        elif disparity > 1.05:
            score -= 5.0

        # 4. 보조지표 혼합
        # MACD가 상승 반전할 때
        if row['MACD_Hist'] > row['Prev_MACD_Hist']:
            score += 5.0
        
        # RSI: 40~60 사이의 안정적 구간 선호, 70 이상은 과열로 판단하여 감점
        if 40 <= rsi <= 60: 
            score += 5.0
        elif rsi > 70: 
            score -= 10.0  # 과열 경고
        # RSI 과매도 구간 (<30)은 가산점 주지 않음 (안정성 향상)
        # if rsi < 30: score += 5.0  # 기존: 과매도 반등 노리기 -> 제거

        # 볼린저 밴드 하단 터치 후 반등 시그널
        if curr <= row['Lower_Band'] * 1.02:
            score += 10.0  # 저점 매수 기회

        # 거래량 실린 양봉
        if row['Vol_Ratio'] >= 1.5 and curr > row['Open']:
            score += 5.0

        # [개선] 장기 추세 (MA120) 반영: 장기 추세 상이면 가산, 하이면 감산
        if 'MA120' in row:
            if curr >= row['MA120']:
                score += 5.0
            else:
                score -= 5.0

        return max(0.0, min(100.0, score))
    except:
        return 0.0

# =========================================================
# 3. 백테스트 엔진 (ATR 기반 청산 로직 적용)
# =========================================================

def prepare_stock_data(ticker_info, start_date):
    """개별 종목 데이터 준비"""
    code, name = ticker_info
    try:
        df_raw = load_price_data(code, start_date)
        if df_raw is None or df_raw.empty or len(df_raw) < 120: # MA120 계산을 위해 데이터 확보 필요
            return None

        df = calculate_indicators_for_backtest(df_raw)
        df['AI_Score'] = df.apply(get_ai_score_row, axis=1)
        df['Ticker'] = code
        df['Name'] = name
        
        # 필요한 열만 선택 (추가된 지표 포함)
        return df[['Open', 'High', 'Low', 'Close_Calc', 'AI_Score', 'ATR', 'MA20', 'Vol_Ratio', 'Ticker', 'Name', 'MA60_Slope', 'MA120']]
    except Exception as e:
        return None

def run_portfolio_backtest(targets, start_date, initial_capital, strategy_mode,
                           max_hold_days, exchange_data, use_compound, selection_mode):
    # ---------------------------------------------------------
    # 1. 전 종목 데이터 준비
    # ---------------------------------------------------------
    all_dfs = []
    for t in targets:
        res = prepare_stock_data(t, start_date)
        if res is not None:
            all_dfs.append(res)
            
    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), initial_capital

    # ---------------------------------------------------------
    # 2. Market Data 통합
    # ---------------------------------------------------------
    market_data = {}
    for df in all_dfs:
        for date, row in df.iterrows():
            if date not in market_data:
                market_data[date] = []
            market_data[date].append(row)
    
    sorted_dates = sorted(market_data.keys())

    # ---------------------------------------------------------
    # 3. 환율 데이터 준비
    # ---------------------------------------------------------
    if isinstance(exchange_data, (float, int)):
        get_rate = lambda d: float(exchange_data)
    else:
        rate_dict = exchange_data.to_dict()
        def get_rate(d):
            ts = pd.Timestamp(d)
            return rate_dict.get(ts, 1430.0)

    # ---------------------------------------------------------
    # 4. 전략별 파라미터 설정 (ATR, 진입 점수 등)
    # ---------------------------------------------------------
    if strategy_mode == 'SuperLocking':
        atr_stop_mult = 1.5
        atr_profit_mult = 2.5
        trailing_mult = 2.0
        score_threshold = 75
        vol_threshold = 0.06
    elif strategy_mode == 'Basic':
        atr_stop_mult = 2.5
        atr_profit_mult = 4.0
        trailing_mult = 3.0
        score_threshold = 65
        vol_threshold = 0.12
    else:  # 기본: Sniper
        atr_stop_mult = 2.0
        atr_profit_mult = 3.0
        trailing_mult = 2.5
        score_threshold = 70
        vol_threshold = 0.08

    balance = initial_capital
    portfolio = {}
    trades_log = []
    equity_curve = []
    
    max_slots = 1 if selection_mode == 'TOP1' else 5 

    # ---------------------------------------------------------
    # 5. 날짜별 루프 (백테스트 메인)
    # ---------------------------------------------------------
    for date in sorted_dates:
        daily_stocks = market_data[date]
        current_rate = get_rate(date)

        # =================================================
        # A. 매도 로직 (Sell Check) - ATR 기반 유동적 대응
        # =================================================
        sell_list = []
        for ticker in sorted(portfolio.keys()):
            info = portfolio[ticker]
            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            
            if stock_row is None: 
                continue
            
            # [데이터 추출]
            rate = 1.0 if ".KS" in ticker else current_rate
            
            raw_open = stock_row['Open']
            raw_high = stock_row['High']
            raw_low = stock_row['Low']
            raw_close = stock_row['Close_Calc']
            atr = stock_row['ATR'] # 변동성 지표 사용
            
            curr_open = raw_open * rate
            curr_close = raw_close * rate
            
            score = stock_row['AI_Score']
            fee_sell = 0.003 if ".KS" in ticker else 0.001
            
            avg_price = info['avg_price']
            buy_price_raw = info.get('buy_price_raw', avg_price / rate)  # 매수 당시 원화가 아닌 달러가 기준

            held_days = (pd.Timestamp(date) - pd.Timestamp(info['buy_date'])).days
            
            should_sell = False
            sell_reason = ""
            final_sell_price = curr_close 
            final_sell_price_raw = raw_close

            # --- [동적 손절/익절 로직] ---
            # 고정 %가 아닌 ATR(변동성)을 사용하여 "숨 쉴 공간"을 부여함
            # ATR이 크면(변동성 큼) 손절폭을 넓게 잡음 -> 휩소(속임수) 방지
            
            # 전략별 ATR 배수 적용
            stop_price_raw = buy_price_raw - (atr * atr_stop_mult)
            target_price_raw = buy_price_raw + (atr * atr_profit_mult)

            # 최고가 갱신 시 손절 라인도 같이 올림 (수익 보전)
            current_max_raw = info.get('max_price_raw', buy_price_raw)
            if raw_high > current_max_raw:
                portfolio[ticker]['max_price_raw'] = raw_high
                # 고점 대비 trailing_mult ATR 하락 시 익절/청산 (기존 -3% 고정보다 유연함)
                new_stop = raw_high - (atr * trailing_mult)
                if new_stop > stop_price_raw:
                    stop_price_raw = new_stop

            if not should_sell:
                # 갭락 손절
                if raw_open < stop_price_raw:
                    should_sell = True
                    sell_reason = "⚡ 갭락(ATR이탈)"
                    final_sell_price = curr_open
                    final_sell_price_raw = raw_open
                # 목표가 달성 시 익절
                elif raw_high >= target_price_raw:
                    should_sell = True
                    sell_reason = "🎯 목표달성익절"
                    final_sell_price_raw = target_price_raw
                    final_sell_price = final_sell_price_raw * rate
                # 장중 손절
                elif raw_low < stop_price_raw:
                    should_sell = True
                    sell_reason = "📉 ATR손절/청산"
                    # 슬리피지 고려: 손절가보다 살짝 아래에서 체결 가정
                    final_sell_price_raw = stop_price_raw * 0.995 
                    final_sell_price = final_sell_price_raw * rate

            # [시나리오 2] 만기 및 스코어 청산
            if not should_sell:
                limit_days = max_hold_days if max_hold_days > 0 else 20 
                
                # 수익권인데 점수가 나빠지면 차익 실현
                if raw_close > buy_price_raw * 1.05 and score < 45:
                    should_sell = True
                    sell_reason = "💰 점수하락익절"
                
                # 너무 오래 들고 있는데 수익이 안 나면 교체
                elif held_days >= limit_days:
                    should_sell = True
                    sell_reason = f"⏱️ 만기청산({held_days}일)"
                
                # 급락 징후 (점수 폭락)
                elif score < 30:
                    should_sell = True
                    sell_reason = "점수급락(30↓)"

            if should_sell:
                real_profit_pct = ((final_sell_price - avg_price) / avg_price) * 100
                return_amt = info['shares'] * final_sell_price * (1 - fee_sell)
                balance += return_amt
                
                trades_log.append({
                    'ticker': ticker,
                    'name': info['name'],
                    'date': date,
                    'type': 'sell',
                    'price': final_sell_price_raw,
                    'shares': info['shares'],
                    'score': score,
                    'profit': real_profit_pct,
                    'reason': sell_reason,
                    'balance': balance
                })
                sell_list.append(ticker)
        
        for t in sell_list: 
            del portfolio[t]

        # =================================================
        # B. 신규 매수 (Buy Logic) - 높은 점수 + 눌림목
        # =================================================
        if len(portfolio) < max_slots:
            candidates = []
            for row in daily_stocks:
                ticker = row['Ticker']
                if ticker in portfolio: 
                    continue
                
                score = row['AI_Score']
                price_raw = row['Close_Calc']
                
                # 필터링: 전략별 최소 점수 만족해야 함
                if score >= score_threshold:
                    # [개선] 변동성 필터: ATR 대비 변동성 높은 종목 제외
                    vol_pct = row['ATR'] / row['Close_Calc'] if row['Close_Calc'] != 0 else 0
                    if vol_pct > vol_threshold:
                        continue
                    # [개선] 추세 필터: 장기 추세 (MA120) 상승 + 중기 추세 (MA60_Slope > 0) 종목만 매수
                    if 'MA120' in row and (row['Close_Calc'] < row['MA120'] or row['MA60_Slope'] <= 0):
                        continue

                    rate = 1.0 if ".KS" in ticker else current_rate
                    price_krw = price_raw * rate
                    
                    vol_power = row.get('Vol_Ratio', 1.0)
                    
                    candidates.append({
                        'ticker': ticker,
                        'name': row['Name'],
                        'price_raw': price_raw,
                        'price_krw': price_krw,
                        'score': score,
                        'vol_power': vol_power,
                        'reason': "AI추천(눌림목/추세)"
                    })

            # 점수 높은 순 -> 거래량 강도 순 정렬
            candidates.sort(key=lambda x: (x['score'], x['vol_power']), reverse=True)
            open_slots = max_slots - len(portfolio)
            buy_targets = candidates[:open_slots]
            
            for target in buy_targets:
                if balance <= 0: 
                    break
                
                current_open_slots = max_slots - len(portfolio)
                slot_budget = balance / current_open_slots
                fee_buy = 0.00015 if ".KS" in target['ticker'] else 0.001
                
                if target['price_krw'] > 0:
                    shares = int(slot_budget / (target['price_krw'] * (1 + fee_buy)))
                    if shares > 0:
                        cost = shares * target['price_krw'] * (1 + fee_buy)
                        balance -= cost
                        portfolio[target['ticker']] = {
                            'name': target['name'],
                            'shares': shares,
                            'avg_price': target['price_krw'],
                            'buy_price_raw': target['price_raw'], # ATR 계산용 원본가 저장
                            'buy_date': date,
                            'max_price_raw': target['price_raw'], # 트레일링 스탑용 고점
                        }
                        trades_log.append({
                            'ticker': target['ticker'],
                            'name': target['name'],
                            'date': date,
                            'type': 'buy',
                            'price': target['price_raw'],
                            'shares': shares,
                            'score': target['score'],
                            'profit': 0,
                            'reason': target['reason'],
                            'balance': balance
                        })

        # =================================================
        # C. 자산 평가
        # =================================================
        current_equity = balance
        for ticker, info in portfolio.items():
            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            if stock_row is not None:
                rate = 1.0 if ".KS" in ticker else current_rate
                p_krw = stock_row['Close_Calc'] * rate
                current_equity += info['shares'] * p_krw
            else:
                current_equity += info['shares'] * info['avg_price']
        
        equity_curve.append({'date': date, 'equity': current_equity})

    # =================================================
    # D. 최종일 기준 보유 종목 리포트 생성 (Mark-to-Market)
    # =================================================
    held_stocks_list = []
    
    if sorted_dates:
        last_date = sorted_dates[-1]
        last_daily_stocks = market_data[last_date]
        last_rate = get_rate(last_date)
        
        for ticker, info in portfolio.items():
            stock_row = next((x for x in last_daily_stocks if x['Ticker'] == ticker), None)
            
            if stock_row is not None: 
                rate = 1.0 if ".KS" in ticker else last_rate
                curr_price = stock_row['Close_Calc'] * rate
                curr_price_raw = stock_row['Close_Calc']
            else:
                curr_price = info['avg_price'] 
                curr_price_raw = 0
            
            fee_sell = 0.003 if ".KS" in ticker else 0.001
            market_value = info['shares'] * curr_price
            net_value = market_value * (1 - fee_sell)
            
            return_pct = ((curr_price - info['avg_price']) / info['avg_price']) * 100
            
            held_stocks_list.append({
                '티커': ticker,
                '종목명': info['name'],
                '보유주수': info['shares'],
                '매수단가(KRW)': info['avg_price'],
                '현재가(KRW)': curr_price,
                '현재가(Raw)': curr_price_raw,
                '평가손익(%)': return_pct,
                '평가금액': net_value
            })

    held_df = pd.DataFrame(held_stocks_list)

    return pd.DataFrame(trades_log), pd.DataFrame(equity_curve), held_df, balance
                            
# =========================================================
# 4. UI 통합 (탭 추가)
# =========================================================

tab4 = st.tabs(["📊 전체 백테스트 시뮬레이션"])[0] 

with tab4:
    st.markdown("### 🧪 포트폴리오 유니버스 백테스트")
    st.caption("AI 전략 시뮬레이터 Final Ver. (ATR 기반 동적 손절/익절 + 이격도 과열 방지)")
    
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        bt_start_date = st.date_input("시작일", value=pd.to_datetime("2024-01-01"))
        max_hold_days = st.slider("⏱️ 타임 컷 (일)", 0, 60, 14, help="매수 후 N일 지나면 강제 매도 (0이면 해제)")
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
    
    c_strat, c_opt, c_btn = st.columns([2, 1, 1])
    with c_strat:
        selected_strategy = st.radio(
            "⚔️ 매매 전략 선택", 
            ["AI 스나이퍼 (추천)", "슈퍼 락킹 (안전)", "기본 모드 (장투)"],
            captions=[
                "ATR 변동성 기반 대응 / 눌림목 매수", 
                "타이트한 ATR 익절", 
                "여유로운 스윙"
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

    # 실행 및 결과 저장
    if 'bt_result_trade' not in st.session_state:
        st.session_state['bt_result_trade'] = pd.DataFrame()
    if 'bt_result_equity' not in st.session_state:
        st.session_state['bt_result_equity'] = pd.DataFrame()
    if 'bt_held_df' not in st.session_state:
        st.session_state['bt_held_df'] = pd.DataFrame()
    if 'bt_final_cash' not in st.session_state:
        st.session_state['bt_final_cash'] = 0.0

    if start_btn:
        if exchange_arg_val == "DYNAMIC":
            with st.spinner("💱 환율 데이터 수집 중..."):
                exchange_data_payload = load_fx_series(str(bt_start_date))
        else:
            exchange_data_payload = float(exchange_arg_val)

        with st.spinner(f"🔄 [{selected_strategy}] 전략으로 전체 시장 스캔 중..."):
            targets = list(TICKER_MAP.items())
            
            t_df, e_df, h_df, f_cash = run_portfolio_backtest(
                targets, str(bt_start_date), initial_cap_input, strat_code, 
                max_hold_days, exchange_data_payload, comp_mode, selection_code
            )
            
            st.session_state['bt_result_trade'] = t_df
            st.session_state['bt_result_equity'] = e_df
            st.session_state['bt_held_df'] = h_df
            st.session_state['bt_final_cash'] = f_cash
            
            st.success("백테스트 완료! 결과를 확인하세요.")

    # 대시보드 출력
    trade_df = st.session_state['bt_result_trade']
    equity_df = st.session_state['bt_result_equity']
    held_df = st.session_state['bt_held_df']
    final_cash = st.session_state['bt_final_cash']

    if not trade_df.empty and not equity_df.empty:
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

            # [섹션 A] 핵심 성과 지표
            st.markdown("#### 🚀 백테스트 요약 리포트")
            
            with st.container(border=True):
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("최종 자산", f"{final_equity/10000:,.0f}만원", 
                          delta=f"{profit_amt/10000:,.0f}만원", delta_color="normal")
                k2.metric("총 수익률", f"{total_return:,.2f}%", 
                          delta="복리 적용" if comp_mode else "단리 적용")
                k3.metric("실현 승률", f"{win_rate:.1f}%", 
                          f"{win_count}승 {total_sells-win_count}패")
                k4.metric("MDD (최대낙폭)", f"{mdd:.2f}%", 
                          "Risk Level", delta_color="off")
                k5.metric("총 매매 횟수", f"{len(trade_df)//2}회", 
                          f"평균 {len(trade_df)//2 / len(equity_df) * 5:.1f}회/주")

            # 기말 자산 상세 현황
            st.subheader("💰 기말 보유 자산 현황")
            st.caption("백테스트 종료일 기준, 현금과 보유 중인 주식의 평가 가치입니다.")
            
            c_assets, c_table = st.columns([1, 2])
            
            held_value_sum = held_df['평가금액'].sum() if not held_df.empty else 0
            
            with c_assets:
                with st.container(border=True):
                    st.metric("💵 현금 잔고", f"{final_cash/10000:,.0f}만원")
                    st.metric("📦 보유 주식 평가액", f"{held_value_sum/10000:,.0f}만원")
                    st.markdown("---")
                    st.metric("합계 (최종 자산)", f"{(final_cash + held_value_sum)/10000:,.0f}만원")

            with c_table:
                if not held_df.empty:
                    st.dataframe(
                        held_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "매수단가(KRW)": st.column_config.NumberColumn(format="%d원"),
                            "현재가(KRW)": st.column_config.NumberColumn(format="%d원"),
                            "현재가(Raw)": st.column_config.NumberColumn(format="%.2f"),
                            "평가손익(%)": st.column_config.NumberColumn(format="%.2f%%"),
                            "평가금액": st.column_config.NumberColumn(format="%d원")
                        }
                    )
                else:
                    st.info("보유 중인 주식이 없습니다. (100% 현금 보유)")

            # [섹션 B] 자산 성장 그래프
            st.markdown("#### 📈 자산 성장 & MDD 추이")
            tab_g1, tab_g2 = st.tabs(["💰 자산 커브 (Equity)", "💧 낙폭 (Drawdown)"])
            
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

            # [섹션 C] 매매 상세 분석
            c_left, c_right = st.columns([1, 1.5])
            
            with c_left:
                st.markdown("#### 🏆 Best & Worst (실현 손익 기준)")
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
                    
                    my_trades = trade_df[trade_df['ticker'] == selected_ticker].sort_values('date')
                    with st.spinner("차트 로딩..."):
                        # [차트 로딩 auto_adjust=False]
                        chart_data = yf.download(selected_ticker, start=str(bt_start_date), progress=False, auto_adjust=False)
                        if isinstance(chart_data.columns, pd.MultiIndex):
                            chart_data.columns = chart_data.columns.get_level_values(0)
                        chart_data = chart_data.loc[:, ~chart_data.columns.duplicated()]

                    if not chart_data.empty:
                        fig_d = go.Figure()
                        fig_d.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'], 
                                                   mode='lines', name='주가', 
                                                   line=dict(color='#888888', width=1.5)))
                        
                        buys = my_trades[my_trades['type'] == 'buy']
                        if not buys.empty:
                            fig_d.add_trace(go.Scatter(x=buys['date'], y=buys['price'], mode='markers', name='매수', 
                                                       marker=dict(symbol='triangle-up', color='#FF4B4B', size=11),
                                                       hovertemplate='매수: %{y:,.0f}<br>날짜: %{x}'))
                        
                        sells_sub = my_trades[my_trades['type'] == 'sell']
                        if not sells_sub.empty:
                            fig_d.add_trace(go.Scatter(x=sells_sub['date'], y=sells_sub['price'], mode='markers', name='매도', 
                                                       marker=dict(symbol='triangle-down', color='#1C83E1', size=11),
                                                       text=[f"{p:.1f}%" for p in sells_sub['profit']], 
                                                       hovertemplate='매도: %{y:,.0f}<br>수익: %{text}'))
                        
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

            # [섹션 D] 전체 거래 일지
            st.subheader("📝 전체 거래 로그")
            
            with st.expander("전체 거래 내역 (펼치기/접기)", expanded=True):
                log_df = trade_df.copy()
                log_df['date'] = log_df['date'].dt.date
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
