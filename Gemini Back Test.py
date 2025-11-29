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
    try:
        df = yf.download(code, start=start_date, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_fx_series(start_date: str):
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
# 2. 지표 계산 로직 (유지)
# =========================================================

def calculate_indicators_for_backtest(df):
    """지표 계산"""
    df = df.copy()
    df['Close_Calc'] = df['Close']
    
    # 이동평균
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA10'] = df['Close_Calc'].rolling(10).mean()
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()
    df['MA120'] = df['Close_Calc'].rolling(120).mean()

    # 이격도 & 기울기
    df['Disparity_20'] = df['Close_Calc'] / df['MA20']
    df['MA20_Slope'] = df['MA20'].diff()
    df['MA60_Slope'] = df['MA60'].diff()
    
    # 볼린저 밴드
    std = df['Close_Calc'].rolling(20).std()
    df['Upper_Band'] = df['MA20'] + (std * 2)
    df['Lower_Band'] = df['MA20'] - (std * 2)
    df['Band_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
    
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
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)
    
    # ATR (변동성) - 핵심
    prev_close = df['Close_Calc'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - prev_close)
    tr3 = abs(df['Low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # 거래량
    if 'Volume' in df.columns:
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    else:
        df['Vol_Ratio'] = 1.0

    df['STD20'] = std
    return df.dropna()

def get_ai_score_row(row):
    """AI 점수 산정 (기존 로직 유지)"""
    try:
        score = 50.0
        curr = row['Close_Calc']
        ma5, ma10, ma20, ma60 = row['MA5'], row['MA10'], row['MA20'], row['MA60']
        rsi = row['RSI']
        
        # 1. 추세
        if row['MA60_Slope'] > 0:
            score += 10.0
            if curr > ma60: score += 5.0
        else:
            score -= 5.0

        # 2. 진입 (눌림목)
        if row['MA20_Slope'] > 0:
            if curr > ma20:
                score += 5.0
                if curr < ma5 * 1.01: score += 5.0
            
        # 3. 과열 방지
        disparity = row['Disparity_20']
        if disparity > 1.10: score -= 20.0
        elif disparity > 1.05: score -= 5.0

        # 4. 보조지표
        if row['MACD_Hist'] > row['Prev_MACD_Hist']: score += 5.0
        if 40 <= rsi <= 60: score += 5.0
        elif rsi > 70: score -= 10.0
        elif rsi < 30: score += 5.0

        if curr <= row['Lower_Band'] * 1.02: score += 10.0
        if row['Vol_Ratio'] >= 1.5 and curr > row['Open']: score += 5.0

        return max(0.0, min(100.0, score))
    except:
        return 0.0

# =========================================================
# 3. 백테스트 엔진 (리스크 관리 + 시장 필터 추가)
# =========================================================

def prepare_stock_data(ticker_info, start_date):
    code, name = ticker_info
    try:
        df_raw = load_price_data(code, start_date)
        if df_raw is None or df_raw.empty or len(df_raw) < 120:
            return None
        df = calculate_indicators_for_backtest(df_raw)
        df['AI_Score'] = df.apply(get_ai_score_row, axis=1)
        df['Ticker'] = code
        df['Name'] = name
        return df[['Open', 'High', 'Low', 'Close_Calc', 'AI_Score', 'ATR', 'MA20', 'MA60', 'Vol_Ratio', 'Ticker', 'Name']]
    except Exception:
        return None

def run_portfolio_backtest(targets, start_date, initial_capital, strategy_mode,
                           max_hold_days, exchange_data, use_compound, selection_mode):
    
    # 1. 전 종목 데이터 준비
    all_dfs = []
    for t in targets:
        res = prepare_stock_data(t, start_date)
        if res is not None:
            all_dfs.append(res)
            
    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), initial_capital

    # 2. Market Data 통합
    market_data = {}
    for df in all_dfs:
        for date, row in df.iterrows():
            if date not in market_data: market_data[date] = []
            market_data[date].append(row)
    
    sorted_dates = sorted(market_data.keys())

    # 3. 환율 설정
    if isinstance(exchange_data, (float, int)):
        get_rate = lambda d: float(exchange_data)
    else:
        rate_dict = exchange_data.to_dict()
        def get_rate(d):
            ts = pd.Timestamp(d)
            return rate_dict.get(ts, 1430.0)

    # 4. 초기화
    balance = initial_capital
    portfolio = {}
    trades_log = []
    equity_curve = []
    
    # [설정] 슬롯 및 리스크 관리 파라미터
    base_max_slots = 1 if selection_mode == 'TOP1' else 5 
    
    # ---------------------------------------------------------
    # 5. 날짜별 루프
    # ---------------------------------------------------------
    for date in sorted_dates:
        daily_stocks = market_data[date]
        current_rate = get_rate(date)

        # -----------------------------------------------------
        # [핵심 추가 1] 시장 국면 판단 (Market Breadth)
        # -----------------------------------------------------
        # 현재 추적 중인 모든 종목 중 "MA60 위에 있는 종목 비율" 계산
        # 이 비율이 낮으면 하락장으로 판단하고 방어 모드 발동
        count_above_ma60 = sum(1 for x in daily_stocks if x['Close_Calc'] > x['MA60'])
        total_active = len(daily_stocks)
        market_breadth = count_above_ma60 / total_active if total_active > 0 else 0.5
        
        is_bear_market = market_breadth < 0.35  # 전체 종목의 35% 미만만 상승세면 '하락장'
        
        # 하락장일 때 패널티 부여
        current_max_slots = max(1, base_max_slots - 2) if is_bear_market and base_max_slots > 1 else base_max_slots
        min_buy_score = 75 if is_bear_market else 70  # 하락장에선 기준 점수 상향
        
        # -----------------------------------------------------
        # A. 매도 로직 (Sell Check)
        # -----------------------------------------------------
        sell_list = []
        for ticker in sorted(portfolio.keys()):
            info = portfolio[ticker]
            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            
            if stock_row is None: continue
            
            rate = 1.0 if ".KS" in ticker else current_rate
            raw_open, raw_high, raw_low, raw_close = stock_row['Open'], stock_row['High'], stock_row['Low'], stock_row['Close_Calc']
            atr = stock_row['ATR']
            
            curr_open = raw_open * rate
            curr_close = raw_close * rate
            score = stock_row['AI_Score']
            fee_sell = 0.003 if ".KS" in ticker else 0.001
            
            avg_price = info['avg_price']
            buy_price_raw = info.get('buy_price_raw', avg_price/rate)
            held_days = (pd.Timestamp(date) - pd.Timestamp(info['buy_date'])).days
            
            should_sell = False
            sell_reason = ""
            final_sell_price = curr_close 
            final_sell_price_raw = raw_close

            # ATR 기반 동적 청산
            # 하락장이면 손절 라인을 더 타이트하게(1.5배) 잡음
            stop_mult = 1.5 if is_bear_market else 2.0
            profit_mult = 3.0
            
            stop_price_raw = buy_price_raw - (atr * stop_mult)
            
            # 트레일링 스탑
            current_max_raw = info.get('max_price_raw', buy_price_raw)
            if raw_high > current_max_raw:
                portfolio[ticker]['max_price_raw'] = raw_high
                # 고점 대비 하락폭도 하락장에선 더 민감하게
                trail_gap = atr * (2.0 if is_bear_market else 2.5)
                new_stop = raw_high - trail_gap
                if new_stop > stop_price_raw:
                    stop_price_raw = new_stop

            if not should_sell:
                if raw_open < stop_price_raw:
                    should_sell = True
                    sell_reason = "⚡ 갭락(ATR)"
                    final_sell_price = curr_open
                    final_sell_price_raw = raw_open
                elif raw_low < stop_price_raw:
                    should_sell = True
                    sell_reason = "📉 ATR손절"
                    final_sell_price_raw = stop_price_raw * 0.995 
                    final_sell_price = final_sell_price_raw * rate

            if not should_sell:
                limit_days = max_hold_days if max_hold_days > 0 else 20 
                if raw_close > buy_price_raw * 1.05 and score < 45:
                    should_sell = True; sell_reason = "💰 점수하락익절"
                elif held_days >= limit_days:
                    should_sell = True; sell_reason = f"⏱️ 만기청산({held_days}일)"
                elif score < 30:
                    should_sell = True; sell_reason = "점수급락(30↓)"
                # [추가] 하락장이고 수익이 미미하면 현금 확보를 위해 조기 매도
                elif is_bear_market and held_days > 5 and raw_close < buy_price_raw:
                    should_sell = True; sell_reason = "시장악화방어"

            if should_sell:
                return_amt = info['shares'] * final_sell_price * (1 - fee_sell)
                balance += return_amt
                real_profit = ((final_sell_price - avg_price) / avg_price) * 100
                trades_log.append({
                    'ticker': ticker, 'name': info['name'], 'date': date, 'type': 'sell',
                    'price': final_sell_price_raw, 'shares': info['shares'], 'score': score,
                    'profit': real_profit, 'reason': sell_reason, 'balance': balance
                })
                sell_list.append(ticker)
        
        for t in sell_list: del portfolio[t]

        # -----------------------------------------------------
        # B. 신규 매수 (Buy Logic) - 변동성 역가중 방식 (Volatility Sizing)
        # -----------------------------------------------------
        # 하락장이 심하면 아예 신규 매수 금지 (현금 관망)
        if len(portfolio) < current_max_slots and not (is_bear_market and selection_mode == 'TOP1'):
            candidates = []
            for row in daily_stocks:
                ticker = row['Ticker']
                if ticker in portfolio: continue
                
                score = row['AI_Score']
                if score >= min_buy_score:
                    candidates.append({
                        'ticker': ticker, 'name': row['Name'],
                        'price_raw': row['Close_Calc'], 'score': score,
                        'atr': row['ATR'], 'vol_power': row.get('Vol_Ratio', 1.0)
                    })

            candidates.sort(key=lambda x: (x['score'], x['vol_power']), reverse=True)
            open_slots = current_max_slots - len(portfolio)
            buy_targets = candidates[:open_slots]
            
            for target in buy_targets:
                if balance <= 0: break
                
                # [핵심 추가 2] 변동성 조절 (Volatility Sizing)
                # 단순히 N빵(1/N) 하지 않고, ATR이 크면 적게, 작으면 많이 삼.
                # 목표: 종목당 리스크를 전체 자산의 2%로 고정
                
                rate = 1.0 if ".KS" in target['ticker'] else current_rate
                price_krw = target['price_raw'] * rate
                atr_krw = target['atr'] * rate
                
                # 리스크 허용액 (총 자산의 2% ~ 5% 유동적)
                risk_per_trade = (balance + sum(p['shares']*p['avg_price'] for p in portfolio.values())) * 0.02
                
                # ATR 2배를 손절폭으로 가정했을 때의 적정 주식 수
                # Volatility Sizing 공식: 주식수 = 리스크허용액 / (2 * ATR)
                vol_adjusted_shares = int(risk_per_trade / (atr_krw * 2)) if atr_krw > 0 else 0
                
                # 단, 최대 투자금은 (잔고 / 남은슬롯)을 넘지 않도록 캡(Cap) 씌움 (자금 고갈 방지)
                equal_weight_budget = balance / (current_max_slots - len(portfolio) + 1) # +1은 안전마진
                max_shares_by_budget = int(equal_weight_budget / price_krw)
                
                # 최종 매수 수량: 변동성 기준과 예산 기준 중 작은 것 선택 (보수적 접근)
                shares = min(vol_adjusted_shares, max_shares_by_budget)
                
                if shares > 0:
                    fee_buy = 0.00015 if ".KS" in target['ticker'] else 0.001
                    cost = shares * price_krw * (1 + fee_buy)
                    
                    if balance >= cost:
                        balance -= cost
                        portfolio[target['ticker']] = {
                            'name': target['name'], 'shares': shares,
                            'avg_price': price_krw, 'buy_price_raw': target['price_raw'],
                            'buy_date': date, 'max_price_raw': target['price_raw']
                        }
                        trades_log.append({
                            'ticker': target['ticker'], 'name': target['name'], 'date': date, 'type': 'buy',
                            'price': target['price_raw'], 'shares': shares, 'score': target['score'],
                            'profit': 0, 'reason': target['name'] + ("(방어)" if is_bear_market else ""), 
                            'balance': balance
                        })

        # C. 자산 평가
        current_equity = balance
        for ticker, info in portfolio.items():
            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            if stock_row:
                rate = 1.0 if ".KS" in ticker else current_rate
                current_equity += info['shares'] * stock_row['Close_Calc'] * rate
            else:
                current_equity += info['shares'] * info['avg_price']
        equity_curve.append({'date': date, 'equity': current_equity})

    # D. 리포트
    held_stocks_list = []
    if sorted_dates:
        last_date = sorted_dates[-1]
        last_daily = market_data[last_date]
        last_rate = get_rate(last_date)
        for ticker, info in portfolio.items():
            stock_row = next((x for x in last_daily if x['Ticker'] == ticker), None)
            curr_price = stock_row['Close_Calc'] * (1.0 if ".KS" in ticker else last_rate) if stock_row else info['avg_price']
            market_val = info['shares'] * curr_price
            held_stocks_list.append({
                '티커': ticker, '종목명': info['name'], '보유주수': info['shares'],
                '매수단가(KRW)': info['avg_price'], '현재가(KRW)': curr_price,
                '평가손익(%)': ((curr_price - info['avg_price']) / info['avg_price']) * 100,
                '평가금액': market_val
            })
            
    return pd.DataFrame(trades_log), pd.DataFrame(equity_curve), pd.DataFrame(held_stocks_list), balance

# =========================================================
# 4. UI 통합
# =========================================================

tab4 = st.tabs(["📊 전체 백테스트 시뮬레이션"])[0] 

with tab4:
    st.markdown("### 🧪 포트폴리오 유니버스 백테스트 (안정화 버전)")
    st.caption("✅ 기능 추가: 시장 국면 필터(Market Breadth) & 변동성 조절(Volatility Sizing) 적용으로 시작 시점에 따른 편차 최소화")
    
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        bt_start_date = st.date_input("시작일", value=pd.to_datetime("2024-01-01"))
        max_hold_days = st.slider("⏱️ 타임 컷 (일)", 0, 60, 20)
    with r1_c2:
        initial_cap_input = st.number_input("💰 초기 자본금", value=10000000, step=1000000, format="%d")
        sel_mode = st.selectbox("🎯 종목 선정", ["조건 만족 전부 매수 (분산)", "점수 1등만 매수 (집중)"])
        selection_code = "TOP1" if "집중" in sel_mode else "ALL"
    with r1_c3:
        ex_method = st.radio("💱 환율 방식", ["실시간 변동 (Dynamic)", "고정 환율 (Fixed)"])
        exchange_arg_val = st.number_input("환율", value=1430.0) if "고정" in ex_method else "DYNAMIC"

    st.divider()
    
    if st.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True):
        if exchange_arg_val == "DYNAMIC":
            with st.spinner("💱 환율 데이터 수집 중..."):
                exchange_data_payload = load_fx_series(str(bt_start_date))
        else:
            exchange_data_payload = float(exchange_arg_val)

        with st.spinner(f"🔄 시장 전체 스캔 및 안정성 시뮬레이션 중..."):
            targets = list(TICKER_MAP.items())
            t_df, e_df, h_df, f_cash = run_portfolio_backtest(
                targets, str(bt_start_date), initial_cap_input, "Sniper", 
                max_hold_days, exchange_data_payload, True, selection_code
            )
            
            st.session_state['bt_result_trade'] = t_df
            st.session_state['bt_result_equity'] = e_df
            st.session_state['bt_held_df'] = h_df
            st.session_state['bt_final_cash'] = f_cash
            st.success("완료!")

    # 결과 출력 (기존과 동일)
    trade_df = st.session_state.get('bt_result_trade', pd.DataFrame())
    equity_df = st.session_state.get('bt_result_equity', pd.DataFrame())
    held_df = st.session_state.get('bt_held_df', pd.DataFrame())
    final_cash = st.session_state.get('bt_final_cash', 0.0)

    if not trade_df.empty and not equity_df.empty:
        final_equity = equity_df.iloc[-1]['equity']
        profit = final_equity - initial_cap_input
        ret = profit / initial_cap_input * 100
        
        st.markdown("#### 🚀 결과 요약")
        k1, k2, k3 = st.columns(3)
        k1.metric("최종 자산", f"{final_equity:,.0f}원", f"{profit:,.0f}원")
        k2.metric("수익률", f"{ret:.2f}%")
        k3.metric("매매 횟수", f"{len(trade_df)}건")
        
        st.subheader("📈 자산 추이")
        fig = px.line(equity_df, x='date', y='equity')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📝 거래 로그")
        st.dataframe(trade_df.sort_values('date', ascending=False), use_container_width=True)
        
        st.subheader("📦 기말 보유")
        if not held_df.empty:
            st.dataframe(held_df, use_container_width=True)
        else:
            st.info("보유 종목 없음")
