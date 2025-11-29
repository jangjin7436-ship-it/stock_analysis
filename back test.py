import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

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
    
    # 수정 종가 사용 (yfinance 데이터 대응)
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['Close_Calc'] = df[col]

    # 기술적 지표 계산
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()
    
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
    
    # 전일 히스토그램 (상승 반전 확인용)
    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)
    
    # 변동성 및 모멘텀
    df['STD20'] = df['Close_Calc'].rolling(20).std()
    
    return df.dropna()

def get_ai_score_row(row):
    """
    한 행(하루치 데이터)에 대해 AI 점수(0~100)를 계산
    """
    try:
        curr = row['Close_Calc']
        ma5, ma20, ma60 = row['MA5'], row['MA20'], row['MA60']
        rsi = row['RSI']
        macd, sig = row['MACD'], row['Signal_Line']
        std20 = row['STD20']
        
        score = 50.0

        # 1. 추세
        if curr > ma60:
            score += 10
            div = (curr - ma60) / ma60
            score += (div * 33) if 0 < div < 0.15 else 2
        else:
            score -= 20
        
        if ma5 > ma20 > ma60: score += 10
        elif ma20 > ma60: score += 5

        # 2. 눌림목
        dist = (curr - ma20) / ma20
        abs_dist = abs(dist)
        if curr > ma60 and abs_dist <= 0.03:
            score += 20 * (1 - (abs_dist / 0.03))
        elif curr > ma60 and 0.03 < dist <= 0.08:
            score += 5
        elif dist > 0.10: # 과열
            score -= 15
            
        # 3. RSI
        if 40 <= rsi <= 60: score += 10 + ((rsi-40)*0.1)
        elif rsi < 30: score += 15
        elif rsi > 70: score -= 15
        elif 60 < rsi <= 70: score += 8
        
        # 4. MACD
        if macd > sig:
            score += 5
            if row['MACD_Hist'] > 0 and row['MACD_Hist'] > row['Prev_MACD_Hist']:
                score += 2
        else:
            score -= 5
            
        # 5. 변동성 페널티
        vol_ratio = std20 / curr if curr > 0 else 0
        if vol_ratio > 0.05: score -= (vol_ratio * 100)
        
        return max(0.0, min(100.0, score))
    except:
        return 0.0

# =========================================================
# 2. 개별 종목 백테스트 엔진
# =========================================================
def prepare_stock_data(ticker_info, start_date):
    """
    개별 종목의 데이터를 미리 준비하는 함수 (병렬 처리용)
    """
    code, name = ticker_info
    try:
        # 데이터 다운로드
        df = yf.download(code, start=start_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) < 60: return None
        
        # 지표 및 점수 계산
        df = calculate_indicators_for_backtest(df)
        df['AI_Score'] = df.apply(get_ai_score_row, axis=1)
        df['Ticker'] = code
        df['Name'] = name
        
        return df[['Close_Calc', 'AI_Score', 'Ticker', 'Name']]
    except:
        return None

def run_portfolio_backtest(targets, start_date, initial_capital, strategy_mode, max_hold_days, exchange_data, use_compound, selection_mode):
    """
    strategy_mode: 'Basic', 'SuperLocking', 'Sniper' (신규 추가)
    """
    # 1. 전 종목 데이터 병렬 준비
    all_dfs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(prepare_stock_data, t, start_date): t for t in targets}
        for future in futures:
            res = future.result()
            if res is not None: all_dfs.append(res)
            
    if not all_dfs: return pd.DataFrame(), pd.DataFrame()

    # 2. Market Data 통합
    market_data = {}
    for df in all_dfs:
        for date, row in df.iterrows():
            if date not in market_data: market_data[date] = []
            market_data[date].append(row)
            
    sorted_dates = sorted(market_data.keys())
    
    # 3. 환율 데이터 준비
    if isinstance(exchange_data, (float, int)):
        get_rate = lambda d: float(exchange_data)
    else:
        rate_dict = exchange_data.to_dict()
        def get_rate(d):
            ts = pd.Timestamp(d)
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
        for ticker, info in portfolio.items():
            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            if stock_row is None: continue 
            
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
                # -------------------------------------------------------
                # [전략 1] 기본 (Basic)
                # -------------------------------------------------------
                if strategy_mode == "Basic":
                    if score <= 45:
                        should_sell = True
                        sell_reason = "AI 45↓"
                
                # -------------------------------------------------------
                # [전략 2] 슈퍼 락킹 (SuperLocking)
                # -------------------------------------------------------
                elif strategy_mode == "SuperLocking":
                    if not info['mode_active'] and profit_ratio >= 0.03:
                        portfolio[ticker]['mode_active'] = True
                        portfolio[ticker]['max_price'] = curr_price_krw
                    
                    if info['mode_active']:
                        if curr_price_krw > portfolio[ticker]['max_price']:
                            portfolio[ticker]['max_price'] = curr_price_krw
                        if curr_price_krw <= portfolio[ticker]['max_price'] * 0.98: # -2% Trailing
                            should_sell = True
                            sell_reason = "💎 락킹 익절"
                    else:
                        if score <= 45:
                            should_sell = True
                            sell_reason = "방어(45↓)"

                # -------------------------------------------------------
                # [전략 3] AI 스나이퍼 (Sniper) - NEW!
                # -------------------------------------------------------
                elif strategy_mode == "Sniper":
                    # a. 손절 (Hard Stop): -3% 도달 시 즉시 매도
                    if profit_ratio <= -0.03:
                        should_sell = True
                        sell_reason = "⚡ 칼손절(-3%)"
                    
                    # b. 익절 (Smart Trailing)
                    # 수익이 5% 넘으면 트레일링 모드 발동
                    elif not info['mode_active'] and profit_ratio >= 0.05:
                        portfolio[ticker]['mode_active'] = True
                        portfolio[ticker]['max_price'] = curr_price_krw
                    
                    if info['mode_active']:
                        # 고점 갱신
                        if curr_price_krw > portfolio[ticker]['max_price']:
                            portfolio[ticker]['max_price'] = curr_price_krw
                        
                        # 고점 대비 -3% 하락 시 익절 (슈퍼락킹보다 여유있게)
                        if curr_price_krw <= portfolio[ticker]['max_price'] * 0.97:
                            should_sell = True
                            sell_reason = "🎯 스나이퍼 익절"
                    
                    # c. 추세 이탈 (점수가 40점 미만으로 깨지면 매도)
                    if not should_sell and score < 40:
                         should_sell = True
                         sell_reason = "추세 이탈(40↓)"

            # 매도 실행
            if should_sell:
                return_amt = info['shares'] * curr_price_krw * (1 - fee_sell)
                balance += return_amt
                
                trades_log.append({
                    'ticker': ticker, 'name': info['name'], 'date': date, 'type': 'sell',
                    'price': curr_price_raw, 'score': score, 'profit': profit_pct, 
                    'reason': sell_reason, 'balance': balance
                })
                sell_list.append(ticker)

        for t in sell_list: del portfolio[t]

        # B. 신규 매수 (Buy Logic)
        if len(portfolio) < max_slots:
            candidates = []
            
            for row in daily_stocks:
                ticker = row['Ticker']
                if ticker in portfolio: continue 
                
                score = row['AI_Score']
                price_raw = row['Close_Calc']
                price_krw = price_raw * (1.0 if ".KS" in ticker else current_rate)
                
                entry_signal = False
                reason = ""
                
                # 진입 조건
                if strategy_mode == "Basic" and score >= 65:
                    entry_signal = True; reason = "AI 65↑"
                elif strategy_mode == "SuperLocking" and score >= 80:
                    entry_signal = True; reason = "강력매수(80↑)"
                elif strategy_mode == "Sniper" and score >= 70: # 스나이퍼는 70점
                    entry_signal = True; reason = "스나이퍼(70↑)"
                
                if entry_signal:
                    candidates.append({
                        'ticker': ticker, 'name': row['Name'], 'price_raw': price_raw,
                        'price_krw': price_krw, 'score': score, 'reason': reason
                    })
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            open_slots = max_slots - len(portfolio)
            buy_targets = candidates[:open_slots] 
            
            if buy_targets:
                if use_compound:
                    per_stock_budget = balance / open_slots
                else:
                    per_stock_budget = min(balance, initial_capital / max_slots)
                
                for target in buy_targets:
                    budget = min(balance, per_stock_budget)
                    fee_buy = 0.00015 if ".KS" in target['ticker'] else 0.001
                    if target['price_krw'] > 0:
                        shares = int(budget / (target['price_krw'] * (1 + fee_buy)))
                    else: shares = 0
                    
                    if shares > 0:
                        cost = shares * target['price_krw'] * (1 + fee_buy)
                        balance -= cost
                        portfolio[target['ticker']] = {
                            'name': target['name'], 'shares': shares, 'avg_price': target['price_krw'],
                            'buy_date': date, 'mode_active': False, 'max_price': 0       
                        }
                        trades_log.append({
                            'ticker': target['ticker'], 'name': target['name'], 'date': date, 
                            'type': 'buy', 'price': target['price_raw'], 'score': target['score'], 
                            'profit': 0, 'reason': target['reason'], 'balance': balance
                        })

        # C. 자산 평가
        current_equity = balance
        for ticker, info in portfolio.items():
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
    st.caption("AI 전략 시뮬레이터 v2.0 (환율/복리/신규전략 탑재)")
    
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
        # 🌟 전략 3개로 확장
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
        # 매핑
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
    if start_btn:
        # 환율 준비
        exchange_data_payload = 1430.0
        if exchange_arg_val == "DYNAMIC":
            with st.spinner("💱 환율 데이터 수집 중..."):
                try:
                    ex_df = yf.download("KRW=X", start=str(bt_start_date), progress=False)
                    if isinstance(ex_df.columns, pd.MultiIndex):
                        ex_df.columns = ex_df.columns.get_level_values(0)
                    exchange_data_payload = ex_df['Close']
                except: pass
        else:
            exchange_data_payload = float(exchange_arg_val)

        # 시뮬레이션
        with st.spinner(f"🔄 [{selected_strategy}] 전략으로 과거를 여행하는 중..."):
            targets = list(TICKER_MAP.items())
            trade_df, equity_df = run_portfolio_backtest(
                targets, str(bt_start_date), initial_cap_input, strat_code, 
                max_hold_days, exchange_data_payload, comp_mode, selection_code
            )
        
        # --------------------------------------------------------------------------------
        # 3. 결과 대시보드
        # --------------------------------------------------------------------------------
        if not trade_df.empty and not equity_df.empty:
            final_equity = equity_df.iloc[-1]['equity']
            total_return = (final_equity - initial_cap_input) / initial_cap_input * 100
            profit_amt = final_equity - initial_cap_input
            
            sells = trade_df[trade_df['type'] == 'sell']
            win_count = len(sells[sells['profit'] > 0])
            total_sells = len(sells)
            win_rate = (win_count / total_sells * 100) if total_sells > 0 else 0.0
            
            st.success(f"✅ 완료! 최종 자산: {final_equity:,.0f}원")
            
            with st.container():
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("총 수익률", f"{total_return:,.2f}%")
                k2.metric("승률", f"{win_rate:.1f}%", f"{win_count}승/{total_sells}전")
                
                amt_str = f"{profit_amt/100000000:,.2f}억" if abs(profit_amt) > 1e8 else f"{profit_amt/10000:,.0f}만"
                k3.metric("총 수익금", f"{amt_str}원", delta_color="normal")
                k4.metric("매매 횟수", f"{len(trade_df)//2}회")

            st.divider()

            # 자산 그래프
            fig = px.line(equity_df, x='date', y='equity', title=f"자산 성장 ({selected_strategy})")
            fig.add_hline(y=initial_cap_input, line_dash="dash", line_color="gray", annotation_text="원금")
            fig.update_traces(fill='tozeroy', line=dict(color='#00CC96', width=2))
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # 🔍 상세 타점 분석 (오류 수정됨)
            st.subheader("🔍 매매 타점 분석기")
            
            traded_tickers = trade_df['ticker'].unique()
            ticker_options = [f"{TICKER_MAP.get(t, t)} ({t})" for t in traded_tickers]
            
            if len(ticker_options) > 0:
                selected_option = st.selectbox("종목 선택", ticker_options)
                selected_ticker = selected_option.split('(')[-1].replace(')', '')
                selected_name = TICKER_MAP.get(selected_ticker, selected_ticker)

                my_trades = trade_df[trade_df['ticker'] == selected_ticker].sort_values('date')
                
                with st.spinner("차트 로딩..."):
                    chart_data = yf.download(selected_ticker, start=str(bt_start_date), progress=False, auto_adjust=True)
                    if isinstance(chart_data.columns, pd.MultiIndex):
                        chart_data.columns = chart_data.columns.get_level_values(0)
                    # 중복 컬럼 제거 (DuplicateError 방지)
                    chart_data = chart_data.loc[:, ~chart_data.columns.duplicated()]
                
                if not chart_data.empty:
                    fig_d = go.Figure()
                    fig_d.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'], mode='lines', name='주가', line=dict(color='gray')))
                    
                    buys = my_trades[my_trades['type'] == 'buy']
                    if not buys.empty:
                        fig_d.add_trace(go.Scatter(x=buys['date'], y=buys['price'], mode='markers', name='매수', 
                                                   marker=dict(symbol='triangle-up', color='red', size=12)))
                    
                    sells = my_trades[my_trades['type'] == 'sell']
                    if not sells.empty:
                        fig_d.add_trace(go.Scatter(x=sells['date'], y=sells['price'], mode='markers', name='매도', 
                                                   marker=dict(symbol='triangle-down', color='blue', size=12),
                                                   text=[f"{p:.1f}%" for p in sells['profit']], hovertemplate='수익률: %{text}'))
                    
                    fig_d.update_layout(title=f"{selected_name} 매매 복기", height=500, template="plotly_dark")
                    st.plotly_chart(fig_d, use_container_width=True)
                    
                    st.dataframe(my_trades[['date', 'type', 'price', 'profit', 'reason', 'score']], hide_index=True, use_container_width=True)
            else:
                st.info("거래 내역이 없습니다.")

            # 전체 로그
            st.subheader("📝 전체 거래 일지")
            log_df = trade_df.copy()
            log_df['date'] = log_df['date'].dt.date
            st.dataframe(
                log_df[['date', 'name', 'type', 'price', 'profit', 'balance', 'reason']].sort_values('date', ascending=False),
                hide_index=True, use_container_width=True, height=400,
                column_config={
                    "price": st.column_config.NumberColumn("가격", format="%.2f"),
                    "profit": st.column_config.NumberColumn("수익률", format="%.2f%%"),
                    "balance": st.column_config.NumberColumn("잔고", format="%d원")
                }
            )
        else:
            st.warning("매매 신호가 발생하지 않았습니다.")
