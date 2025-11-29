import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import warnings

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 페이지 설정 및 상수 정의
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Stable-Alpha x AI Sniper: 하이브리드 전략 시스템")

# 기본 티커 목록 (코드 2의 유니버스 반영)
DEFAULT_TICKERS = [
    "INTC", "SOXL", "WDC", "NFLX", "KLAC", "BAC", "NEM", "FCX", 
    "NVDA", "ASML", "GE", "V", "BA", "TXN", "GM", "F", "DELL", "JNJ", 
    "QCOM", "XOM", "AVGO", "OXY", "SLB", "TQQQ", "UPRO", "FNGU", "BULZ", "TSLA", "AMD"
]

RISK_FREE_RATE = 0.04  # 샤프 지수 계산용 무위험 이자율 (4%)

# -----------------------------------------------------------------------------
# 클래스 1: 지표 엔진 (Indicator Engine)
# 코드 2의 정교한 지표(ATR, Disparity, AI Score) 로직 이식
# -----------------------------------------------------------------------------
class IndicatorEngine:
    @staticmethod
    def calculate_indicators(df):
        """
        코드 2의 지표 계산 로직 통합 (ATR, 이격도, 볼린저 밴드, RSI, MACD 등)
        """
        df = df.copy()
        
        # 실제 종가 기준 (yfinance auto_adjust=False 가정)
        # 데이터 로더에서 Adj Close 처리를 하겠지만, 계산상 편의를 위해 Close 컬럼 사용
        df['Close_Calc'] = df['Close']

        # 1. 이동평균
        df['MA5'] = df['Close_Calc'].rolling(5).mean()
        df['MA10'] = df['Close_Calc'].rolling(10).mean()
        df['MA20'] = df['Close_Calc'].rolling(20).mean()
        df['MA60'] = df['Close_Calc'].rolling(60).mean()
        df['MA120'] = df['Close_Calc'].rolling(120).mean()

        # 이격도 및 기울기
        df['Disparity_20'] = df['Close_Calc'] / df['MA20']
        df['MA20_Slope'] = df['MA20'].diff()
        df['MA60_Slope'] = df['MA60'].diff()

        # 2. 볼린저 밴드
        std = df['Close_Calc'].rolling(20).std()
        df['Upper_Band'] = df['MA20'] + (std * 2)
        df['Lower_Band'] = df['MA20'] - (std * 2)
        
        # 3. RSI (14일 표준)
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

        # 5. ATR (Average True Range) - 변동성 지표 핵심
        prev_close = df['Close_Calc'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - prev_close)
        tr3 = abs(df['Low'] - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # 6. 거래량 비율
        if 'Volume' in df.columns:
            df['Vol_MA20'] = df['Volume'].rolling(20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
        else:
            df['Vol_Ratio'] = 1.0

        # 7. 역변동성 가중을 위한 연율화 변동성 (코드 1 기능 유지)
        df['Volatility'] = df['Close_Calc'].pct_change().rolling(window=20).std() * np.sqrt(252)

        return df

    @staticmethod
    def get_ai_score(row):
        """
        코드 2의 AI 점수 산출 로직 (Score 0~100)
        """
        try:
            score = 50.0
            curr = row['Close_Calc']
            ma5, ma20, ma60 = row['MA5'], row['MA20'], row['MA60']
            rsi = row['RSI']
            
            # 1. 추세 판단
            if row['MA60_Slope'] > 0:
                score += 10.0
                if curr > ma60: score += 5.0
            else:
                score -= 5.0

            # 2. 진입 타이밍 (눌림목)
            if row['MA20_Slope'] > 0:
                if curr > ma20:
                    score += 5.0
                    # 눌림목 보너스
                    if curr < ma5 * 1.01: 
                        score += 5.0
            
            # 3. 과열 방지 (이격도)
            disparity = row['Disparity_20']
            if disparity > 1.10: score -= 20.0 # 과열
            elif disparity > 1.05: score -= 5.0

            # 4. 보조지표 혼합
            # MACD 반전
            if row['MACD_Hist'] > row['Prev_MACD_Hist']:
                score += 5.0
            
            # RSI 구간
            if 40 <= rsi <= 60: score += 5.0
            elif rsi > 70: score -= 10.0
            elif rsi < 30: score += 5.0

            # 볼린저 하단 반등
            if curr <= row['Lower_Band'] * 1.02:
                score += 10.0

            # 거래량 실린 양봉
            if row['Vol_Ratio'] >= 1.5 and curr > row['Open']:
                score += 5.0

            return max(0.0, min(100.0, score))
        except:
            return 0.0

# -----------------------------------------------------------------------------
# 클래스 2: 데이터 로더 (Data Loader)
# -----------------------------------------------------------------------------
class DataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        # 지표 계산(MA120 등)을 위해 넉넉히 앞선 데이터 로드
        fetch_start = self.start_date - timedelta(days=365)
        data_dict = {}
        
        def get_ticker_data(ticker):
            try:
                # 코드 2와 동일하게 auto_adjust=False 사용 (실제 가격 흐름 반영)
                df = yf.download(ticker, start=fetch_start, end=self.end_date, progress=False, auto_adjust=False)
                if len(df) > 120:
                    return ticker, df
            except Exception:
                return ticker, None
            return ticker, None

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(get_ticker_data, self.tickers))

        for ticker, df in results:
            if df is not None:
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        df = df.xs(ticker, axis=1, level=1)
                    except:
                        pass
                
                # 지표 계산 엔진 호출
                df = IndicatorEngine.calculate_indicators(df)
                
                # AI 점수 계산
                df['AI_Score'] = df.apply(IndicatorEngine.get_ai_score, axis=1)
                
                # NaN 제거
                df.dropna(inplace=True)
                data_dict[ticker] = df
        
        return data_dict

# -----------------------------------------------------------------------------
# 클래스 3: 전략 엔진 (Strategy Engine)
# 코드 1의 자금관리 구조에 코드 2의 매수/매도 알고리즘 이식
# -----------------------------------------------------------------------------
class StrategyEngine:
    def __init__(self, data_dict, initial_capital, max_holding_days=10):
        self.data_dict = data_dict
        self.initial_capital = initial_capital
        self.max_holding_days = max_holding_days
        self.trades = []
        self.equity_curve = {}
        
        # ATR Multiplier 설정 (코드 2의 설정값)
        self.atr_stop_mult = 2.0   # 손절 2 ATR
        self.atr_profit_mult = 3.0 # 익절 3 ATR
        self.atr_trail_mult = 2.5  # 트레일링 2.5 ATR

    def run_backtest(self, start_date, end_date):
        all_dates = sorted(list(set([d for df in self.data_dict.values() for d in df.index if d >= pd.to_datetime(start_date) and d <= pd.to_datetime(end_date)])))
        
        cash = self.initial_capital
        positions = {} 
        # positions 구조 변경: 
        # {ticker: {'shares': x, 'entry_price': p, 'entry_date': d, 'max_price': p, 'stop_loss': p, 'target_price': p}}

        for current_date in all_dates:
            # ---------------------------------------------------------
            # 1. 청산(Exit) 로직 처리 - 코드 2의 ATR 동적 청산 적용
            # ---------------------------------------------------------
            tickers_to_sell = []
            
            for ticker, pos in positions.items():
                df = self.data_dict[ticker]
                if current_date not in df.index: continue
                
                row = df.loc[current_date]
                raw_open = row['Open']
                raw_high = row['High']
                raw_low = row['Low']
                price = row['Close_Calc'] # 종가
                atr = row['ATR']
                score = row['AI_Score']

                days_held = (current_date - pos['entry_date']).days
                
                # 동적 청산 로직
                should_sell = False
                sell_reason = ""
                exit_price = price # 기본은 종가 청산

                # A. 트레일링 스탑 업데이트
                if raw_high > pos['max_price']:
                    positions[ticker]['max_price'] = raw_high
                    # 고점이 높아지면 손절라인도 올림 (ATR 기반)
                    new_stop = raw_high - (atr * self.atr_trail_mult)
                    if new_stop > pos['stop_loss']:
                        positions[ticker]['stop_loss'] = new_stop

                # B. 조건 검사
                # 1. 갭락/손절 (ATR 이탈)
                if raw_open < pos['stop_loss']:
                    should_sell = True
                    sell_reason = "Gap Stop (ATR)"
                    exit_price = raw_open
                elif raw_low < pos['stop_loss']:
                    should_sell = True
                    sell_reason = "Stop Loss (ATR)"
                    exit_price = pos['stop_loss'] * 0.995 # 슬리피지

                # 2. 익절 (목표가 도달)
                elif raw_high > pos['target_price']:
                    # 이미 목표가 넘었으면 분할 매도 혹은 전량 매도 -> 여기선 전량 처리
                    # 보수적으로 목표가에서 체결되었다고 가정
                    should_sell = True
                    sell_reason = "Profit Target (3ATR)"
                    exit_price = pos['target_price']

                # 3. 타임 컷 (Time Stop)
                elif days_held >= self.max_holding_days:
                    should_sell = True
                    sell_reason = f"Time Stop ({days_held}d)"

                # 4. 점수 급락 청산 (코드 2 로직)
                elif score < 30:
                    should_sell = True
                    sell_reason = "Score Drop (<30)"
                
                # 5. 수익권인데 점수 하락 시 차익 실현
                elif price > pos['entry_price'] * 1.05 and score < 45:
                    should_sell = True
                    sell_reason = "Profit Check (Score)"

                if should_sell:
                    revenue = pos['shares'] * exit_price
                    cash += revenue
                    
                    pnl = revenue - (pos['shares'] * pos['entry_price'])
                    pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']

                    self.trades.append({
                        'Ticker': ticker,
                        'Entry Date': pos['entry_date'],
                        'Exit Date': current_date,
                        'Days Held': days_held,
                        'Entry Price': pos['entry_price'],
                        'Exit Price': exit_price,
                        'PnL': pnl,
                        'Return (%)': pnl_pct * 100,
                        'Reason': sell_reason
                    })
                    tickers_to_sell.append(ticker)
            
            for t in tickers_to_sell:
                del positions[t]

            # ---------------------------------------------------------
            # 2. 진입(Entry) 로직 처리 - 코드 2의 AI Score 진입 적용
            # ---------------------------------------------------------
            MAX_POSITIONS = 10
            available_slots = MAX_POSITIONS - len(positions)
            
            candidates = []
            
            if available_slots > 0:
                for ticker, df in self.data_dict.items():
                    if ticker in positions: continue
                    if current_date not in df.index: continue
                    
                    row = df.loc[current_date]
                    
                    # 진입 조건: AI Score >= 70 (코드 2)
                    if row['AI_Score'] >= 70:
                        # 역변동성 계산 (코드 1의 자금 관리 철학 유지)
                        vol = row['Volatility'] if row['Volatility'] > 0 else 0.01
                        inv_vol = 1 / vol
                        
                        candidates.append({
                            'ticker': ticker,
                            'inv_vol': inv_vol,
                            'price': row['Close_Calc'],
                            'score': row['AI_Score'],
                            'atr': row['ATR'],
                            'vol_power': row['Vol_Ratio']
                        })
            
            # ---------------------------------------------------------
            # 3. 자금 집행 (역변동성 가중 + AI 점수 선정)
            # ---------------------------------------------------------
            if candidates:
                # 점수 높은 순 -> 거래량 파워 순 정렬 (코드 2 방식)
                candidates.sort(key=lambda x: (x['score'], x['vol_power']), reverse=True)
                selected = candidates[:available_slots]
                
                # 자금 배분은 역변동성(Risk Parity) 방식 사용 (코드 1 방식)
                # 선정된 종목끼리 위험 균형을 맞춤
                total_inv_vol = sum([x['inv_vol'] for x in selected])
                investable_cash = cash * (len(selected) / MAX_POSITIONS)
                
                for item in selected:
                    weight = item['inv_vol'] / total_inv_vol
                    position_value = investable_cash * weight
                    price = item['price']
                    
                    if position_value > price:
                        shares = position_value / price
                        cash -= (shares * price)
                        
                        # ATR 기반 목표가/손절가 설정
                        atr = item['atr']
                        stop_loss = price - (atr * self.atr_stop_mult)
                        target_price = price + (atr * self.atr_profit_mult)
                        
                        positions[item['ticker']] = {
                            'shares': shares,
                            'entry_date': current_date,
                            'entry_price': price,
                            'max_price': price,        # 트레일링 스탑용 고점
                            'stop_loss': stop_loss,    # 초기 손절가
                            'target_price': target_price # 1차 목표가
                        }

            # ---------------------------------------------------------
            # 4. 자산 가치 평가
            # ---------------------------------------------------------
            current_equity = cash
            for ticker, pos in positions.items():
                if current_date in self.data_dict[ticker].index:
                    current_equity += pos['shares'] * self.data_dict[ticker].loc[current_date]['Close_Calc']
                else:
                    current_equity += pos['shares'] * pos['entry_price']
            
            self.equity_curve[current_date] = current_equity
            
        return pd.Series(self.equity_curve), pd.DataFrame(self.trades)

# -----------------------------------------------------------------------------
# 메인 애플리케이션 UI (Streamlit)
# -----------------------------------------------------------------------------
st.title("🛡️ Stable-Alpha x AI Sniper")
st.markdown("""
**Code 1의 구조(역변동성 자금관리) + Code 2의 두뇌(AI 점수, ATR 청산)**가 결합된 하이브리드 시스템입니다.
- **진입:** AI Score >= 70 (추세 + 눌림목 + 거래량)
- **청산:** ATR 기반 동적 손절/익절/트레일링 스탑 + 타임 컷
- **자금:** 역변동성 가중(Low Volatility -> High Weight)
""")

with st.sidebar:
    st.header("파라미터 설정")
    input_tickers = st.text_area("대상 종목 (쉼표 구분)", ", ".join(DEFAULT_TICKERS), height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("종료일", datetime.today())
        
    initial_cap = st.number_input("초기 자본금 ($)", 10000, 10000000, 100000)
    max_hold = st.slider("타임 컷 (최대 보유일)", 5, 60, 20)

if st.button("🚀 하이브리드 전략 실행"):
    if start_date >= end_date:
        st.error("날짜 설정 오류")
        st.stop()

    ticker_list = [x.strip().upper() for x in input_tickers.split(',') if x.strip()]
    
    with st.spinner("데이터 수집 및 AI 지표(ATR, Score) 계산 중..."):
        loader = DataLoader(ticker_list, pd.Timestamp(start_date), pd.Timestamp(end_date))
        data_store = loader.fetch_data()
        
        if not data_store:
            st.error("데이터 로드 실패")
            st.stop()
            
    with st.spinner("시뮬레이션 진행 중 (ATR 청산 & 역변동성 배분)..."):
        engine = StrategyEngine(data_store, initial_cap, max_holding_days=max_hold)
        equity_series, trade_log = engine.run_backtest(start_date, end_date)
        
        if equity_series.empty:
            st.warning("거래 없음")
        else:
            total_return = (equity_series.iloc[-1] - initial_cap) / initial_cap
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            annualized_return = ((1 + total_return) ** (365/days)) - 1
            
            daily_ret = equity_series.pct_change().dropna()
            volatility = daily_ret.std() * np.sqrt(252)
            sharpe = (annualized_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
            
            running_max = equity_series.cummax()
            drawdown = (equity_series / running_max) - 1
            max_dd = drawdown.min()
            
            st.subheader("📊 성과 리포트")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("총 수익률", f"{total_return*100:.2f}%")
            k2.metric("CAGR", f"{annualized_return*100:.2f}%")
            k3.metric("Sharpe", f"{sharpe:.2f}")
            k4.metric("MDD", f"{max_dd*100:.2f}%")
            
            tab1, tab2 = st.tabs(["수익 곡선", "매매 일지"])
            
            with tab1:
                fig = px.line(equity_series, title="Portfolio Equity")
                fig.add_hline(y=initial_cap, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
                fig_dd = px.area(drawdown, title="Drawdown")
                st.plotly_chart(fig_dd, use_container_width=True)
                
            with tab2:
                if not trade_log.empty:
                    st.dataframe(trade_log.sort_values("Exit Date", ascending=False), use_container_width=True)
                    
                    wins = trade_log[trade_log['PnL'] > 0]
                    win_rate = len(wins) / len(trade_log)
                    st.metric("승률", f"{win_rate*100:.1f}%")
                    
                    fig_pie = px.pie(trade_log, names='Reason', title="청산 사유")
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("거래 내역 없음")
