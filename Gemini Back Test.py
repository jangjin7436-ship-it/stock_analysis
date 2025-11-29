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
st.set_page_config(layout="wide", page_title="Stable-Alpha: AI 스나이퍼 전략 시스템")

# 기본 설정값
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "INTC", 
    "005930.KS", "000660.KS", "TQQQ", "SOXL"
]
RISK_FREE_RATE = 0.04  # 샤프 지수 계산용 무위험 이자율 (4%)

# -----------------------------------------------------------------------------
# 클래스 1: 지표 엔진 (Indicator Engine)
# 코드 B의 AI 점수 및 ATR, 보조지표 계산 로직 이식
# -----------------------------------------------------------------------------
class IndicatorEngine:
    @staticmethod
    def calculate_indicators(df):
        """
        코드 B의 지표 계산 로직 통합 (ATR, 이격도, 추세, AI Score)
        """
        df = df.copy()
        
        # [중요] 실제 종가 기준 계산 (수정주가 아님을 가정하거나 로직 통일)
        # yfinance에서 auto_adjust=False로 가져온 데이터를 가정
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
        
        # 2. 볼린저 밴드
        std = df['Close_Calc'].rolling(20).std()
        df['Upper_Band'] = df['MA20'] + (std * 2)
        df['Lower_Band'] = df['MA20'] - (std * 2)
        df['Band_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
        
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
        
        # [중요] ATR (Average True Range) - 변동성 지표
        prev_close = df['Close_Calc'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - prev_close)
        tr3 = abs(df['Low'] - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # 5. 거래량 강도
        if 'Volume' in df.columns:
            df['Vol_MA20'] = df['Volume'].rolling(20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
        else:
            df['Vol_Ratio'] = 1.0

        return df

    @staticmethod
    def get_ai_score_row(row):
        """
        코드 B의 AI 점수 계산 로직 (0~100점)
        """
        try:
            score = 50.0
            curr = row['Close_Calc']
            ma5, ma10, ma20, ma60 = row['MA5'], row['MA10'], row['MA20'], row['MA60']
            rsi = row['RSI']
            
            # 1. 추세 판단 (장기 이평선 기울기가 중요)
            if row['MA60_Slope'] > 0:
                score += 10.0
                if curr > ma60: score += 5.0
            else:
                score -= 5.0

            # 2. 진입 타이밍 (눌림목 우대)
            if row['MA20_Slope'] > 0:
                if curr > ma20:
                    score += 5.0
                    # 골든크로스 초입이거나 눌림목일 때 가산점
                    if curr < ma5 * 1.01: 
                        score += 5.0  # 눌림목 보너스
            
            # 3. 과열 방지 (이격도 필터)
            disparity = row['Disparity_20']
            if disparity > 1.10: 
                score -= 20.0  # 고점 추격 매수 방지
            elif disparity > 1.05:
                score -= 5.0

            # 4. 보조지표 혼합
            if row['MACD_Hist'] > row['Prev_MACD_Hist']:
                score += 5.0
            
            # RSI: 40~60 사이의 안정적 구간 선호
            if 40 <= rsi <= 60: score += 5.0
            elif rsi > 70: score -= 10.0  # 과열 경고
            elif rsi < 30: score += 5.0   # 과매도 반등 노리기

            # 볼린저 밴드 하단 터치 후 반등 시그널
            if curr <= row['Lower_Band'] * 1.02:
                score += 10.0 # 저점 매수 기회

            # 거래량 실린 양봉
            if row['Vol_Ratio'] >= 1.5 and curr > row['Open']:
                score += 5.0

            return max(0.0, min(100.0, score))
        except:
            return 0.0

# -----------------------------------------------------------------------------
# 클래스 2: 데이터 로더 (Data Loader)
# 멀티스레딩을 이용한 고속 데이터 수집 (코드 A 구조 유지, 로직 수정)
# -----------------------------------------------------------------------------
class DataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        # 이동평균선(120일) 계산을 위해 충분한 과거 데이터 로드
        fetch_start = self.start_date - timedelta(days=365)
        
        data_dict = {}
        
        def get_ticker_data(ticker):
            try:
                # 코드 B는 auto_adjust=False 사용 (실제 가격 흐름 반영 위함)
                df = yf.download(ticker, start=fetch_start, end=self.end_date, progress=False, auto_adjust=False)
                if len(df) > 120:
                    return ticker, df
            except Exception as e:
                return ticker, None
            return ticker, None

        # 병렬 처리로 다운로드 속도 향상
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(get_ticker_data, self.tickers))

        for ticker, df in results:
            if df is not None:
                # 멀티인덱스 컬럼 처리
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        df = df.xs(ticker, axis=1, level=1)
                    except:
                        pass
                
                # 지표 계산 적용
                df = IndicatorEngine.calculate_indicators(df)
                
                # NaN 제거
                df.dropna(inplace=True)
                
                # AI 점수 계산 (행별 적용)
                df['AI_Score'] = df.apply(IndicatorEngine.get_ai_score_row, axis=1)
                
                data_dict[ticker] = df
        
        return data_dict

# -----------------------------------------------------------------------------
# 클래스 3: 전략 엔진 (Strategy Engine)
# 코드 B의 핵심 알고리즘 (ATR 손절, AI 스코어 진입) 이식
# -----------------------------------------------------------------------------
class StrategyEngine:
    def __init__(self, data_dict, initial_capital, max_holding_days=10):
        self.data_dict = data_dict
        self.initial_capital = initial_capital
        self.max_holding_days = max_holding_days
        self.trades = []
        self.equity_curve = {}
        
        # 코드 B 스타일의 포트폴리오 관리를 위한 설정
        self.positions = {} # {ticker: {shares, avg_price, max_price_raw, buy_date}}
        self.max_slots = 5  # 최대 5종목 분산

    def run_backtest(self, start_date, end_date):
        """
        이벤트 기반 백테스팅 루프 (코드 B의 매매 로직 적용)
        """
        # 모든 종목의 날짜 인덱스 통합 및 정렬
        all_dates = sorted(list(set([d for df in self.data_dict.values() for d in df.index if d >= pd.to_datetime(start_date) and d <= pd.to_datetime(end_date)])))
        
        cash = self.initial_capital
        
        for current_date in all_dates:
            # ---------------------------------------------------------
            # 1. 청산(Exit) 로직 처리 (ATR 기반 동적 대응)
            # ---------------------------------------------------------
            tickers_to_sell = []
            
            # 보유 종목 순회
            # 정렬된 키 순서로 처리하여 결정론적 결과 보장
            for ticker in sorted(list(self.positions.keys())):
                pos = self.positions[ticker]
                df = self.data_dict[ticker]
                
                if current_date not in df.index: continue
                
                row = df.loc[current_date]
                
                # 현재가 정보
                raw_open = row['Open']
                raw_high = row['High']
                raw_low = row['Low']
                raw_close = row['Close_Calc']
                atr = row['ATR']
                score = row['AI_Score']
                
                # 보유 기간
                days_held = (current_date - pos['buy_date']).days
                
                # 매수 가격 (Raw)
                buy_price_raw = pos['avg_price']
                
                should_sell = False
                sell_reason = ""
                final_sell_price_raw = raw_close
                
                # --- [동적 손절/익절 로직] ---
                atr_multiplier_stop = 2.0  # 손절: 2 ATR
                stop_price_raw = buy_price_raw - (atr * atr_multiplier_stop)
                
                # Trailing Stop: 고점 갱신 시 손절 라인 상향
                current_max_raw = max(pos.get('max_price_raw', buy_price_raw), raw_high)
                self.positions[ticker]['max_price_raw'] = current_max_raw
                
                # 고점 대비 2.5 ATR 하락 시 청산
                trailing_stop = current_max_raw - (atr * 2.5)
                if trailing_stop > stop_price_raw:
                    stop_price_raw = trailing_stop
                
                # 1) 갭락 손절 (시가가 손절가 아래)
                if raw_open < stop_price_raw:
                    should_sell = True
                    sell_reason = "Gap Loss (ATR)"
                    final_sell_price_raw = raw_open
                    
                # 2) 장중 손절 (저가가 손절가 터치)
                elif raw_low < stop_price_raw:
                    should_sell = True
                    sell_reason = "Stop Loss (ATR)"
                    final_sell_price_raw = stop_price_raw * 0.995 # 슬리피지
                
                # 3) 만기 및 스코어 청산 (종가 기준 판단)
                if not should_sell:
                    # 수익권인데 점수가 나빠지면 차익 실현
                    if raw_close > buy_price_raw * 1.05 and score < 45:
                        should_sell = True
                        sell_reason = "Score Drop (Profit)"
                    
                    # 타임 컷
                    elif days_held >= self.max_holding_days:
                        should_sell = True
                        sell_reason = f"Time Stop ({days_held}d)"
                    
                    # 점수 급락
                    elif score < 30:
                        should_sell = True
                        sell_reason = "Score Crash (<30)"
                
                # 매도 실행
                if should_sell:
                    revenue = pos['shares'] * final_sell_price_raw
                    cash += revenue
                    
                    pnl = revenue - (pos['shares'] * buy_price_raw)
                    pnl_pct = (final_sell_price_raw - buy_price_raw) / buy_price_raw
                    
                    self.trades.append({
                        'Ticker': ticker,
                        'Entry Date': pos['buy_date'],
                        'Exit Date': current_date,
                        'Days Held': days_held,
                        'Entry Price': buy_price_raw,
                        'Exit Price': final_sell_price_raw,
                        'PnL': pnl,
                        'Return (%)': pnl_pct * 100,
                        'Reason': sell_reason,
                        'Score': score
                    })
                    tickers_to_sell.append(ticker)
            
            # 포지션 제거
            for t in tickers_to_sell:
                del self.positions[t]
            
            # ---------------------------------------------------------
            # 2. 진입(Entry) 로직 처리 (AI 점수 기반)
            # ---------------------------------------------------------
            available_slots = self.max_slots - len(self.positions)
            candidates = []
            
            if available_slots > 0:
                for ticker, df in self.data_dict.items():
                    if ticker in self.positions: continue
                    if current_date not in df.index: continue
                    
                    row = df.loc[current_date]
                    score = row['AI_Score']
                    
                    # 진입 조건: AI Score >= 70 (확실한 추세/눌림목)
                    if score >= 70:
                        vol_power = row.get('Vol_Ratio', 1.0)
                        price_raw = row['Close_Calc']
                        
                        candidates.append({
                            'ticker': ticker,
                            'score': score,
                            'vol_power': vol_power,
                            'price': price_raw
                        })
                
                # 점수 높은 순 -> 거래량 강도 순 정렬
                candidates.sort(key=lambda x: (x['score'], x['vol_power']), reverse=True)
                buy_targets = candidates[:available_slots]
                
                for target in buy_targets:
                    if cash <= 0: break
                    
                    # 자금 관리: 남은 슬롯 수에 비례하여 균등 분할
                    current_slots_left = self.max_slots - len(self.positions)
                    slot_budget = cash / current_slots_left
                    
                    price = target['price']
                    if price > 0 and slot_budget > price:
                        shares = int(slot_budget / price)
                        cost = shares * price
                        cash -= cost
                        
                        self.positions[target['ticker']] = {
                            'shares': shares,
                            'avg_price': price,
                            'buy_date': current_date,
                            'max_price_raw': price # ATR 트레일링 스탑 초기화
                        }
            
            # ---------------------------------------------------------
            # 3. 자산 가치 평가 (Mark-to-Market)
            # ---------------------------------------------------------
            current_equity = cash
            for ticker, pos in self.positions.items():
                if current_date in self.data_dict[ticker].index:
                    current_price = self.data_dict[ticker].loc[current_date]['Close_Calc']
                    current_equity += pos['shares'] * current_price
                else:
                    current_equity += pos['shares'] * pos['avg_price']
            
            self.equity_curve[current_date] = current_equity
            
        return pd.Series(self.equity_curve), pd.DataFrame(self.trades)

# -----------------------------------------------------------------------------
# 메인 애플리케이션 UI (Streamlit)
# -----------------------------------------------------------------------------
st.title("🛡️ Stable-Alpha: AI 스나이퍼 전략 시스템")
st.markdown("""
이 시스템은 **AI 점수 기반 스나이퍼 전략**을 사용합니다.
**AI Score(70점 이상)**로 진입하며, **ATR(평균 변동폭) 기반 동적 손절/익절** 라인을 사용하여 리스크를 제어합니다.
기존의 역변동성 가중 방식 대신 **슬롯 기반 자금 분할**을 통해 확실한 기회에만 집중합니다.
""")

# 사이드바 설정
with st.sidebar:
    st.header("전략 파라미터 설정")
    
    st.info("💡 AI Score는 추세, 눌림목, 거래량, 보조지표를 종합하여 0~100점으로 산출됩니다.")
    
    input_tickers = st.text_area(
        "대상 종목 (쉼표로 구분)", 
        ", ".join(DEFAULT_TICKERS),
        height=150
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("백테스트 시작일", datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("백테스트 종료일", datetime.today())
        
    initial_cap = st.number_input("초기 자본금 ($)", 10000, 10000000, 100000)
    max_hold = st.slider("최대 보유 기간 (일)", 5, 60, 20, help="타임 컷: 수익/손실 여부 상관없이 청산")

# 실행 버튼
if st.button("🚀 전략 백테스트 실행"):
    if start_date >= end_date:
        st.error("시작일은 종료일보다 빨라야 합니다.")
        st.stop()

    ticker_list = [x.strip().upper() for x in input_tickers.split(',') if x.strip()]
    
    with st.spinner(f"데이터 수집 및 AI 지표(ATR, Score) 계산 중... ({len(ticker_list)}개 종목)"):
        loader = DataLoader(ticker_list, pd.Timestamp(start_date), pd.Timestamp(end_date))
        data_store = loader.fetch_data()
        
        if not data_store:
            st.error("데이터를 가져올 수 없습니다. 티커를 확인해주세요.")
            st.stop()
            
    with st.spinner("이벤트 기반 시뮬레이션 (ATR 손절/익절 적용) 중..."):
        engine = StrategyEngine(data_store, initial_cap, max_holding_days=max_hold)
        equity_series, trade_log = engine.run_backtest(start_date, end_date)
        
        # ---------------------------------------------------------
        # 결과 분석 및 시각화
        # ---------------------------------------------------------
        if equity_series.empty:
            st.warning("거래가 발생하지 않았습니다. 조건을 완화하거나 기간을 늘려보세요.")
        else:
            # 주요 성과 지표 (KPI) 계산
            total_return = (equity_series.iloc[-1] - initial_cap) / initial_cap
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            annualized_return = ((1 + total_return) ** (365/days)) - 1 if days > 0 else 0
            
            daily_ret = equity_series.pct_change().dropna()
            volatility = daily_ret.std() * np.sqrt(252)
            sharpe = (annualized_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
            
            # MDD 계산
            running_max = equity_series.cummax()
            drawdown = (equity_series / running_max) - 1
            max_dd = drawdown.min()
            
            # KPI 대시보드
            st.subheader("📊 전략 성과 리포트")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("총 수익률", f"{total_return*100:.2f}%")
            kpi2.metric("연환산 수익률 (CAGR)", f"{annualized_return*100:.2f}%")
            kpi3.metric("샤프 지수 (Sharpe)", f"{sharpe:.2f}")
            kpi4.metric("최대 낙폭 (MDD)", f"{max_dd*100:.2f}%")
            
            # 탭 구성
            tab1, tab2, tab3 = st.tabs(["수익 곡선", "매매 일지", "전략 해설"])
            
            with tab1:
                # 수익 곡선 차트
                fig = px.line(equity_series, title="Portfolio Equity Curve")
                fig.add_hline(y=initial_cap, line_dash="dash", line_color="red", annotation_text="Initial Capital")
                st.plotly_chart(fig, use_container_width=True)
                
                # 낙폭 차트 (Underwter Plot)
                fig_dd = px.area(drawdown, title="Drawdown (Underwater Plot)")
                fig_dd.update_layout(yaxis_title="Drawdown %", showlegend=False)
                st.plotly_chart(fig_dd, use_container_width=True)
                
            with tab2:
                if not trade_log.empty:
                    st.dataframe(
                        trade_log.sort_values(by="Exit Date", ascending=False), 
                        use_container_width=True,
                        column_config={
                            "Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
                            "Score": st.column_config.NumberColumn(format="%.1f점")
                        }
                    )
                    
                    # 승률 계산
                    wins = trade_log[trade_log['PnL'] > 0]
                    win_rate = len(wins) / len(trade_log)
                    avg_hold = trade_log['Days Held'].mean()
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("승률 (Win Rate)", f"{win_rate*100:.1f}%")
                    col_b.metric("평균 보유 기간", f"{avg_hold:.1f} 일")
                    
                    # 청산 사유 분포
                    fig_reason = px.pie(trade_log, names='Reason', title="청산 사유 분포 (Exit Reasons)")
                    st.plotly_chart(fig_reason, use_container_width=True)
                else:
                    st.info("해당 기간 동안 거래 내역이 없습니다.")
                    
            with tab3:
                st.markdown("""
                ### 🧠 AI 스나이퍼 (ATR & Score) 전략 메커니즘
                
                1. **AI Score 진입 (Entry > 70점)**
                   - 추세(MA60), 눌림목(MA20 지지), 거래량 파워, RSI 안정권(40~60) 등을 종합 평가합니다.
                   - 단순 돌파가 아닌 '확실한 자리'를 선별하여 진입합니다.
                
                2. **ATR 기반 동적 손절 (Dynamic Risk Control)**
                   - 고정 % 손절이 아닌, 종목의 변동성(ATR)을 반영하여 손절가를 설정합니다.
                   - **Gap Loss**: 시가 갭락 발생 시 즉시 탈출
                   - **Trailing Stop**: 고점 대비 2.5 ATR 하락 시 익절하여 수익을 보존합니다.
                
                3. **자금 관리 (Slot Budgeting)**
                   - 역변동성 가중 대신, 최대 5개 슬롯에 자금을 균등 분배하여 확실한 종목에 집중 투자합니다.
                """)
