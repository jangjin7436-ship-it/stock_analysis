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
st.set_page_config(layout="wide", page_title="Stable-Alpha: 역변동성 기반 평균회귀 시스템")

# 기본 설정값
DEFAULT_TICKERS = {
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
RISK_FREE_RATE = 0.04  # 샤프 지수 계산용 무위험 이자율 (4%)

# -----------------------------------------------------------------------------
# 클래스 1: 지표 엔진 (Indicator Engine) - [수정됨]
# 입력값이 Series가 되도록 강제하여 차원 오류 방지
# -----------------------------------------------------------------------------
class IndicatorEngine:
    @staticmethod
    def _ensure_series(data):
        """데이터가 DataFrame일 경우 Series로 변환 (Squeeze)"""
        if isinstance(data, pd.DataFrame):
            return data.squeeze()
        return data

    @staticmethod
    def calculate_rsi(series, period=2):
        series = IndicatorEngine._ensure_series(series)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)

    @staticmethod
    def calculate_sma(series, period):
        series = IndicatorEngine._ensure_series(series)
        return series.rolling(window=period).mean()

    @staticmethod
    def calculate_mfi(high, low, close, volume, period=14):
        # 모든 입력을 1차원 Series로 강제 변환
        high = IndicatorEngine._ensure_series(high)
        low = IndicatorEngine._ensure_series(low)
        close = IndicatorEngine._ensure_series(close)
        volume = IndicatorEngine._ensure_series(volume)

        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        delta = typical_price.diff()
        pos_flow = pd.Series(np.where(delta > 0, money_flow, 0), index=typical_price.index)
        neg_flow = pd.Series(np.where(delta < 0, money_flow, 0), index=typical_price.index)
        
        raw_pos_flow = pos_flow.rolling(window=period).sum()
        raw_neg_flow = neg_flow.rolling(window=period).sum()
        
        money_ratio = raw_pos_flow / raw_neg_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi.fillna(50) 

    @staticmethod
    def calculate_adx(high, low, close, period=14):
        high = IndicatorEngine._ensure_series(high)
        low = IndicatorEngine._ensure_series(low)
        close = IndicatorEngine._ensure_series(close)
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='outer').max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.fillna(0)

    @staticmethod
    def calculate_volatility(close, window=20):
        close = IndicatorEngine._ensure_series(close)
        return close.pct_change().rolling(window=window).std() * np.sqrt(252)


# -----------------------------------------------------------------------------
# 클래스 2: 데이터 로더 (Data Loader) - [수정됨]
# MultiIndex 컬럼을 확실하게 1차원으로 펴주는 로직 추가
# -----------------------------------------------------------------------------
class DataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        fetch_start = self.start_date - timedelta(days=365)
        data_dict = {}
        
        def get_ticker_data(ticker):
            try:
                # auto_adjust=True: 수정주가(배당락/분할 반영) 사용
                df = yf.download(ticker, start=fetch_start, end=self.end_date, progress=False, auto_adjust=True)
                if len(df) > 200:
                    return ticker, df
            except Exception:
                return ticker, None
            return ticker, None

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(get_ticker_data, self.tickers))

        for ticker, df in results:
            if df is not None and not df.empty:
                # [핵심 수정] MultiIndex 컬럼 처리 (('Close', 'AAPL') -> 'Close')
                # 기존의 try-except 구문을 제거하고 더 확실한 방법 사용
                if isinstance(df.columns, pd.MultiIndex):
                    # 레벨 0(Price Type)만 남기고 나머지(Ticker) 레벨 삭제
                    df.columns = df.columns.get_level_values(0)

                # 데이터 정제: 중복된 컬럼이 있다면 제거
                df = df.loc[:, ~df.columns.duplicated()]

                # 지표 계산 (이제 df['Close']는 확실히 1차원 Series입니다)
                try:
                    df['RSI'] = IndicatorEngine.calculate_rsi(df['Close'], period=2)
                    df['SMA'] = IndicatorEngine.calculate_sma(df['Close'], period=200)
                    df['MFI'] = IndicatorEngine.calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'], period=14)
                    df['ADX'] = IndicatorEngine.calculate_adx(df['High'], df['Low'], df['Close'], period=14)
                    df['Volatility'] = IndicatorEngine.calculate_volatility(df['Close'], window=20)
                    
                    df['NextOpen'] = df['Open'].shift(-1) 
                    
                    df.dropna(inplace=True)
                    data_dict[ticker] = df
                except Exception as e:
                    # 계산 중 에러 발생 시 해당 종목 스킵하고 로그 출력 (디버깅용)
                    print(f"Error calculating indicators for {ticker}: {e}")
                    continue
        
        return data_dict

# -----------------------------------------------------------------------------
# 클래스 3: 전략 엔진 (Strategy Engine)
# -----------------------------------------------------------------------------
class StrategyEngine:
    def __init__(self, data_dict, initial_capital, max_holding_days=10):
        self.data_dict = data_dict
        self.initial_capital = initial_capital
        self.max_holding_days = max_holding_days
        self.trades = []  # [수정] 빈 리스트로 초기화 (기존 문법 오류 발생 지점)
        self.equity_curve = {}

    def run_backtest(self, start_date, end_date):
        # 모든 종목의 날짜 인덱스 통합 및 정렬
        all_dates = sorted(list(set([d for df in self.data_dict.values() for d in df.index if d >= pd.to_datetime(start_date) and d <= pd.to_datetime(end_date)])))
        
        cash = self.initial_capital
        positions = {} 
        
        for current_date in all_dates:
            # ---------------------------------------------------------
            # 1. 청산(Exit) 로직 처리
            # ---------------------------------------------------------
            tickers_to_sell = []  # [수정] 빈 리스트로 초기화
            
            for ticker, pos in positions.items():
                df = self.data_dict[ticker]
                if current_date not in df.index: continue
                
                row = df.loc[current_date]
                days_held = (current_date - pos['entry_date']).days
                
                price = row['Close']
                rsi = row['RSI'] # [수정] 컬럼명 명시
                
                # 청산 조건
                is_profit_target = rsi > 75
                is_time_stop = days_held >= self.max_holding_days
                is_stop_loss = price < pos['entry_price'] * 0.90
                
                if is_profit_target or is_time_stop or is_stop_loss:
                    revenue = pos['shares'] * price
                    cash += revenue
                    
                    pnl = (revenue - (pos['shares'] * pos['entry_price']))
                    pnl_pct = (price - pos['entry_price']) / pos['entry_price']
                    
                    reason = 'Time Stop' if is_time_stop else ('Stop Loss' if is_stop_loss else 'Profit Target')
                    
                    self.trades.append({
                        'Ticker': ticker,
                        'Entry Date': pos['entry_date'],
                        'Exit Date': current_date,
                        'Days Held': days_held,
                        'Entry Price': pos['entry_price'],
                        'Exit Price': price,
                        'PnL': pnl,
                        'Return (%)': pnl_pct * 100,
                        'Reason': reason
                    })
                    tickers_to_sell.append(ticker)
            
            # 포지션 목록에서 제거
            for t in tickers_to_sell:
                del positions[t]
                
            # ---------------------------------------------------------
            # 2. 진입(Entry) 로직 처리
            # ---------------------------------------------------------
            MAX_POSITIONS = 10
            available_slots = MAX_POSITIONS - len(positions)
            
            candidates = []  # [수정] 빈 리스트로 초기화
            
            if available_slots > 0:
                for ticker, df in self.data_dict.items():
                    if ticker in positions: continue
                    if current_date not in df.index: continue
                    
                    row = df.loc[current_date]
                    
                    # [수정] row 자체 비교가 아닌 구체적인 컬럼(SMA, RSI, ADX)과 비교하도록 수정
                    if (row['Close'] > row['SMA'] and 
                        row['RSI'] < 10 and 
                        row['ADX'] > 20 and
                        row['MFI'] < 40):
                        
                        vol = row['Volatility'] if row['Volatility'] > 0 else 0.01
                        inv_vol = 1 / vol
                        candidates.append((ticker, inv_vol, row['Close']))
            
            # ---------------------------------------------------------
            # 3. 자금 집행 (역변동성 가중 - Risk Parity)
            # ---------------------------------------------------------
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                selected = candidates[:available_slots]
                
                total_inv_vol = sum([x[1] for x in selected])
                investable_cash = cash * (len(selected) / MAX_POSITIONS)
                
                for ticker, inv_vol, price in selected:
                    weight = inv_vol / total_inv_vol
                    position_value = investable_cash * weight
                    
                    if position_value > price:
                        shares = position_value / price
                        cash -= (shares * price)
                        positions[ticker] = {
                            'shares': shares,
                            'entry_date': current_date,
                            'entry_price': price
                        }
            
            # ---------------------------------------------------------
            # 4. 자산 가치 평가
            # ---------------------------------------------------------
            current_equity = cash
            for ticker, pos in positions.items():
                if current_date in self.data_dict[ticker].index:
                    current_equity += pos['shares'] * self.data_dict[ticker].loc[current_date]['Close']
                else:
                    current_equity += pos['shares'] * pos['entry_price']
            
            self.equity_curve[current_date] = current_equity
            
        return pd.Series(self.equity_curve), pd.DataFrame(self.trades)

# -----------------------------------------------------------------------------
# 메인 애플리케이션 UI (Streamlit)
# -----------------------------------------------------------------------------
st.title("🛡️ Stable-Alpha: 변동성 제어형 평균회귀 시스템")
st.markdown("""
이 시스템은 **시점 의존적인 수익률 변동성** 문제를 해결하기 위해 설계되었습니다.
단순 금액 배분이 아닌 **역변동성 가중(Inverse Volatility Weighting)**을 사용하여 리스크를 제어하며,
**10일 타임 컷(Time Stop)**을 엄격하게 적용하여 단기 자금 회전율을 극대화합니다.
""")

with st.sidebar:
    st.header("전략 파라미터 설정")
    st.info("💡 팁: 상관관계가 낮은 다양한 섹터의 우량주를 섞을수록 변동성 제어 효과가 커집니다.")
    
    input_tickers = st.text_area(
        "대상 종목 (쉼표로 구분)", 
        ", ".join(DEFAULT_TICKERS),
        height=150
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("백테스트 시작일", datetime(2022, 1, 1))
    with col2:
        end_date = st.date_input("백테스트 종료일", datetime.today())
        
    initial_cap = st.number_input("초기 자본금 ($)", 10000, 10000000, 100000)
    max_hold = st.slider("최대 보유 기간 (일)", 5, 20, 10, help="사용자 제약조건: 최대 2주(10거래일)")

if st.button("🚀 전략 백테스트 실행"):
    if start_date >= end_date:
        st.error("시작일은 종료일보다 빨라야 합니다.")
        st.stop()

    ticker_list = [x.strip().upper() for x in input_tickers.split(',') if x.strip()]
    
    with st.spinner(f"데이터 수집 및 지표 계산 중... ({len(ticker_list)}개 종목)"):
        loader = DataLoader(ticker_list, pd.Timestamp(start_date), pd.Timestamp(end_date))
        data_store = loader.fetch_data()
        
        if not data_store:
            st.error("데이터를 가져올 수 없습니다. 티커를 확인해주세요.")
            st.stop()
            
    with st.spinner("이벤트 기반 시뮬레이션 및 역변동성 가중 적용 중..."):
        engine = StrategyEngine(data_store, initial_cap, max_holding_days=max_hold)
        equity_series, trade_log = engine.run_backtest(start_date, end_date)
        
        if equity_series.empty:
            st.warning("거래가 발생하지 않았습니다. 조건을 완화하거나 기간을 늘려보세요.")
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
            
            st.subheader("📊 전략 성과 리포트")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("총 수익률", f"{total_return*100:.2f}%")
            kpi2.metric("연환산 수익률 (CAGR)", f"{annualized_return*100:.2f}%", help="목표: >10%")
            kpi3.metric("샤프 지수 (Sharpe)", f"{sharpe:.2f}", help=">1.0: 양호, >2.0: 우수")
            kpi4.metric("최대 낙폭 (MDD)", f"{max_dd*100:.2f}%", help="리스크 관리의 핵심 지표")
            
            tab1, tab2, tab3 = st.tabs(["수익 곡선", "매매 일지", "전략 해설"])
            
            with tab1:
                fig = px.line(equity_series, title="Portfolio Equity Curve")
                fig.add_hline(y=initial_cap, line_dash="dash", line_color="red", annotation_text="Initial Capital")
                st.plotly_chart(fig, use_container_width=True)
                
                fig_dd = px.area(drawdown, title="Drawdown (Underwater Plot)")
                fig_dd.update_layout(yaxis_title="Drawdown %", showlegend=False)
                st.plotly_chart(fig_dd, use_container_width=True)
                
            with tab2:
                if not trade_log.empty:
                    st.dataframe(trade_log.sort_values(by="Exit Date", ascending=False), use_container_width=True)
                    
                    wins = trade_log[trade_log['PnL'] > 0]
                    win_rate = len(wins) / len(trade_log)
                    avg_hold = trade_log['Days Held'].mean()
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("승률 (Win Rate)", f"{win_rate*100:.1f}%")
                    col_b.metric("평균 보유 기간", f"{avg_hold:.1f} 일", help="10일 제한 준수 여부 확인")
                    
                    fig_reason = px.pie(trade_log, names='Reason', title="청산 사유 분포 (Exit Reasons)")
                    st.plotly_chart(fig_reason, use_container_width=True)
                else:
                    st.info("해당 기간 동안 거래 내역이 없습니다.")
                    
            with tab3:
                st.markdown("""
                ### 🧠 Stable-Alpha 전략의 핵심 메커니즘
                
                1. **왜 변동성이 줄어드는가? (Inverse Volatility Weighting)**
                   - 기존 방식: 모든 종목에 $1,000씩 투자 -> 변동성이 큰 종목이 계좌 수익률을 지배함.
                   - 본 전략: 변동성이 큰 종목은 비중을 줄이고, 안정적인 종목은 비중을 늘림.
                   - 결과: 어떤 종목이 신호를 주더라도 계좌 전체에 미치는 '위험 충격(Risk Impact)'이 일정해짐.
                
                2. **왜 수익률이 개선되는가? (Regime Filtering)**
                   - `ADX > 20` 및 `Price > SMA200` 필터를 통해 하락 추세를 피합니다.
                   - 상승장 속의 일시적 조정(Dip)만 골라내므로 승률이 상승합니다.
                
                3. **단타 원칙 준수 (Time Stop)**
                   - 10일(2주)이 지나면 무조건 청산하여 자금 회전율을 높입니다.
                """)
