import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# [기존 코드 재사용] 지표 계산 함수
# ---------------------------------------------------------
def calculate_indicators_for_backtest(df):
    """
    백테스트용 지표 계산 (기존 로직과 동일하되 전체 DF 반환)
    """
    df = df.copy()
    
    # 수정 종가 사용
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['Close_Calc'] = df[col]

    # 이동평균
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
    
    # MACD 히스토그램 및 전일 대비 증감 (로직 구현을 위해 shift 사용)
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1) # 전일 히스토그램

    # 볼린저밴드 관련 (STD20)
    df['STD20'] = df['Close_Calc'].rolling(20).std()
    
    # 모멘텀
    df['MOM10'] = df['Close_Calc'].pct_change(10)

    return df.dropna()

# ---------------------------------------------------------
# [핵심] 점수 계산 로직 (Row-by-Row 적용을 위해 변환)
# ---------------------------------------------------------
def get_score_from_row(row):
    """
    DataFrame의 한 행(row)을 받아 점수를 반환하는 함수
    (사용자의 analyze_advanced_strategy 로직을 행 단위로 분해)
    """
    try:
        curr = row['Close_Calc']
        ma5, ma20, ma60 = row['MA5'], row['MA20'], row['MA60']
        rsi = row['RSI']
        macd, sig = row['MACD'], row['Signal_Line']
        std20 = row['STD20']
        
        # 🟢 기본 점수
        score = 50.0

        # 1. 추세 (Trend)
        if curr > ma60:
            score += 10
            divergence_60 = (curr - ma60) / ma60
            if 0 < divergence_60 < 0.15:
                score += divergence_60 * 33
            else:
                score += 2
        else:
            score -= 20 # 역배열 감점

        if ma5 > ma20 > ma60: score += 10 # 정배열
        elif ma20 > ma60: score += 5

        # 2. 위치 & 눌림목
        dist_ma20 = (curr - ma20) / ma20
        abs_dist = abs(dist_ma20)

        if curr > ma60 and abs_dist <= 0.03: # 황금 눌림목
            proximity_score = 20 * (1 - (abs_dist / 0.03))
            score += proximity_score
        elif curr > ma60 and 0.03 < dist_ma20 <= 0.08:
            score += 5
        elif dist_ma20 > 0.10: # 과열
            score -= 15

        # 3. RSI
        if 40 <= rsi <= 60:
            score += 10 + ((rsi - 40) * 0.1)
        elif 30 <= rsi < 40:
            score += 5 + ((40 - rsi) * 0.5)
        elif 60 < rsi <= 70:
            score += 8
        elif rsi < 30: # 과매도
            score += 15
        elif rsi > 70: # 과매수
            score -= 15

        # 4. MACD
        macd_hist = row['MACD_Hist']
        if macd > sig:
            score += 5
            hist_bonus = min(5.0, (macd_hist / curr) * 1000) if curr > 0 else 0
            score += hist_bonus
            # 상승 에너지 확대 (전일 대비 히스토그램 증가)
            if macd_hist > 0 and macd_hist > row['Prev_MACD_Hist']:
                score += 2 # 가산점 (임의 부여)
        else:
            score -= 5

        # 5. 변동성 페널티
        vol_ratio = std20 / curr if curr > 0 else 0
        if vol_ratio > 0.05:
            score -= (vol_ratio * 100)

        return max(0.0, min(100.0, score))
    except:
        return 0.0

# ---------------------------------------------------------
# [백테스트] 시뮬레이션 엔진
# ---------------------------------------------------------
def run_backtest(ticker, period="1y", initial_capital=10000000):
    # 1. 데이터 다운로드
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty: return None, "데이터 없음"
        
        # MultiIndex 컬럼 평탄화 (yfinance 최신버전 이슈 대응)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception as e:
        return None, f"다운로드 오류: {e}"

    # 2. 지표 계산
    df = calculate_indicators_for_backtest(df)
    if len(df) < 60: return None, "데이터 부족 (최소 60일 이상 필요)"

    # 3. AI 점수 과거 데이터 생성 (apply 사용)
    #    lambda를 사용하여 각 행(row)에 대해 점수 계산 로직 수행
    df['AI_Score'] = df.apply(lambda row: get_score_from_row(row), axis=1)

    # 4. 매매 시뮬레이션
    balance = initial_capital
    shares = 0
    avg_price = 0
    trades = []
    equity_curve = []
    
    # 수수료 설정 (국내/해외 구분)
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
    fee_buy = 0.00015 if is_kr else 0.001  # 매수 수수료 (가정)
    fee_sell = 0.003 if is_kr else 0.001   # 매도 수수료+세금 (가정)

    for date, row in df.iterrows():
        price = row['Close_Calc']
        score = row['AI_Score']
        
        # 전략 로직
        # 매수: 점수 >= 65 (매수 우위) AND 미보유
        if score >= 65 and shares == 0:
            can_buy_qty = int(balance / (price * (1 + fee_buy)))
            if can_buy_qty > 0:
                shares = can_buy_qty
                buy_cost = shares * price * (1 + fee_buy)
                balance -= buy_cost
                avg_price = price
                trades.append({
                    "Date": date, "Type": "Buy", "Price": price, 
                    "Score": score, "Balance": balance
                })

        # 매도: 점수 <= 45 (관망/매도) AND 보유 중
        elif score <= 45 and shares > 0:
            sell_amount = shares * price * (1 - fee_sell)
            balance += sell_amount
            
            # 수익률 계산
            profit = (price - avg_price) / avg_price * 100
            trades.append({
                "Date": date, "Type": "Sell", "Price": price, 
                "Score": score, "Profit_Pct": profit, "Balance": balance
            })
            shares = 0
            avg_price = 0

        # 자산 평가액 기록 (현금 + 주식평가액)
        current_equity = balance + (shares * price)
        equity_curve.append(current_equity)

    df['Equity'] = equity_curve
    return df, trades

# ---------------------------------------------------------
# UI 부분 (백테스트 탭)
# ---------------------------------------------------------
st.title("🧪 알고리즘 백테스트 (Backtest)")
st.caption("현재 AI 알고리즘을 과거 데이터에 적용하여 수익률을 검증합니다.")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    bt_ticker = st.text_input("종목 코드 입력", value="NVDA")
with col2:
    bt_period = st.selectbox("기간 설정", ["6mo", "1y", "2y", "5y"], index=1)
with col3:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 백테스트 실행", type="primary")

if run_btn:
    with st.spinner(f"{bt_ticker} 과거 데이터 분석 중..."):
        df_res, trades = run_backtest(bt_ticker, bt_period)
        
        if df_res is None:
            st.error(trades) # 에러 메시지 출력
        else:
            # 결과 계산
            initial_cap = 10000000
            final_cap = df_res['Equity'].iloc[-1]
            total_return = ((final_cap - initial_cap) / initial_cap) * 100
            
            # 벤치마크 (Buy & Hold) 수익률
            start_price = df_res['Close_Calc'].iloc[0]
            end_price = df_res['Close_Calc'].iloc[-1]
            buy_hold_return = ((end_price - start_price) / start_price) * 100

            # --- 결과 요약 표시 ---
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AI 전략 수익률", f"{total_return:.2f}%", delta_color="normal")
            m2.metric("존버(Buy&Hold) 수익률", f"{buy_hold_return:.2f}%")
            m3.metric("총 거래 횟수", f"{len([t for t in trades if t['Type']=='Sell'])}회")
            m4.metric("최종 자산", f"{final_cap:,.0f}")

            # --- 차트 그리기 (Plotly) ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])

            # 1. 주가 및 매매 포인트
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Close_Calc'], name="주가", line=dict(color='gray', width=1)), row=1, col=1)
            
            # 매수/매도 마커
            buy_dates = [t['Date'] for t in trades if t['Type'] == 'Buy']
            buy_prices = [t['Price'] for t in trades if t['Type'] == 'Buy']
            sell_dates = [t['Date'] for t in trades if t['Type'] == 'Sell']
            sell_prices = [t['Price'] for t in trades if t['Type'] == 'Sell']

            fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='매수 (Score>=65)',
                                     marker=dict(symbol='triangle-up', color='red', size=12)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='매도 (Score<=45)',
                                     marker=dict(symbol='triangle-down', color='blue', size=12)), row=1, col=1)

            # 2. AI 점수 흐름
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['AI_Score'], name="AI 점수", 
                                     line=dict(color='purple', width=1.5)), row=2, col=1)
            
            # 기준선 (65점, 45점)
            fig.add_hline(y=65, line_dash="dot", annotation_text="매수 기준(65)", row=2, col=1, line_color="red")
            fig.add_hline(y=45, line_dash="dot", annotation_text="매도 기준(45)", row=2, col=1, line_color="blue")

            fig.update_layout(height=600, title_text=f"{bt_ticker} AI 알고리즘 백테스트 결과")
            st.plotly_chart(fig, use_container_width=True)

            # --- 거래 기록 로그 ---
            with st.expander("📄 상세 거래 기록 보기"):
                trade_df = pd.DataFrame(trades)
                if not trade_df.empty:
                    trade_df['Date'] = trade_df['Date'].dt.date
                    trade_df['Profit_Pct'] = trade_df['Profit_Pct'].fillna(0).map(lambda x: f"{x:.2f}%" if x != 0 else "-")
                    trade_df['Price'] = trade_df['Price'].map(lambda x: f"{x:,.2f}")
                    trade_df['Balance'] = trade_df['Balance'].map(lambda x: f"{x:,.0f}")
                    trade_df['Score'] = trade_df['Score'].map(lambda x: f"{x:.1f}")
                    st.dataframe(trade_df, use_container_width=True)
                else:
                    st.write("거래 내역이 없습니다.")
