import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

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
def run_single_stock_backtest(ticker, name, start_date="2023-01-01", initial_capital=1000000, strategy_mode="Basic"):
    """
    strategy_mode: "Basic" (기본 65/45) 또는 "SuperLocking" (슈퍼 락킹)
    """
    try:
        # 데이터 수집
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) < 60: return None

        # 지표 및 AI 점수 계산
        df = calculate_indicators_for_backtest(df)
        df['AI_Score'] = df.apply(get_ai_score_row, axis=1)

        # 시뮬레이션 변수
        balance = initial_capital
        shares = 0
        avg_price = 0
        trades = []
        
        # 슈퍼 락킹 모드 전용 변수
        locking_mode = False  # 모드 발동 여부
        max_price_in_mode = 0 # 모드 진입 후 최고가
        
        # 수수료 설정
        fee_buy = 0.00015 if ".KS" in ticker else 0.001
        fee_sell = 0.003 if ".KS" in ticker else 0.001

        for date, row in df.iterrows():
            price = row['Close_Calc']
            score = row['AI_Score']
            
            # -----------------------------------------------
            # [전략 1] 기본 AI 전략 (Basic)
            # -----------------------------------------------
            if strategy_mode == "Basic":
                # 매수: 65점 이상 & 미보유
                if score >= 65 and shares == 0:
                    can_buy = int(balance / (price * (1 + fee_buy)))
                    if can_buy > 0:
                        shares = can_buy
                        balance -= shares * price * (1 + fee_buy)
                        avg_price = price
                        trades.append({'date': date, 'type': 'buy', 'price': price, 'score': score, 'reason': 'AI 65↑'})

                # 매도: 45점 이하 & 보유 중
                elif score <= 45 and shares > 0:
                    return_amt = shares * price * (1 - fee_sell)
                    balance += return_amt
                    profit_pct = (price - avg_price) / avg_price * 100
                    trades.append({'date': date, 'type': 'sell', 'price': price, 'score': score, 'profit': profit_pct, 'reason': 'AI 45↓'})
                    shares = 0
                    avg_price = 0

            # -----------------------------------------------
            # [전략 2] 슈퍼 락킹 전략 (SuperLocking)
            # -----------------------------------------------
            elif strategy_mode == "SuperLocking":
                # A. 매수: 80점 이상 (강력 매수) & 미보유
                if score >= 80 and shares == 0:
                    can_buy = int(balance / (price * (1 + fee_buy)))
                    if can_buy > 0:
                        shares = can_buy
                        balance -= shares * price * (1 + fee_buy)
                        avg_price = price
                        
                        # 모드 초기화
                        locking_mode = False
                        max_price_in_mode = 0
                        trades.append({'date': date, 'type': 'buy', 'price': price, 'score': score, 'reason': 'Strong Buy(80↑)'})
                
                # B. 보유 중 관리
                elif shares > 0:
                    curr_return = (price - avg_price) / avg_price
                    
                    # 1. 락킹 모드 발동 체크 (평단 대비 +3% 이상)
                    if not locking_mode and curr_return >= 0.03:
                        locking_mode = True
                        max_price_in_mode = price # 발동 시점 가격을 일단 최고가로 설정
                    
                    # 2. 모드 상태별 로직
                    if locking_mode:
                        # 모드 ON: 고점 갱신 확인
                        if price > max_price_in_mode:
                            max_price_in_mode = price
                        
                        # 모드 ON: 고점 대비 -2% 하락 시 매도 (익절)
                        threshold_price = max_price_in_mode * 0.98
                        if price <= threshold_price:
                            return_amt = shares * price * (1 - fee_sell)
                            balance += return_amt
                            profit_pct = (price - avg_price) / avg_price * 100
                            trades.append({'date': date, 'type': 'sell', 'price': price, 'score': score, 'profit': profit_pct, 'reason': '💎 Locking Trailing'})
                            shares = 0
                            locking_mode = False
                            
                    else:
                        # 모드 OFF (아직 +3% 못감): AI 점수 45 이하면 방어적 매도 (손절/본전)
                        # *주의: 3% 가기 전에 폭락하면 팔아야 하므로 최소한의 안전장치
                        if score <= 45:
                            return_amt = shares * price * (1 - fee_sell)
                            balance += return_amt
                            profit_pct = (price - avg_price) / avg_price * 100
                            trades.append({'date': date, 'type': 'sell', 'price': price, 'score': score, 'profit': profit_pct, 'reason': 'Defense(45↓)'})
                            shares = 0

        # 최종 평가금 계산
        final_price = df['Close_Calc'].iloc[-1]
        final_equity = balance + (shares * final_price)
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        return {
            "ticker": ticker,
            "name": name,
            "total_return": total_return,
            "final_equity": final_equity,
            "trade_count": len(trades) // 2,
            "trades": trades,
            "win_rate": np.mean([t['profit'] > 0 for t in trades if 'profit' in t]) * 100 if trades else 0
        }
    except Exception as e:
        return None

# =========================================================
# 3. UI 통합 (탭 추가)
# =========================================================
# (기존 코드의 tab1, tab2, tab3 정의 아래에 tab4를 추가한다고 가정)

tab4 = st.tabs(["📊 전체 백테스트 시뮬레이션"])[0] # 기존 tabs 리스트에 추가 필요

with tab4:
    st.markdown("### 🧪 포트폴리오 유니버스 백테스트")
    st.caption("과거 데이터 기반 전략 시뮬레이션")
    
    # 설정 UI
    col_set1, col_set2, col_set3 = st.columns([1, 1, 2])
    with col_set1:
        bt_start_date = st.date_input("시작일", value=pd.to_datetime("2024-01-01"))
    with col_set2:
        # 🌟 전략 선택 라디오 버튼
        selected_strategy = st.radio(
            "⚔️ 전략 선택", 
            ["기본 (Basic)", "슈퍼 락킹 (SuperLocking)"],
            captions=["매수 65↑ / 매도 45↓", "매수 80↑ / +3%후 고점대비 -2% 매도"]
        )
        # 문자열 매핑
        strat_code = "Basic" if "기본" in selected_strategy else "SuperLocking"
        
    with col_set3:
        st.write("")
        st.write("")
        start_btn = st.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True)

    if start_btn:
        results = []
        progress_text = st.empty()
        bar = st.progress(0)
        
        targets = list(TICKER_MAP.items())
        total_stocks = len(targets)
        
        # 병렬 처리 실행 (전략 모드 전달)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    run_single_stock_backtest, 
                    code, 
                    name, 
                    str(bt_start_date), 
                    1000000, 
                    strat_code  # 🌟 선택된 전략 전달
                ): code for code, name in targets
            }
            
            completed = 0
            for future in futures:
                res = future.result()
                if res: results.append(res)
                completed += 1
                bar.progress(completed / total_stocks)
                progress_text.text(f"[{selected_strategy}] 분석 중... ({completed}/{total_stocks})")

        bar.empty()
        progress_text.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            avg_return = df_res['total_return'].mean()
            win_rate_avg = df_res['win_rate'].mean()
            total_profit_sum = df_res['final_equity'].sum() - (1000000 * len(df_res))
            
            st.success(f"✅ {selected_strategy} 백테스트 완료!")
            
            # 결과 표시 (기존과 동일)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("평균 수익률", f"{avg_return:.2f}%", delta_color="normal")
            m2.metric("평균 승률", f"{win_rate_avg:.1f}%")
            m3.metric("총 종목 수", f"{len(df_res)}개")
            m4.metric("총 수익금", f"{total_profit_sum:,.0f}원")
            
            st.divider()
            
            c_best, c_worst = st.columns(2)
            with c_best:
                st.subheader("🏆 수익률 Top 5")
                top5 = df_res.sort_values('total_return', ascending=False).head(5)
                for _, r in top5.iterrows():
                    st.write(f"**{r['name']}**: +{r['total_return']:.1f}% ({r['trade_count']}회)")
            
            with c_worst:
                st.subheader("💀 수익률 Worst 5")
                worst5 = df_res.sort_values('total_return', ascending=True).head(5)
                for _, r in worst5.iterrows():
                    st.write(f"**{r['name']}**: {r['total_return']:.1f}% ({r['trade_count']}회)")
            
            st.markdown("#### 📄 상세 내역")
            st.dataframe(df_res[['name', 'total_return', 'win_rate', 'trade_count', 'final_equity']], use_container_width=True)
            
            # 히스토그램
            fig = px.histogram(df_res, x="total_return", nbins=20, title=f"[{selected_strategy}] 수익률 분포")
            fig.add_vline(x=avg_return, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("결과 없음")
