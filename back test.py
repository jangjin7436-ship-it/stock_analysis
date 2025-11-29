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
def run_single_stock_backtest(ticker, name, start_date="2023-01-01", initial_capital=1000000, strategy_mode="Basic", max_holding_days=0):
    """
    max_holding_days: 0이면 기간 제한 없음. 0보다 크면 해당 일수 경과 시 강제 매도 (Time Cut)
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
        
        # 추가된 변수: 매수일
        buy_date = None
        
        # 슈퍼 락킹 모드 변수
        locking_mode = False
        max_price_in_mode = 0
        
        # 수수료
        fee_buy = 0.00015 if ".KS" in ticker else 0.001
        fee_sell = 0.003 if ".KS" in ticker else 0.001

        for date, row in df.iterrows():
            price = row['Close_Calc']
            score = row['AI_Score']
            
            # -----------------------------------------------------------
            # [공통] 타임 컷 (Time Cut) 체크
            # 주식을 보유 중이고, 최대 보유 기간 설정이 되어 있다면 검사
            # -----------------------------------------------------------
            if shares > 0 and max_holding_days > 0 and buy_date is not None:
                # 경과일 계산 (현재 날짜 - 매수 날짜)
                days_held = (date - buy_date).days
                
                if days_held >= max_holding_days:
                    # 강제 매도 실행
                    return_amt = shares * price * (1 - fee_sell)
                    balance += return_amt
                    profit_pct = (price - avg_price) / avg_price * 100
                    trades.append({
                        'date': date, 'type': 'sell', 'price': price, 'score': score, 
                        'profit': profit_pct, 'reason': f'⏱️ TimeCut({days_held}일)'
                    })
                    shares = 0
                    buy_date = None
                    locking_mode = False
                    continue # 이번 턴 종료 (이미 팔았으므로 아래 로직 건너뜀)

            # -----------------------------------------------
            # [전략 1] 기본 AI 전략 (Basic)
            # -----------------------------------------------
            if strategy_mode == "Basic":
                # 매수
                if score >= 65 and shares == 0:
                    can_buy = int(balance / (price * (1 + fee_buy)))
                    if can_buy > 0:
                        shares = can_buy
                        balance -= shares * price * (1 + fee_buy)
                        avg_price = price
                        buy_date = date # 매수일 기록
                        trades.append({'date': date, 'type': 'buy', 'price': price, 'score': score, 'reason': 'AI 65↑'})

                # 매도
                elif score <= 45 and shares > 0:
                    return_amt = shares * price * (1 - fee_sell)
                    balance += return_amt
                    profit_pct = (price - avg_price) / avg_price * 100
                    trades.append({'date': date, 'type': 'sell', 'price': price, 'score': score, 'profit': profit_pct, 'reason': 'AI 45↓'})
                    shares = 0
                    buy_date = None

            # -----------------------------------------------
            # [전략 2] 슈퍼 락킹 전략 (SuperLocking)
            # -----------------------------------------------
            elif strategy_mode == "SuperLocking":
                # 매수 (80점 이상)
                if score >= 80 and shares == 0:
                    can_buy = int(balance / (price * (1 + fee_buy)))
                    if can_buy > 0:
                        shares = can_buy
                        balance -= shares * price * (1 + fee_buy)
                        avg_price = price
                        buy_date = date # 매수일 기록
                        
                        # 모드 초기화
                        locking_mode = False
                        max_price_in_mode = 0
                        trades.append({'date': date, 'type': 'buy', 'price': price, 'score': score, 'reason': 'Strong Buy(80↑)'})
                
                # 보유 관리
                elif shares > 0:
                    curr_return = (price - avg_price) / avg_price
                    
                    if not locking_mode and curr_return >= 0.03:
                        locking_mode = True
                        max_price_in_mode = price 
                    
                    if locking_mode:
                        if price > max_price_in_mode: max_price_in_mode = price
                        
                        # 익절 (-2%)
                        if price <= max_price_in_mode * 0.98:
                            return_amt = shares * price * (1 - fee_sell)
                            balance += return_amt
                            profit_pct = (price - avg_price) / avg_price * 100
                            trades.append({'date': date, 'type': 'sell', 'price': price, 'score': score, 'profit': profit_pct, 'reason': '💎 Locking Trailing'})
                            shares = 0
                            buy_date = None
                            locking_mode = False
                            
                    else:
                        # 손절 방어 (45점 이하)
                        if score <= 45:
                            return_amt = shares * price * (1 - fee_sell)
                            balance += return_amt
                            profit_pct = (price - avg_price) / avg_price * 100
                            trades.append({'date': date, 'type': 'sell', 'price': price, 'score': score, 'profit': profit_pct, 'reason': 'Defense(45↓)'})
                            shares = 0
                            buy_date = None

        # 최종 평가금
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
    
    # 설정 UI (3단 컬럼)
    col_set1, col_set2, col_set3 = st.columns([1, 1, 1.5])
    
    with col_set1:
        bt_start_date = st.date_input("시작일", value=pd.to_datetime("2024-01-01"))
        
        # 🌟 타임 컷 설정 추가
        max_hold_days = st.slider(
            "⏱️ 최대 보유 기간 (Time Cut)", 
            min_value=0, 
            max_value=60, 
            value=0, 
            step=1,
            help="0일은 제한 없음. 예: 7일 선택 시, 매수 후 7일째 되는 날 무조건 매도합니다."
        )
        time_msg = "제한 없음 (Unlimited)" if max_hold_days == 0 else f"{max_hold_days}일 후 강제 청산"
        st.caption(f"설정: :red[{time_msg}]")

    with col_set2:
        selected_strategy = st.radio(
            "⚔️ 전략 선택", 
            ["기본 (Basic)", "슈퍼 락킹 (SuperLocking)"],
            captions=["매수 65↑ / 매도 45↓", "매수 80↑ / +3%후 고점대비 -2% 매도"]
        )
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
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    run_single_stock_backtest, 
                    code, 
                    name, 
                    str(bt_start_date), 
                    1000000, 
                    strat_code,
                    max_hold_days  # 🌟 추가된 파라미터 전달
                ): code for code, name in targets
            }
            
            # ... (이후 결과 처리 로직은 기존과 동일하므로 생략) ...
            completed = 0
            for future in futures:
                res = future.result()
                if res: results.append(res)
                completed += 1
                bar.progress(completed / total_stocks)
                progress_text.text(f"[{selected_strategy}] 분석 중... ({completed}/{total_stocks})")

        bar.empty()
        progress_text.empty()
        
        # 결과 출력 부분 (기존 코드 그대로 사용)
        if results:
            df_res = pd.DataFrame(results)
            avg_return = df_res['total_return'].mean()
            win_rate_avg = df_res['win_rate'].mean()
            total_profit_sum = df_res['final_equity'].sum() - (1000000 * len(df_res))
            
            st.success(f"✅ {selected_strategy} (TimeCut: {max_hold_days}일) 완료!")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("평균 수익률", f"{avg_return:.2f}%")
            m2.metric("평균 승률", f"{win_rate_avg:.1f}%")
            m3.metric("총 종목 수", f"{len(df_res)}개")
            m4.metric("총 수익금", f"{total_profit_sum:,.0f}원")
            
            st.divider()
            
            c_best, c_worst = st.columns(2)
            with c_best:
                st.subheader("🏆 수익률 Top 5")
                st.dataframe(df_res.sort_values('total_return', ascending=False).head(5)[['name', 'total_return', 'trade_count']], hide_index=True)
            
            with c_worst:
                st.subheader("💀 수익률 Worst 5")
                st.dataframe(df_res.sort_values('total_return', ascending=True).head(5)[['name', 'total_return', 'trade_count']], hide_index=True)

            st.markdown("#### 📄 상세 내역")
            st.dataframe(df_res[['name', 'total_return', 'win_rate', 'trade_count', 'final_equity']], use_container_width=True)
            
            fig = px.histogram(df_res, x="total_return", nbins=20, title="수익률 분포")
            fig.add_vline(x=avg_return, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
