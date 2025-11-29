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
    전체 종목을 날짜별로 순회하며 포트폴리오를 운용하는 시뮬레이션
    """
    # 1. 전 종목 데이터 병렬 준비
    all_dfs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(prepare_stock_data, t, start_date): t for t in targets}
        for future in futures:
            res = future.result()
            if res is not None: all_dfs.append(res)
            
    if not all_dfs: return pd.DataFrame(), pd.DataFrame() # 빈 데이터프레임 반환

    # 2. 데이터를 날짜 기준으로 통합 (Market Data)
    market_data = {}
    for df in all_dfs:
        for date, row in df.iterrows():
            if date not in market_data: market_data[date] = []
            market_data[date].append(row)
            
    sorted_dates = sorted(market_data.keys())
    
    # 3. 환율 데이터 준비
    exchange_map = {}
    if isinstance(exchange_data, (float, int)):
        get_rate = lambda d: float(exchange_data)
    else:
        rate_dict = exchange_data.to_dict()
        def get_rate(d):
            ts = pd.Timestamp(d)
            # 환율 데이터가 없으면 전날 데이터 또는 기본값 사용
            return rate_dict.get(ts, 1430.0)

    # 4. 시뮬레이션 상태 변수
    balance = initial_capital
    portfolio = {} 
    trades_log = []
    equity_curve = []
    
    max_slots = 1 if selection_mode == 'TOP1' else 10 

    # --- 날짜별 루프 (Time Loop) ---
    for date in sorted_dates:
        daily_stocks = market_data[date]
        current_rate = get_rate(date)
        
        # A. 보유 종목 관리 (매도 체크)
        sell_list = []
        
        for ticker, info in portfolio.items():
            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            
            # 🔴 [수정] 여기가 에러 원인이었습니다. (is not None 추가)
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
                if strategy_mode == "Basic":
                    if score <= 45:
                        should_sell = True
                        sell_reason = "AI 45↓"
                        
                elif strategy_mode == "SuperLocking":
                    if not info['mode_active'] and profit_ratio >= 0.03:
                        portfolio[ticker]['mode_active'] = True
                        portfolio[ticker]['max_price'] = curr_price_krw
                    
                    if info['mode_active']:
                        if curr_price_krw > portfolio[ticker]['max_price']:
                            portfolio[ticker]['max_price'] = curr_price_krw
                        
                        if curr_price_krw <= portfolio[ticker]['max_price'] * 0.98:
                            should_sell = True
                            sell_reason = "💎 Locking Trailing"
                    else:
                        if score <= 45:
                            should_sell = True
                            sell_reason = "Defense(45↓)"

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
                if strategy_mode == "Basic" and score >= 65:
                    entry_signal = True
                    reason = "AI 65↑"
                elif strategy_mode == "SuperLocking" and score >= 80:
                    entry_signal = True
                    reason = "Strong Buy(80↑)"
                
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
                    
                    # 0으로 나누기 방지
                    if target['price_krw'] > 0:
                        shares = int(budget / (target['price_krw'] * (1 + fee_buy)))
                    else:
                        shares = 0
                    
                    if shares > 0:
                        cost = shares * target['price_krw'] * (1 + fee_buy)
                        balance -= cost
                        
                        portfolio[target['ticker']] = {
                            'name': target['name'],
                            'shares': shares,
                            'avg_price': target['price_krw'],
                            'buy_date': date,
                            'mode_active': False, 
                            'max_price': 0       
                        }
                        
                        trades_log.append({
                            'ticker': target['ticker'], 'name': target['name'], 'date': date, 
                            'type': 'buy', 'price': target['price_raw'], 'score': target['score'], 
                            'profit': 0, 'reason': target['reason'], 'balance': balance
                        })

        # C. 자산 평가 (Equity Curve)
        current_equity = balance
        for ticker, info in portfolio.items():
            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            
            # 🔴 [수정] 여기도 에러 원인이 될 수 있으므로 is not None 추가
            if stock_row is not None:
                p_raw = stock_row['Close_Calc']
                p_krw = p_raw * (1.0 if ".KS" in ticker else current_rate)
                current_equity += info['shares'] * p_krw
            else:
                # 오늘 데이터가 없으면(휴장 등) 어제 가격(평단가 등)으로 임시 평가
                current_equity += info['shares'] * info['avg_price'] # 혹은 직전 가격 유지
                
        equity_curve.append({'date': date, 'equity': current_equity})

    return pd.DataFrame(trades_log), pd.DataFrame(equity_curve)
# =========================================================
# 3. UI 통합 (탭 추가)
# =========================================================
# (기존 코드의 tab1, tab2, tab3 정의 아래에 tab4를 추가한다고 가정)

tab4 = st.tabs(["📊 전체 백테스트 시뮬레이션"])[0] # 기존 tabs 리스트에 추가 필요

with tab4:
    st.markdown("### 🧪 포트폴리오 유니버스 백테스트")
    st.caption("전체 시장을 대상으로 날짜별 시뮬레이션을 수행합니다. (환율/복리/집중투자 반영)")
    
    # --------------------------------------------------------------------------------
    # 1. 설정 UI (3단 컬럼 구성)
    # --------------------------------------------------------------------------------
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    
    with r1_c1:
        bt_start_date = st.date_input("시작일", value=pd.to_datetime("2024-01-01"))
        max_hold_days = st.slider("⏱️ 타임 컷 (일)", 0, 60, 0, help="0: 제한 없음. 설정 시 N일 후 강제 매도")

    with r1_c2:
        initial_cap_input = st.number_input("💰 초기 자본금 (원)", value=10000000, step=1000000, format="%d")
        
        # 🌟 투자 스타일 선택 (All vs Top1)
        sel_mode = st.selectbox(
            "🎯 종목 선정 방식", 
            ["조건 만족 전부 매수 (분산)", "점수 1등만 매수 (집중)"],
            help="분산: 최대 10종목까지 자금을 쪼개서 매수\n집중: 가장 점수 높은 1개 종목에 자금 올인"
        )
        selection_code = "TOP1" if "집중" in sel_mode else "ALL"

    with r1_c3:
        # ⚔️ 전략 선택
        selected_strategy = st.radio("⚔️ 매매 전략", ["기본 (65/45)", "슈퍼 락킹 (80/Trailing)"])
        strat_code = "Basic" if "기본" in selected_strategy else "SuperLocking"
        
        # 옵션 체크박스
        comp_mode = st.checkbox("복리 투자 (수익 재투자)", value=True)
        ex_mode = st.checkbox("실시간 환율 적용 (Dynamic)", value=True)
    
    # 실행 버튼
    st.write("")
    start_btn = st.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True)

    # --------------------------------------------------------------------------------
    # 2. 시뮬레이션 로직 실행
    # --------------------------------------------------------------------------------
    if start_btn:
        progress_text = st.empty()
        
        # A. 환율 데이터 준비
        exchange_data_payload = 1430.0 # 기본값
        if ex_mode:
            with st.spinner("💱 과거 환율 데이터(KRW=X) 수집 중..."):
                try:
                    ex_df = yf.download("KRW=X", start=str(bt_start_date), progress=False)
                    if isinstance(ex_df.columns, pd.MultiIndex):
                        ex_df.columns = ex_df.columns.get_level_values(0)
                    exchange_data_payload = ex_df['Close']
                    st.success("환율 데이터 적용 완료")
                except: 
                    st.warning("환율 데이터 수집 실패. 고정 환율(1,430원)을 사용합니다.")

        # B. 포트폴리오 시뮬레이션 실행 (run_portfolio_backtest 함수 호출)
        # *주의: run_portfolio_backtest 함수가 코드 상단에 정의되어 있어야 합니다.
        with st.spinner("🔄 전 종목 스캔 및 타임머신 가동 중... (약 15~30초)"):
            targets = list(TICKER_MAP.items())
            
            trade_df, equity_df = run_portfolio_backtest(
                targets, 
                str(bt_start_date), 
                initial_cap_input, 
                strat_code, 
                max_hold_days, 
                exchange_data_payload, 
                comp_mode, 
                selection_code # ALL or TOP1 전달
            )
        
        # --------------------------------------------------------------------------------
        # 3. 결과 시각화
        # --------------------------------------------------------------------------------
        if not trade_df.empty and not equity_df.empty:
            
            # (1) 핵심 지표 계산
            final_equity = equity_df.iloc[-1]['equity']
            total_return = (final_equity - initial_cap_input) / initial_cap_input * 100
            profit_amt = final_equity - initial_cap_input
            
            # 승률 계산 (매도 거래 기준)
            sells = trade_df[trade_df['type'] == 'sell']
            win_count = len(sells[sells['profit'] > 0])
            total_sells = len(sells)
            win_rate = (win_count / total_sells * 100) if total_sells > 0 else 0.0
            
            st.success(f"✅ 시뮬레이션 완료! | 방식: {sel_mode}")
            
            # (2) KPI 대시보드
            with st.container():
                k1, k2, k3, k4 = st.columns(4)
                
                k1.metric(
                    "총 수익률 (Total Return)", 
                    f"{total_return:,.2f}%", 
                    help=f"{bt_start_date}부터 현재까지의 누적 수익률"
                )
                k2.metric(
                    "매매 승률 (Win Rate)", 
                    f"{win_rate:.1f}%", 
                    f"{win_count}승 / {total_sells}전"
                )
                
                # 금액 단위 포맷팅 (억/만)
                if abs(profit_amt) >= 100000000:
                    amt_str = f"{profit_amt/100000000:,.2f}억 원"
                else:
                    amt_str = f"{profit_amt/10000:,.0f}만 원"
                
                k3.metric("총 수익금", amt_str, delta_color="normal")
                k4.metric("총 매매 횟수", f"{len(trade_df)//2}회") # 매수+매도=1회

            st.divider()

            # (3) 자산 곡선 (Equity Curve)
            st.subheader("📈 내 계좌 자산 변화 (Equity Curve)")
            
            fig = px.line(
                equity_df, 
                x='date', 
                y='equity', 
                title=f"자산 성장 그래프 ({sel_mode})",
                labels={'equity': '평가 금액(원)', 'date': '날짜'}
            )
            # 원금 라인 표시
            fig.add_hline(y=initial_cap_input, line_dash="dash", line_color="gray", annotation_text="원금")
            
            # 영역 채우기 (시각적 효과)
            fig.update_traces(fill='tozeroy', line=dict(color='#00CC96', width=2))
            fig.update_layout(yaxis_tickformat=',d') # Y축 콤마 포맷
            
            st.plotly_chart(fig, use_container_width=True)

            # (4) 상세 거래 일지 (Trade Log)
            st.subheader("📝 상세 거래 일지")
            
            # 보기 좋게 가공
            display_log = trade_df.copy()
            # 날짜 포맷
            display_log['date'] = display_log['date'].dt.date
            # 필요한 컬럼만 선택
            display_log = display_log[['date', 'name', 'type', 'price', 'profit', 'balance', 'reason']]
            
            # 데이터프레임 출력
            st.dataframe(
                display_log.sort_values('date', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "날짜",
                    "name": st.column_config.TextColumn("종목명", width="medium"),
                    "type": st.column_config.TextColumn("구분", width="small"),
                    "price": st.column_config.NumberColumn("체결가($)", format="%.2f"),
                    "profit": st.column_config.NumberColumn("수익률", format="%.2f%%"),
                    "balance": st.column_config.NumberColumn("거래 후 잔고", format="%d원"),
                    "reason": st.column_config.TextColumn("사유", width="large"),
                },
                height=500
            )
            
            # (5) 종목별 성과 요약 (집계)
            st.subheader("📊 종목별 실현 손익 집계")
            if not sells.empty:
                # 종목별로 그룹화하여 수익금 합계 계산
                # (매도 기록을 기준으로 계산)
                # 주의: 단순히 profit %를 더하는 건 부정확할 수 있으나, 대략적인 흐름 파악용
                
                # 정확한 종목별 손익금 계산을 위해 trade_df 재가공 필요하나, 
                # 여기서는 매도 리스트를 기반으로 표시
                stock_summary = sells.groupby('name').agg(
                    total_profit_pct=('profit', 'sum'),
                    trade_count=('profit', 'count')
                ).reset_index().sort_values('total_profit_pct', ascending=False)
                
                st.dataframe(
                    stock_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "name": "종목명",
                        "total_profit_pct": st.column_config.NumberColumn("누적 수익률 합계", format="%.2f%%"),
                        "trade_count": st.column_config.NumberColumn("매도 횟수", format="%d회"),
                    }
                )

        else:
            st.warning("⚠️ 매매 신호가 발생하지 않았습니다. (조건을 완화하거나 기간을 늘려보세요)")
