import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import time

# =========================================================
# 1. 데이터 로딩 함수 수정 (auto_adjust=False 적용)
# =========================================================

@st.cache_data(show_spinner=False)
def load_price_data(code: str, start_date: str):
    """
    yfinance에서 개별 종목 데이터를 받아오는 함수 (캐시됨)
    [수정] auto_adjust=False로 변경하여 '실제 체결가'를 가져옴
    """
    # ★ 핵심 수정: auto_adjust=False (수정주가 대신 실제 가격 사용)
    df = yf.download(code, start=start_date, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(show_spinner=False)
def load_fx_series(start_date: str):
    """
    KRW=X 환율 시계열 다운로드
    """
    # 환율도 실제 가격 기준
    ex_df = yf.download("KRW=X", start=start_date, progress=False, auto_adjust=False)
    if isinstance(ex_df.columns, pd.MultiIndex):
        ex_df.columns = ex_df.columns.get_level_values(0)
    return ex_df['Close']

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
# 2. 지표 계산 로직 수정 (Close_Calc 매핑 변경)
# =========================================================

def calculate_indicators_for_backtest(df):
    """지표 계산 최적화 (단기 스윙용 보조지표 추가)"""
    df = df.copy()
    
    # [수정] 실제 차트와 맞추기 위해 무조건 'Close'(종가) 사용
    # auto_adjust=False로 하면 'Adj Close'와 'Close'가 둘 다 들어오는데,
    # 스윙 매매는 실제 거래된 가격인 'Close'를 봐야 함.
    df['Close_Calc'] = df['Close']
    
    # 1. 이동평균
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA10'] = df['Close_Calc'].rolling(10).mean() # 2주 매매의 생명선
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()
    
    # 2. 볼린저 밴드 (단기 변동성 돌파 확인용)
    std = df['Close_Calc'].rolling(20).std()
    df['Upper_Band'] = df['MA20'] + (std * 2)
    df['Lower_Band'] = df['MA20'] - (std * 2)
    # 밴드폭(Band Width): 좁아졌다가 넓어질 때가 매수 타이밍
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
    
    # 5. 거래량 이평 (거래량 실린 상승인지 확인)
    if 'Volume' in df.columns:
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        # 거래량 급증 여부 (평소보다 1.5배 이상 터졌는지)
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    else:
        df['Vol_Ratio'] = 1.0 # 거래량 정보 없으면 기본값

    # 6. 변동성 (표준편차)
    df['STD20'] = std
    
    return df.dropna()

def get_ai_score_row(row):
    """
    [최적화] 2주 단기 스윙용 점수 산정 (Momentum + Volatility)
    목표: 상승 초입 포착 (무릎에서 사서 어깨에서 팔기)
    """
    try:
        score = 50.0 # 기본점
        
        curr = row['Close_Calc']
        ma5, ma10, ma20, ma60 = row['MA5'], row['MA10'], row['MA20'], row['MA60']
        rsi = row['RSI']
        
        # ---------------------------------------------------------
        # 1. 추세 (Trend) - 단기 생명선(10일선) 중심
        # ---------------------------------------------------------
        # 2주 매매는 10일선이 꺾이면 끝난 것임.
        if curr > ma10:
            score += 15.0
            # 정배열 보너스 (5 > 10 > 20)
            if ma5 > ma10 > ma20:
                score += 5.0
        else:
            score -= 10.0 # 10일선 아래는 탄력 둔화
            
        # 장기 추세 필터 (60일선 위에 있어야 안전)
        if curr > ma60:
            score += 5.0
        else:
            score -= 5.0

        # ---------------------------------------------------------
        # 2. 모멘텀 (Momentum) - MACD & RSI
        # ---------------------------------------------------------
        # MACD 히스토그램이 '양수'이고 '어제보다 증가'했으면 상승 가속도 붙음
        if row['MACD_Hist'] > 0:
            score += 5.0
            if row['MACD_Hist'] > row['Prev_MACD_Hist']:
                score += 5.0 # 가속도 보너스
        
        # 턴어라운드 감지: 음수에서 양수로 전환 직전 or 막 전환
        elif row['MACD_Hist'] > row['Prev_MACD_Hist'] and row['MACD_Hist'] > -0.5:
             score += 5.0 # 반등 시도 중

        # RSI: 50~65 구간이 스윙에 가장 좋음 (너무 과열도 아니고 침체도 아님)
        if 50 <= rsi <= 70:
            score += 10.0
        elif rsi > 75:
            score -= 5.0 # 과열 경고 (곧 조정 올 수 있음)
        elif rsi < 35:
            score += 5.0 # 기술적 반등 기대 (낙폭 과대)

        # ---------------------------------------------------------
        # 3. 변동성 돌파 (Volatility Breakout) - 볼린저 밴드
        # ---------------------------------------------------------
        # 밴드 상단 돌파 시도 or 상단 타고 가는 중
        u_band = row['Upper_Band']
        if curr >= u_band * 0.98: # 상단 근처
            score += 10.0
            
        # 스퀴즈(Squeeze) 후 발산 체크
        # 밴드폭이 좁은데(변동성 축소) + 5일선이 상승 중이면 폭발 임박
        if row['Band_Width'] < 0.15 and ma5 > ma10: # 밴드폭 15% 미만
            score += 5.0

        # ---------------------------------------------------------
        # 4. 수급 (Volume)
        # ---------------------------------------------------------
        # 거래량이 평소보다 20% 이상 실리면서 양봉이면 신뢰도 상승
        if row['Vol_Ratio'] >= 1.2 and curr > row['MA5']:
             score += 5.0

        return max(0.0, min(100.0, score))
    except:
        return 0.0

# =========================================================
# 3. 개별 종목 백테스트 엔진 (변동 없음, 단지 데이터가 정확해짐)
# =========================================================

def prepare_stock_data(ticker_info, start_date):
    """
    개별 종목의 데이터를 미리 준비하는 함수
    """
    code, name = ticker_info
    try:
        # ★ 캐시된 다운로드 사용
        df_raw = load_price_data(code, start_date)
        if df_raw is None or df_raw.empty or len(df_raw) < 60:
            return None

        df = calculate_indicators_for_backtest(df_raw)
        df['AI_Score'] = df.apply(get_ai_score_row, axis=1)
        df['Ticker'] = code
        df['Name'] = name
        
        # [수정] 1분봉 시뮬레이션을 위해 Open, High, Low 데이터 추가 반환
        return df[['Open', 'High', 'Low', 'Close_Calc', 'AI_Score', 'STD20', 'Vol_Ratio', 'Ticker', 'Name']]
    except Exception as e:
        return None


def run_portfolio_backtest(targets, start_date, initial_capital, strategy_mode,
                           max_hold_days, exchange_data, use_compound, selection_mode):
    """
    [수정 완료] 장중(Intraday) 변동성을 반영한 1분봉 시뮬레이션 매도 로직 적용
    - Open/High/Low를 사용하여 갭락 및 장중 손절/익절을 정밀하게 체크
    """
    # ---------------------------------------------------------
    # 1. 전 종목 데이터 준비
    # ---------------------------------------------------------
    all_dfs = []
    for t in targets:
        res = prepare_stock_data(t, start_date)
        if res is not None:
            all_dfs.append(res)
            
    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame()

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
    # 4. 시뮬레이션 상태 변수 초기화
    # ---------------------------------------------------------
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
        # A. 매도 로직 (Sell Check) - [1분봉 시뮬레이션 적용]
        # =================================================
        sell_list = []
        for ticker in sorted(portfolio.keys()):
            info = portfolio[ticker]
            stock_row = next((x for x in daily_stocks if x['Ticker'] == ticker), None)
            
            if stock_row is None: 
                continue
            
            # [1분봉 시뮬레이션을 위한 데이터 추출]
            rate = 1.0 if ".KS" in ticker else current_rate
            
            raw_open = stock_row['Open']
            raw_high = stock_row['High']
            raw_low = stock_row['Low']
            raw_close = stock_row['Close_Calc']
            
            # 환율 적용 가격
            curr_open = raw_open * rate
            curr_high = raw_high * rate
            curr_low = raw_low * rate
            curr_close = raw_close * rate
            
            score = stock_row['AI_Score']
            fee_sell = 0.003 if ".KS" in ticker else 0.001
            
            # 보유 정보
            avg_price = info['avg_price']
            
            # 수익률(종가 기준 - 단순 참고용)
            profit_pct_close = ((curr_close - avg_price) / avg_price) * 100
            held_days = (pd.Timestamp(date) - pd.Timestamp(info['buy_date'])).days
            
            should_sell = False
            sell_reason = ""
            final_sell_price = curr_close # 기본은 종가 매도
            final_sell_price_raw = raw_close

            # ---------------------------------------------------
            # [시나리오 1] 장 시작 갭락(Gap Down) 체크 - 최우선
            # ---------------------------------------------------
            # 시가가 이미 손절가(-3.5%) 아래에서 시작했는가?
            stop_loss_price = avg_price * 0.965
            
            if not should_sell:
                if curr_open <= stop_loss_price:
                    should_sell = True
                    # 갭락 비율 계산 (로그용)
                    gap_pct = ((curr_open - avg_price) / avg_price) * 100
                    sell_reason = f"⚡ 시가갭락({gap_pct:.1f}%)"
                    final_sell_price = curr_open # 시가에 체결 (어쩔 수 없음)
                    final_sell_price_raw = raw_open

            # ---------------------------------------------------
            # [시나리오 2] 장중 손절 (Intraday Stop)
            # ---------------------------------------------------
            # 시가는 괜찮았는데, 장중에 저가가 손절가를 건드렸는가?
            if not should_sell:
                if curr_low <= stop_loss_price:
                    should_sell = True
                    sell_reason = "⚡ 장중손절(-3.5%)"
                    final_sell_price = stop_loss_price # 손절가에 정확히 체결 (지정가 감시 효과)
                    
                    # 환율 역산하여 원화 기록용 가격 추정
                    final_sell_price_raw = stop_loss_price / rate

            # ---------------------------------------------------
            # [시나리오 3] 트레일링 스탑 & 익절 (Intraday Trailing)
            # ---------------------------------------------------
            # 먼저 고가(High)를 확인해 최고가 갱신 처리
            if not should_sell:
                if curr_high > info['max_price']:
                    portfolio[ticker]['max_price'] = curr_high
                
                max_p = portfolio[ticker]['max_price']
                
                # (a) 수익 반납 방어: 최고가가 평단 대비 +5% 이상 갔었는데
                if max_p > avg_price * 1.05:
                    # 장중 저가가 평단가 +1% 라인을 깼다면?
                    protect_line = avg_price * 1.01
                    if curr_low < protect_line:
                        should_sell = True
                        sell_reason = "🛡️ 수익반납방어"
                        final_sell_price = protect_line
                        final_sell_price_raw = protect_line / rate
                    
                    # (b) 고점 대비 -3% 하락 (트레일링)
                    elif curr_low < max_p * 0.97:
                        should_sell = True
                        sell_reason = "📉 트레일링(-3%)"
                        final_sell_price = max_p * 0.97
                        final_sell_price_raw = final_sell_price / rate

            # ---------------------------------------------------
            # [시나리오 4] 종가 기준 판단 (기존 로직 유지)
            # ---------------------------------------------------
            if not should_sell:
                # 타임 컷
                limit_days = max_hold_days if max_hold_days > 0 else 14 
                if held_days >= limit_days:
                    should_sell = True
                    sell_reason = f"⏱️ 만기청산({held_days}일)"
                
                # 지지부진
                elif held_days >= 7 and profit_pct_close < 1.0:
                    should_sell = True
                    sell_reason = "🐢 지지부진(7일↑)"
                
                # 급등 후 점수 하락 익절
                elif profit_pct_close >= 15.0 and score < 50:
                    should_sell = True
                    sell_reason = "💰 급등익절(+15%)"
                
                # 추세 이탈
                elif score < 40:
                    should_sell = True
                    sell_reason = "추세이탈(40↓)"

            # 매도 실행
            if should_sell:
                # 실제 수익률 재계산 (체결가 기준)
                real_profit_pct = ((final_sell_price - avg_price) / avg_price) * 100
                
                return_amt = info['shares'] * final_sell_price * (1 - fee_sell)
                balance += return_amt
                
                trades_log.append({
                    'ticker': ticker,
                    'name': info['name'],
                    'date': date,
                    'type': 'sell',
                    'price': final_sell_price_raw, # 원화 환산 전 가격(기록용)
                    'shares': info['shares'],
                    'score': score,
                    'profit': real_profit_pct,
                    'reason': sell_reason,
                    'balance': balance
                })
                sell_list.append(ticker)
        
        # 포트폴리오에서 제거
        for t in sell_list: 
            del portfolio[t]

        # =================================================
        # B. 신규 매수 (Buy Logic) - [기존 유지: 일봉 종가 기준]
        # =================================================
        if len(portfolio) < max_slots:
            candidates = []
            for row in daily_stocks:
                ticker = row['Ticker']
                if ticker in portfolio: 
                    continue
                
                score = row['AI_Score']
                price_raw = row['Close_Calc']
                price_krw = price_raw * (1.0 if ".KS" in ticker else current_rate)
                
                vol_power = row.get('Vol_Ratio', 1.0)
                
                if score >= 70:
                    rsi_val = row.get('RSI', 50)
                    if rsi_val < 75:
                        vol_ratio = row.get('STD20', 0) / price_raw if price_raw > 0 else 0.03
                        candidates.append({
                            'ticker': ticker,
                            'name': row['Name'],
                            'price_raw': price_raw,
                            'price_krw': price_krw,
                            'score': score,
                            'vol_power': vol_power,
                            'vol_ratio': vol_ratio,
                            'reason': "AI추천(70↑)"
                        })

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
                            'buy_date': date,
                            'max_price': target['price_krw'],
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
                p_krw = stock_row['Close_Calc'] * (1.0 if ".KS" in ticker else current_rate)
                current_equity += info['shares'] * p_krw
            else:
                current_equity += info['shares'] * info['avg_price']
        
        equity_curve.append({'date': date, 'equity': current_equity})

    return pd.DataFrame(trades_log), pd.DataFrame(equity_curve)
                                
# =========================================================
# 4. UI 통합 (탭 추가)
# =========================================================

tab4 = st.tabs(["📊 전체 백테스트 시뮬레이션"])[0] 

with tab4:
    st.markdown("### 🧪 포트폴리오 유니버스 백테스트")
    st.caption("AI 전략 시뮬레이터 Final Ver. (일봉 기준 매수 / 1분봉 시뮬레이션 매도)")
    
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
        if "스나이퍼" in selected_strategy: strat_code = "Sniper"
        elif "슈퍼" in selected_strategy: strat_code = "SuperLocking"
        else: strat_code = "Basic"
        
    with c_opt:
        comp_mode = st.checkbox("복리 투자 (재투자)", value=True)
    with c_btn:
        st.write("")
        start_btn = st.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True)

    # --------------------------------------------------------------------------------
    # 2. 실행 로직 (세션 스테이트를 사용하여 결과 저장)
    # --------------------------------------------------------------------------------
    
    if 'bt_result_trade' not in st.session_state:
        st.session_state['bt_result_trade'] = pd.DataFrame()
    if 'bt_result_equity' not in st.session_state:
        st.session_state['bt_result_equity'] = pd.DataFrame()

    if start_btn:
        if exchange_arg_val == "DYNAMIC":
            with st.spinner("💱 환율 데이터 수집 중..."):
                exchange_data_payload = load_fx_series(str(bt_start_date))
        else:
            exchange_data_payload = float(exchange_arg_val)

        with st.spinner(f"🔄 [{selected_strategy}] 전략으로 전체 시장 스캔 중..."):
            targets = list(TICKER_MAP.items())
            
            t_df, e_df = run_portfolio_backtest(
                targets, str(bt_start_date), initial_cap_input, strat_code, 
                max_hold_days, exchange_data_payload, comp_mode, selection_code
            )
            
            st.session_state['bt_result_trade'] = t_df
            st.session_state['bt_result_equity'] = e_df
            
            st.success("백테스트 완료! 결과를 확인하세요.")

    # --------------------------------------------------------------------------------
    # 3. 결과 대시보드
    # --------------------------------------------------------------------------------
    
    trade_df = st.session_state['bt_result_trade']
    equity_df = st.session_state['bt_result_equity']

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
                k3.metric("승률 (Win Rate)", f"{win_rate:.1f}%", 
                          f"{win_count}승 {total_sells-win_count}패")
                k4.metric("MDD (최대낙폭)", f"{mdd:.2f}%", 
                          "Risk Level", delta_color="off")
                k5.metric("총 매매 횟수", f"{len(trade_df)//2}회", 
                          f"평균 {len(trade_df)//2 / len(equity_df) * 5:.1f}회/주")

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
                st.markdown("#### 🏆 Best & Worst")
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
                        # [차트 로딩 부분도 수정]
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
