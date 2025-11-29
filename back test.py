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
def run_single_stock_backtest(ticker, name, start_date="2023-01-01", initial_capital=1000000, strategy_mode="Basic", max_holding_days=0, exchange_data=1430.0):
    """
    exchange_data: 
      - float/int일 경우: 고정 환율 적용 (예: 1430)
      - pd.Series일 경우: 날짜별 환율 데이터 (Index가 Datetime)
    """
    try:
        # 1. 주가 데이터 수집
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) < 60: return None

        # 2. 지표 계산
        df = calculate_indicators_for_backtest(df)
        df['AI_Score'] = df.apply(get_ai_score_row, axis=1)

        # -----------------------------------------------------------
        # 💱 [핵심] 환율 데이터 병합 (Merge)
        # -----------------------------------------------------------
        is_kr = ".KS" in ticker or ".KQ" in ticker
        
        if is_kr:
            # 한국 주식은 환율 1.0 고정
            df['Exchange_Rate'] = 1.0
        else:
            # 미국 주식
            if isinstance(exchange_data, (float, int)):
                # A. 고정 환율 모드
                df['Exchange_Rate'] = float(exchange_data)
            else:
                # B. 변동 환율 모드 (과거 데이터 매핑)
                # 인덱스(날짜)를 기준으로 환율 데이터를 합칩니다.
                # 휴장일 등으로 환율 데이터가 비어있으면 전날 환율(ffill)을 사용합니다.
                df['Exchange_Rate'] = exchange_data.reindex(df.index, method='ffill').fillna(method='bfill')
                
                # 혹시라도 NaN이 남으면 기본값 1400원으로 채움 (안전장치)
                df['Exchange_Rate'] = df['Exchange_Rate'].fillna(1400.0)

        # -----------------------------------------------------------

        # 시뮬레이션 변수
        balance = initial_capital
        shares = 0
        avg_price = 0
        trades = []
        buy_date = None
        
        # 슈퍼 락킹 변수
        locking_mode = False
        max_price_in_mode = 0
        
        # 수수료
        fee_buy = 0.00015 if is_kr else 0.001
        fee_sell = 0.003 if is_kr else 0.001

        for date, row in df.iterrows():
            # 🌟 그 날의 환율이 반영된 가격 계산
            rate = row['Exchange_Rate']
            raw_price = row['Close_Calc']     # 달러(또는 원화)
            price_krw = raw_price * rate      # 원화 환산 가격
            
            score = row['AI_Score']
            
            # --- 타임 컷 (Time Cut) ---
            if shares > 0 and max_holding_days > 0 and buy_date is not None:
                days_held = (date - buy_date).days
                if days_held >= max_holding_days:
                    return_amt = shares * price_krw * (1 - fee_sell)
                    balance += return_amt
                    profit_pct = (price_krw - avg_price) / avg_price * 100
                    trades.append({
                        'date': date, 'type': 'sell', 'price': raw_price, 
                        'score': score, 'profit': profit_pct, 'reason': f'⏱️ TimeCut({days_held}일)',
                        'rate': rate # 환율 기록
                    })
                    shares = 0
                    buy_date = None
                    locking_mode = False
                    continue

            # --- [전략 1] 기본 (Basic) ---
            if strategy_mode == "Basic":
                # 매수
                if score >= 65 and shares == 0:
                    can_buy = int(balance / (price_krw * (1 + fee_buy)))
                    if can_buy > 0:
                        shares = can_buy
                        balance -= shares * price_krw * (1 + fee_buy)
                        avg_price = price_krw
                        buy_date = date
                        trades.append({'date': date, 'type': 'buy', 'price': raw_price, 'score': score, 'reason': 'AI 65↑', 'rate': rate})

                # 매도
                elif score <= 45 and shares > 0:
                    return_amt = shares * price_krw * (1 - fee_sell)
                    balance += return_amt
                    profit_pct = (price_krw - avg_price) / avg_price * 100
                    trades.append({'date': date, 'type': 'sell', 'price': raw_price, 'score': score, 'profit': profit_pct, 'reason': 'AI 45↓', 'rate': rate})
                    shares = 0
                    buy_date = None

            # --- [전략 2] 슈퍼 락킹 (SuperLocking) ---
            elif strategy_mode == "SuperLocking":
                # 매수
                if score >= 80 and shares == 0:
                    can_buy = int(balance / (price_krw * (1 + fee_buy)))
                    if can_buy > 0:
                        shares = can_buy
                        balance -= shares * price_krw * (1 + fee_buy)
                        avg_price = price_krw
                        buy_date = date
                        locking_mode = False
                        max_price_in_mode = 0
                        trades.append({'date': date, 'type': 'buy', 'price': raw_price, 'score': score, 'reason': 'Strong Buy(80↑)', 'rate': rate})
                
                # 보유 관리
                elif shares > 0:
                    curr_return = (price_krw - avg_price) / avg_price
                    
                    if not locking_mode and curr_return >= 0.03:
                        locking_mode = True
                        max_price_in_mode = price_krw
                    
                    if locking_mode:
                        if price_krw > max_price_in_mode: max_price_in_mode = price_krw
                        
                        if price_krw <= max_price_in_mode * 0.98:
                            return_amt = shares * price_krw * (1 - fee_sell)
                            balance += return_amt
                            profit_pct = (price_krw - avg_price) / avg_price * 100
                            trades.append({'date': date, 'type': 'sell', 'price': raw_price, 'score': score, 'profit': profit_pct, 'reason': '💎 Locking Trailing', 'rate': rate})
                            shares = 0
                            buy_date = None
                            locking_mode = False
                    else:
                        if score <= 45:
                            return_amt = shares * price_krw * (1 - fee_sell)
                            balance += return_amt
                            profit_pct = (price_krw - avg_price) / avg_price * 100
                            trades.append({'date': date, 'type': 'sell', 'price': raw_price, 'score': score, 'profit': profit_pct, 'reason': 'Defense(45↓)', 'rate': rate})
                            shares = 0
                            buy_date = None

        # 최종 평가 (마지막 날 환율 적용)
        final_row = df.iloc[-1]
        final_price_krw = final_row['Close_Calc'] * final_row['Exchange_Rate']
        
        final_equity = balance + (shares * final_price_krw)
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
    
    # 설정 UI (4단 컬럼 구성)
    col_set1, col_set2, col_set3, col_set4 = st.columns([1.2, 1.2, 1.2, 1.2])
    
    with col_set1:
        bt_start_date = st.date_input("시작일", value=pd.to_datetime("2024-01-01"))
        max_hold_days = st.slider("⏱️ 타임 컷 (일)", 0, 60, 0, help="0: 제한 없음")

    with col_set2:
        # 🌟 환율 설정
        ex_mode = st.radio("💱 환율 적용 방식", ["고정 환율 (Fixed)", "실시간 변동 (Dynamic)"])
        
        if "고정" in ex_mode:
            fixed_rate_val = st.number_input("적용 환율(원/$)", value=1430.0, step=10.0)
            exchange_arg = fixed_rate_val
        else:
            st.caption("📅 매수/매도일 당시 환율을 적용합니다.")
            exchange_arg = "DYNAMIC" # 플래그

    with col_set3:
        selected_strategy = st.radio(
            "⚔️ 전략 선택", 
            ["기본 (Basic)", "슈퍼 락킹 (SuperLocking)"],
            captions=["65↑ 매수 / 45↓ 매도", "80↑ 매수 / +3% 후 익절"]
        )
        strat_code = "Basic" if "기본" in selected_strategy else "SuperLocking"
        
    with col_set4:
        st.write("")
        st.write("")
        start_btn = st.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True)

    if start_btn:
        results = []
        progress_text = st.empty()
        bar = st.progress(0)
        
        # 1. 변동 환율 모드일 경우, 환율 데이터 먼저 다운로드 (한 번만!)
        exchange_data_payload = exchange_arg
        if exchange_arg == "DYNAMIC":
            with st.spinner("💱 과거 환율 데이터(KRW=X) 수집 중..."):
                try:
                    # 시작일보다 조금 더 여유있게 가져옴
                    ex_df = yf.download("KRW=X", start=str(bt_start_date), progress=False)
                    if isinstance(ex_df.columns, pd.MultiIndex):
                        ex_df.columns = ex_df.columns.get_level_values(0)
                    exchange_data_payload = ex_df['Close'] # Series 전달
                    st.success(f"환율 데이터 로드 완료 ({len(exchange_data_payload)}일)")
                except Exception as e:
                    st.error(f"환율 데이터 수집 실패: {e}")
                    st.stop()
        
        # 2. 병렬 시뮬레이션 시작
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
                    max_hold_days,
                    exchange_data_payload # 🌟 환율 데이터(값 또는 Series) 전달
                ): code for code, name in targets
            }
            
            completed = 0
            for future in futures:
                res = future.result()
                if res: results.append(res)
                completed += 1
                bar.progress(completed / total_stocks)
                progress_text.text(f"분석 중... ({completed}/{total_stocks})")

        bar.empty()
        progress_text.empty()
        
if results:
            df_res = pd.DataFrame(results)
            
            # ---------------------------------------------------------
            # 1. 데이터 가공 및 통계 계산
            # ---------------------------------------------------------
            # 수익률의 평균은 '일간'이 아니라, 시뮬레이션 '전체 기간' 동안의 평균입니다.
            avg_return = df_res['total_return'].mean()
            win_rate_avg = df_res['win_rate'].mean()
            
            # 초기 자본 총액 (종목 수 * 100만원) 대비 최종 자산 총액
            initial_total_capital = 1000000 * len(df_res)
            final_total_equity = df_res['final_equity'].sum()
            total_profit_amt = final_total_equity - initial_total_capital
            total_profit_pct = (total_profit_amt / initial_total_capital) * 100
            
            st.success(f"✅ 분석 완료! ({bt_start_date} ~ 현재) | 전략: {selected_strategy}")
            
            # ---------------------------------------------------------
            # 2. 메인 대시보드 (KPI 카드)
            # ---------------------------------------------------------
            st.markdown("### 📊 포트폴리오 성과 요약")
            
            # 스타일링된 컨테이너 사용
            with st.container():
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                # 기간 수익률 (전체 자산 기준)
                kpi1.metric(
                    label="총 누적 수익률 (기간)",
                    value=f"{total_profit_pct:,.2f}%",
                    delta=f"{avg_return:,.2f}% (종목 평균)",
                    help="설정한 기간 동안 전체 계좌가 얼마나 불어났는지를 의미합니다."
                )
                
                # 평균 승률
                kpi2.metric(
                    label="평균 승률",
                    value=f"{win_rate_avg:.1f}%",
                    help="익절로 끝난 매매의 비율입니다."
                )
                
                # 총 수익금
                kpi3.metric(
                    label="총 예상 수익금",
                    value=f"{total_profit_amt/10000:,.0f}만 원", # 만원 단위로 축약
                    delta_color="normal",
                    help="종목당 100만 원 투자 시 예상되는 총 수익금입니다."
                )
                
                # 종목 수
                kpi4.metric(
                    label="분석 종목 수",
                    value=f"{len(df_res)}개",
                    help="백테스트에 포함된 총 종목 개수입니다."
                )

            st.divider()

            # ---------------------------------------------------------
            # 3. 차트 섹션 (좌: 수익률 분포 / 우: Top & Worst)
            # ---------------------------------------------------------
            col_chart, col_list = st.columns([1.5, 1])
            
            with col_chart:
                st.markdown("#### 📈 수익률 분포 (Histogram)")
                # Plotly 디자인 개선
                fig = px.histogram(
                    df_res, 
                    x="total_return", 
                    nbins=25,
                    color_discrete_sequence=['#4C78A8']
                )
                fig.update_layout(
                    xaxis_title="기간 수익률 (%)",
                    yaxis_title="종목 개수",
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", # 투명 배경
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                # 평균선 강조
                fig.add_vline(x=avg_return, line_dash="dash", line_color="#FF4B4B", annotation_text="평균")
                st.plotly_chart(fig, use_container_width=True)

            with col_list:
                st.markdown("#### 🏆 수익률 Best 3")
                top3 = df_res.sort_values('total_return', ascending=False).head(3)
                
                # 미니 데이터프레임 (깔끔하게)
                st.dataframe(
                    top3[['name', 'total_return']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "name": "종목명",
                        "total_return": st.column_config.NumberColumn("수익률", format="%.2f%%")
                    }
                )
                
                st.markdown("#### 💀 수익률 Worst 3")
                worst3 = df_res.sort_values('total_return', ascending=True).head(3)
                st.dataframe(
                    worst3[['name', 'total_return']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "name": "종목명",
                        "total_return": st.column_config.NumberColumn("수익률", format="%.2f%%")
                    }
                )

            # ---------------------------------------------------------
            # 4. 전체 상세 내역 (비주얼 업그레이드)
            # ---------------------------------------------------------
            st.markdown("#### 📑 종목별 상세 리포트")
            
            # 데이터프레임 컬럼 설정 (핵심 디자인)
            column_configuration = {
                "name": st.column_config.TextColumn("종목명", width="medium"),
                
                # 수익률: 숫자가 클수록 진하게 표시되는 히트맵 효과는 없지만, 깔끔하게 포맷팅
                "total_return": st.column_config.NumberColumn(
                    "기간 수익률",
                    help="해당 기간 동안의 총 수익률",
                    format="%.2f%%"
                ),
                
                # 승률: 0~100% 진행바(Bar)로 표시 -> 엑셀 느낌 탈피!
                "win_rate": st.column_config.ProgressColumn(
                    "승률 (Win Rate)",
                    help="매매 승률",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                
                # 매매 횟수
                "trade_count": st.column_config.NumberColumn(
                    "매매 횟수",
                    format="%d회"
                ),
                
                # 최종 자산
                "final_equity": st.column_config.NumberColumn(
                    "최종 평가금",
                    help="100만 원 투자 시 최종 금액",
                    format="%d원"
                )
            }
            
            st.dataframe(
                df_res[['name', 'total_return', 'win_rate', 'trade_count', 'final_equity']].sort_values('total_return', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config=column_configuration,
                height=500 # 높이 고정으로 스크롤 편의성 제공
            )
            
        else:
            st.error("결과가 없습니다. 날짜를 변경하거나 데이터를 확인해주세요.")
