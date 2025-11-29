import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ---------------------------------------------------------
# 1. 페이지 설정 및 디자인
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 주식 차트 분석기",
    page_icon="📈",
    layout="wide"
)

st.title("📈 내 손안의 AI 주식 분석 비서")
st.markdown("""
이 서비스는 기술적 지표(RSI, MACD, Bollinger Bands)를 기반으로 
**안전한 진입 시점**과 **보유 종목 대응 전략**을 분석해줍니다.
""")

# 사이드바: 종목 입력
st.sidebar.header("🔍 종목 검색")
ticker_input = st.sidebar.text_input("티커(Ticker) 입력", value="AAPL")
st.sidebar.info("""
**티커 예시:**
- 미국: AAPL (애플), TSLA (테슬라), NVDA (엔비디아)
- 한국(코스피): 005930.KS (삼성전자)
- 한국(코스닥): 035720.KQ (카카오)
""")

days_input = st.sidebar.slider("분석 기간 (일)", 100, 1000, 365)

# ---------------------------------------------------------
# 2. 데이터 로드 및 지표 계산 함수
# ---------------------------------------------------------
@st.cache_data
def load_data(ticker, days):
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=days)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        # yfinance 데이터 구조 평탄화 (MultiIndex 처리)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception as e:
        return None

def add_indicators(df):
    # 1. 이동평균선 (MA)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # 2. RSI (상대강도지수, 14일 기준)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD (이동평균 수렴확산)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bollinger Bands (볼린저 밴드)
    df['BB_Upper'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)

    return df

# ---------------------------------------------------------
# 3. 매매 전략 로직 (AI 판단)
# ---------------------------------------------------------
def analyze_market_status(df):
    """현재 차트 상태를 분석하여 신규 진입 여부를 판단"""
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['Signal_Line'].iloc[-1]
    bb_lower = df['BB_Lower'].iloc[-1]
    
    score = 0
    reasons = []

    # 전략 1: 과매도 구간 (RSI < 30) -> 반등 가능성 높음
    if rsi < 30:
        score += 3
        reasons.append("🟢 RSI가 30 미만(과매도)입니다. 기술적 반등 위치입니다.")
    elif rsi < 40:
        score += 1
        reasons.append("⚪ RSI가 낮아 가격 매력도가 있습니다.")
    elif rsi > 70:
        score -= 3
        reasons.append("🔴 RSI가 70 초과(과매수)입니다. 조정 가능성이 있습니다.")

    # 전략 2: 볼린저 밴드 하단 터치 -> 지지선 역할
    if current_price <= bb_lower * 1.02: # 하단 밴드 근처 2% 이내
        score += 2
        reasons.append("🟢 주가가 볼린저 밴드 하단 근처에 있어 지지받을 확률이 높습니다.")

    # 전략 3: MACD 골든크로스 (상승 추세 전환)
    if macd > macd_signal and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
        score += 2
        reasons.append("🟢 MACD가 시그널 선을 상향 돌파했습니다 (골든크로스). 추세 상승 신호입니다.")

    # 종합 판단
    if score >= 4:
        decision = "강력 매수 (Strong Buy)"
        color = "green"
    elif score >= 1:
        decision = "매수 관점 (Buy)"
        color = "blue"
    elif score <= -2:
        decision = "매도 관점 (Sell)"
        color = "red"
    else:
        decision = "관망 (Hold/Wait)"
        color = "gray"

    return decision, color, reasons

def analyze_my_position(df, my_price):
    """내 평단가와 비교하여 대응 전략 제시"""
    current_price = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    
    profit_rate = ((current_price - my_price) / my_price) * 100
    
    advice = ""
    status = ""
    
    # 수익 중일 때
    if profit_rate > 0:
        status = f"🎉 현재 **{profit_rate:.2f}% 수익** 중입니다."
        if rsi > 70:
            advice = "현재 RSI가 과매수 구간입니다. **분할 매도**를 통해 수익을 실현하는 것을 추천합니다."
        elif current_price < ma20:
            advice = "수익 중이나 주가가 20일 이동평균선을 하회했습니다. 추세가 꺾일 수 있으니 주의하세요."
        else:
            advice = "상승 추세가 유지되고 있습니다. **계속 보유(Hold)**하여 수익을 극대화하세요."
            
    # 손실 중일 때
    else:
        status = f"💧 현재 **{profit_rate:.2f}% 손실** 중입니다."
        if rsi < 30:
            advice = "현재 과매도 구간입니다. 지금 매도하기보다는 반등을 기다리거나, **물타기(추가 매수)**를 고려해볼 만한 시점입니다."
        elif current_price < df['BB_Lower'].iloc[-1]:
            advice = "주가가 바닥권(볼린저 하단)에 있습니다. 투매보다는 **반등 시 탈출**을 목표로 하세요."
        else:
            advice = "뚜렷한 반등 신호가 없습니다. 섣불리 추가 매수하기보다 바닥을 다지는지 확인하고 대응하세요 (관망)."
            
    return status, advice

# ---------------------------------------------------------
# 4. 메인 실행 로직
# ---------------------------------------------------------

data = load_data(ticker_input, days_input)

if data is not None:
    # 지표 계산
    df = add_indicators(data)
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📊 차트 분석 (신규 진입)", "💼 내 보유 종목 진단"])

    # === TAB 1: 차트 및 신규 진입 분석 ===
    with tab1:
        st.subheader(f"{ticker_input} 분석 결과")
        
        # 최신 데이터 표시
        latest_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        diff = latest_close - prev_close
        diff_pct = (diff / prev_close) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("현재 주가", f"{latest_close:.2f}", f"{diff:.2f} ({diff_pct:.2f}%)")
        col2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
        col3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")

        # AI 분석 결과
        decision, color, reasons = analyze_market_status(df)
        st.markdown(f"### 🤖 AI 판단: :{color}[{decision}]")
        for reason in reasons:
            st.write(reason)
            
        if not reasons:
            st.write("특별한 기술적 과열/침체 신호가 없습니다. 추세를 지켜보세요.")

        # 차트 그리기 (Candlestick + BB + MA)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # 캔들스틱
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'],
                                     name='OHLC'), row=1, col=1)
        # 이동평균선
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='BB Lower'), row=1, col=1)

        # 거래량 (또는 RSI)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
        # RSI 기준선 (30, 70)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: 보유 종목 진단 ===
    with tab2:
        st.subheader("내 평단가 기준 분석")
        st.write("이미 이 종목을 보유하고 계신가요? 평단가를 입력해보세요.")
        
        my_price = st.number_input("내 평균 단가 입력", value=float(df['Close'].iloc[-1]), step=1.0)
        
        if st.button("내 포지션 분석하기"):
            status_msg, advice_msg = analyze_my_position(df, my_price)
            
            st.markdown(f"### 진단 결과")
            st.info(status_msg)
            st.success(f"💡 **AI 조언:**\n\n{advice_msg}")
            
            st.caption("※ 주의: 이 결과는 기술적 지표에 기반한 참고용이며, 투자에 대한 책임은 본인에게 있습니다.")

else:
    st.error("데이터를 불러올 수 없습니다. 티커를 확인해주세요. (예: 삼성전자 -> 005930.KS)")
