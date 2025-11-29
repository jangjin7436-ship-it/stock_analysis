import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import json

# ---------------------------------------------------------
# 0. 파이어베이스(DB) 설정 (서버 저장용)
# ---------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, firestore

# DB 연결 함수
def get_db():
    # 이미 초기화되었는지 확인
    if not firebase_admin._apps:
        try:
            if 'firebase_key' in st.secrets:
                # secrets에서 값을 가져옴
                secret_val = st.secrets['firebase_key']
                
                # 상황 1: 이미 딕셔너리(AttrDict) 형태로 들어온 경우 (TOML 포맷)
                if isinstance(secret_val, dict) or str(type(secret_val)) == "<class 'streamlit.runtime.secrets.AttrDict'>":
                    key_dict = dict(secret_val)
                # 상황 2: 문자열(JSON String) 형태로 들어온 경우
                else:
                    key_dict = json.loads(secret_val)
                
                # private_key 줄바꿈 문자(\n) 처리 (가끔 문자열로 들어올 때 깨지는 경우 대비)
                if 'private_key' in key_dict:
                    key_dict['private_key'] = key_dict['private_key'].replace('\\n', '\n')

                cred = credentials.Certificate(key_dict)
                firebase_admin.initialize_app(cred)
            else:
                return None
        except Exception as e:
            st.warning(f"DB 연결 실패: {e}")
            return None
    return firestore.client()

# ---------------------------------------------------------
# 1. 페이지 설정 및 사용자 리스트
# ---------------------------------------------------------
st.set_page_config(page_title="AI 주식 스캐너 & 포트폴리오", page_icon="📈", layout="wide")

# 사용자가 요청한 감시 종목 리스트
USER_WATCHLIST = [
    "INTC", "005290", "SOXL", "316140", "WDC", "NFLX", "000990", "KLAC", "009540", "006360", 
    "024110", "042660", "105560", "BAC", "NEM", "FCX", "272210", "240810", "005930", "010140", 
    "006400", "267250", "028260", "SLV", "079550", "039030", "C", "009830", "LLY", "128940", 
    "WFC", "012450", "ASML", "NVDA", "GE", "V", "XLE", "005935", "041510", "BA", "000660", 
    "000810", "000250", "TXN", "122990", "GM", "302440", "F", "DELL", "JNJ", "263750", "012330",
    "QCOM", "XOM", "AVGO", "OXY", "SLB", "086790", "TQQQ", "UPRO", "FNGU", "BULZ", "TMF", 
    "TSLA", "AMD", "BITX", "TSLL"
]

# 한국 주식 코드 변환 헬퍼 (숫자만 있으면 .KS 붙임)
def format_ticker(ticker):
    ticker = ticker.strip().upper()
    if ticker.isdigit():
        return f"{ticker}.KS"
    return ticker

# ---------------------------------------------------------
# 2. 데이터 로드 및 분석 로직
# ---------------------------------------------------------
@st.cache_data(ttl=3600)  # 1시간 캐시
def get_bulk_data(tickers_list):
    """여러 종목 데이터를 한 번에 다운로드"""
    formatted_tickers = [format_ticker(t) for t in tickers_list]
    # 수정: 기간을 '2y'(2년)로 대폭 늘려 충분한 과거 데이터를 확보합니다.
    # 이렇게 하면 MA60, MA200 등을 계산할 때 앞부분이 잘려도 최신 데이터는 안전합니다.
    data = yf.download(formatted_tickers, period="2y", group_by='ticker', threads=True)
    return data, formatted_tickers

def calculate_indicators(df):
    """단일 종목 DataFrame에 지표 추가"""
    # 데이터가 너무 적으면 계산 불가 (최소 60일은 있어야 MA60 계산 가능)
    if len(df) < 60: return None
    
    df = df.copy()
    
    # 0. 데이터 정제 (빈 데이터가 섞여있을 경우 앞뒤로 채움)
    df['Close'] = df['Close'].ffill()

    # 1. 이평선
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 중요: 지표 계산 후 NaN(계산 불가능한 앞부분) 제거
    # 2년치를 가져왔으므로 앞부분 60일을 잘라내도 최근 1년 반 데이터가 남습니다.
    df = df.dropna()
    
    return df

def analyze_strategy(df):
    """스윙 전략 분석 (매수/매도/관망)"""
    # 데이터가 비어있거나 마지막 날짜 데이터가 없으면 예외 처리
    if df is None or df.empty: return "데이터 로딩 중...", "gray", 0
    
    current_price = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['Signal_Line'].iloc[-1]

    score = 0
    reasons = []

    # 1. 추세 (20일선 위)
    if current_price > ma20:
        score += 1
        if current_price > ma60: score += 1
    else:
        score -= 2 # 추세 이탈

    # 2. 눌림목 (20일선 근접 지지)
    if current_price > ma20 and current_price <= ma20 * 1.03:
        score += 3
    
    # 3. RSI (과매도 반등 노림)
    if 30 <= rsi <= 45 and current_price > ma60:
        score += 2
    elif rsi > 70:
        score -= 3 # 과열

    # 4. MACD 골든크로스
    if macd > macd_signal and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
        score += 2

    # 결론 도출
    if score >= 4: return "강력 매수", "green", score
    elif score >= 2: return "매수 관점", "blue", score
    elif score <= -1: return "매도/관망", "red", score
    else: return "보유/관망", "gray", score

# ---------------------------------------------------------
# 3. 메인 UI
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 매니저")

tab1, tab2 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)"])

# === TAB 1: 전체 종목 일괄 분석 ===
with tab1:
    st.markdown("### 📋 관심 종목 일괄 진단")
    st.write("지정해주신 60여 개 종목을 AI가 실시간으로 분석하여 매수 타점을 포착합니다.")
    
    if st.button("전체 리스트 분석 시작 (Click)"):
        with st.spinner('데이터 수집 및 분석 중입니다... (약 10~20초 소요)'):
            raw_data, tickers = get_bulk_data(USER_WATCHLIST)
            
            results = []
            
            progress_bar = st.progress(0)
            for i, ticker in enumerate(tickers):
                try:
                    # MultiIndex 처리: 해당 티커의 데이터만 추출
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        try:
                            df_ticker = raw_data.xs(ticker, axis=1, level=1)
                        except:
                            df_ticker = raw_data[ticker]
                    else:
                        df_ticker = raw_data # 단일 종목일 경우
                        
                    df_ticker = df_ticker.dropna(how='all') # 모든 컬럼이 NaN인 행만 제거
                    
                    if df_ticker.empty: continue
                    
                    df_indi = calculate_indicators(df_ticker)
                    if df_indi is None or df_indi.empty: continue

                    action, color, score = analyze_strategy(df_indi)
                    
                    # 결과 저장
                    current_price = df_indi['Close'].iloc[-1]
                    rsi = df_indi['RSI'].iloc[-1]
                    
                    results.append({
                        "종목": ticker,
                        "현재가": f"{current_price:,.0f}",
                        "RSI": f"{rsi:.1f}",
                        "AI 판단": action,
                        "점수": score, # 정렬용
                        "색상": color # 표시용
                    })
                except Exception as e:
                    continue
                progress_bar.progress((i + 1) / len(tickers))
            
            # 결과 표시
            st.success("분석 완료!")
            
            if results:
                # DataFrame 변환 및 정렬 (점수 높은 순 = 매수 추천 순)
                res_df = pd.DataFrame(results)
                res_df = res_df.sort_values(by="점수", ascending=False)
                
                # 데이터프레임 표시
                st.dataframe(
                    res_df[['종목', '현재가', 'AI 판단', 'RSI']],
                    use_container_width=True,
                    height=600
                )

                # 강력 매수 추천만 따로 표시
                st.markdown("#### 🔥 오늘 강력 매수 추천 종목")
                strong_buys = res_df[res_df['AI 판단'] == '강력 매수']
                if not strong_buys.empty:
                    for idx, row in strong_buys.iterrows():
                        st.info(f"**{row['종목']}**: 눌림목 혹은 강력한 상승 모멘텀 발생! (RSI: {row['RSI']})")
                else:
                    st.write("현재 '강력 매수' 신호가 뜬 종목이 없습니다. 관망하세요.")
            else:
                st.warning("분석된 종목이 없습니다. 데이터 소스 연결 상태를 확인해주세요.")


# === TAB 2: 내 포트폴리오 (Firebase 연동) ===
with tab2:
    st.markdown("### ☁️ 내 자산 관리 (서버 저장)")
    
    db = get_db()
    
    if db is None:
        st.warning("⚠️ 데이터베이스(Firebase)가 연결되지 않았습니다.")
        st.info("""
        **[설정 방법]**
        1. Streamlit Cloud -> App Settings -> Secrets에 접속
        2. 제공드린 TOML 내용을 복사해서 붙여넣으세요.
        """)
    else:
        # 사용자 식별
        user_id = st.text_input("사용자 닉네임 (이 키로 데이터를 불러옵니다)", value="my_portfolio")
        
        if user_id:
            # 컬렉션 참조
            doc_ref = db.collection('portfolios').document(user_id)
            
            # 1. 데이터 불러오기
            try:
                doc = doc_ref.get()
                if doc.exists:
                    portfolio_data = doc.to_dict().get('stocks', [])
                else:
                    portfolio_data = []
            except Exception as e:
                st.error(f"데이터 불러오기 오류: {e}")
                portfolio_data = []

            # 2. 종목 추가 UI
            with st.form("add_stock_form"):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    new_ticker = st.text_input("종목 코드 (예: TSLA, 005930)")
                with col2:
                    new_price = st.number_input("평단가", min_value=0.0)
                with col3:
                    st.write("") 
                    st.write("") 
                    submitted = st.form_submit_button("저장")
                
                if submitted:
                    if new_ticker and new_price > 0:
                        formatted = format_ticker(new_ticker)
                        # 중복 제거 후 추가
                        portfolio_data = [p for p in portfolio_data if p['ticker'] != formatted]
                        portfolio_data.append({"ticker": formatted, "price": new_price})
                        
                        # DB 업데이트
                        doc_ref.set({'stocks': portfolio_data})
                        st.success(f"{formatted} 저장 완료!")
                        st.rerun()

            st.divider()

            # 3. 저장된 종목 분석
            if portfolio_data:
                st.subheader(f"💼 {user_id}님의 포트폴리오 진단")
                
                my_tickers = [p['ticker'] for p in portfolio_data]
                
                with st.spinner("포트폴리오 분석 중..."):
                    my_data, _ = get_bulk_data(my_tickers)
                
                for item in portfolio_data:
                    tk = item['ticker']
                    my_avg = item['price']
                    
                    try:
                        # 데이터 추출 (안전하게)
                        if isinstance(my_data.columns, pd.MultiIndex):
                            try:
                                df_tk = my_data.xs(tk, axis=1, level=1)
                            except:
                                df_tk = my_data[tk]
                        else:
                            df_tk = my_data
                            
                        df_tk = df_tk.dropna(how='all')
                        
                        # 지표 계산
                        df_tk = calculate_indicators(df_tk)
                        if df_tk is None:
                            st.warning(f"{tk}: 데이터 로딩 중...")
                            continue

                        curr = df_tk['Close'].iloc[-1]
                        profit = ((curr - my_avg) / my_avg) * 100
                        
                        # 스윙 전략 분석 (보유자 관점)
                        ma20 = df_tk['MA20'].iloc[-1]
                        rsi = df_tk['RSI'].iloc[-1]
                        
                        msg = ""
                        if profit > 0:
                            profit_color = "green"
                            if rsi > 70: msg = "🔥 익절 고려 (과열)"
                            elif curr < ma20: msg = "⚠️ 20일선 이탈 (주의)"
                            else: msg = "✅ 보유 (추세 지속)"
                        else:
                            profit_color = "red"
                            if curr < ma20 * 0.97: msg = "✂️ 손절 검토 (추세 붕괴)"
                            elif rsi < 30: msg = "💧 물타기/반등 대기 (과매도)"
                            else: msg = "⏳ 관망"

                        # 카드 형태로 출력
                        with st.container():
                            c1, c2, c3, c4 = st.columns([1, 2, 2, 3])
                            c1.markdown(f"**{tk}**")
                            c2.write(f"평단: {my_avg:,.0f}")
                            c3.markdown(f":{profit_color}[수익률: {profit:.2f}%]")
                            c4.markdown(f"**{msg}**")
                            
                        st.divider()

                    except Exception as e:
                        st.error(f"{tk} 분석 오류: {e}")
                
                if st.button("포트폴리오 초기화 (전체 삭제)"):
                    doc_ref.delete()
                    st.rerun()
            else:
                st.info("저장된 종목이 없습니다. 위에서 추가해주세요.")
