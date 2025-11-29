import streamlit as st
import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import datetime
import time
import json
import concurrent.futures

# ---------------------------------------------------------
# 0. 파이어베이스(DB) 설정
# ---------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, firestore

def get_db():
    if not firebase_admin._apps:
        try:
            if 'firebase_key' in st.secrets:
                secret_val = st.secrets['firebase_key']
                if isinstance(secret_val, str):
                    key_dict = json.loads(secret_val)
                else:
                    key_dict = dict(secret_val)
                
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
# 1. 설정 및 매핑
# ---------------------------------------------------------
st.set_page_config(page_title="AI 주식 스캐너 Pro", page_icon="📈", layout="wide")

if 'scan_result_df' not in st.session_state:
    st.session_state['scan_result_df'] = None

TICKER_MAP = {
    "INTC": "인텔 (Intel)", "005290.KS": "동진쎄미켐", "SOXL": "반도체 3X(Bull)", 
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
    "NVDA": "엔비디아", "GE": "GE에어로스페이스", "V": "비자(Visa)", 
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

SEARCH_LIST = [f"{name} ({code})" for code, name in TICKER_MAP.items()]
SEARCH_MAP = {f"{name} ({code})": code for code, name in TICKER_MAP.items()}
USER_WATCHLIST = list(TICKER_MAP.keys())

# ---------------------------------------------------------
# 2. 데이터 수집 (NXT/After Market 대응)
# ---------------------------------------------------------
def fetch_single_kr_stock(ticker):
    try:
        code = ticker.split('.')[0]
        df = fdr.DataReader(code, '2023-01-01')
        if df.empty: return None
        return (ticker, df)
    except:
        return None

def get_realtime_price_us(ticker):
    """
    미국 주식 실시간 가격 (NXT/After Market 포함)
    yfinance의 fast_info를 사용하여 최신 체결가를 가져옴
    """
    try:
        # fast_info는 API 호출 없이 최근 메타데이터를 빠르게 가져옴
        info = yf.Ticker(ticker).fast_info
        # last_price는 정규장 종료 후에는 After market 가격을 반영하는 경우가 많음
        return info['last_price']
    except:
        return None

@st.cache_data(ttl=5)
def get_hybrid_data(tickers_list):
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]
    combined_data = {}

    # 1. 한국 주식 (병렬)
    if kr_tickers:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_ticker = {executor.submit(fetch_single_kr_stock, t): t for t in kr_tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                result = future.result()
                if result:
                    ticker, df = result
                    combined_data[ticker] = df

    # 2. 미국 주식 (Bulk History) - 지표 계산용
    if us_tickers:
        try:
            yf_data = yf.download(us_tickers, period="2y", group_by='ticker', threads=True, prepost=True)
            if len(us_tickers) == 1:
                combined_data[us_tickers[0]] = yf_data
            else:
                for t in us_tickers:
                    try:
                        if isinstance(yf_data.columns, pd.MultiIndex):
                            df = yf_data.xs(t, axis=1, level=1)
                        else: df = yf_data
                        if not df.empty: combined_data[t] = df
                    except: pass
        except Exception as e:
            pass # 다운로드 실패 시 개별 처리로 넘어감
                    
    return combined_data

def calculate_indicators(df):
    if len(df) < 60: return None
    df = df.copy()
    df['Close'] = df['Close'].ffill()

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    if 'Volume' in df.columns:
        df['VolMA20'] = df['Volume'].rolling(window=20).mean()
    else:
        df['VolMA20'] = 0

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)
    
    return df.dropna()

def calculate_net_profit(ticker, avg_price, current_price):
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
    if is_kr: fee_tax_rate = 0.0018 
    else: fee_tax_rate = 0.002
    
    net_sell_price = current_price * (1 - fee_tax_rate)
    profit_amt = net_sell_price - avg_price
    profit_pct = (profit_amt / avg_price) * 100
    currency = "₩" if is_kr else "$"
    
    return profit_pct, profit_amt, currency

# ---------------------------------------------------------
# 3. 전략 분석
# ---------------------------------------------------------
def analyze_advanced_strategy(df):
    if df is None or df.empty: return "분석 불가", "gray", "데이터 부족", 0
    
    curr = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    sig = df['Signal_Line'].iloc[-1]
    bb_upper = df['BB_Upper'].iloc[-1]
    bb_lower = df['BB_Lower'].iloc[-1]
    vol = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
    vol_ma = df['VolMA20'].iloc[-1] if 'VolMA20' in df.columns else 0

    prev_macd = df['MACD'].iloc[-2]
    prev_sig = df['Signal_Line'].iloc[-2]

    trend_up = curr > ma60
    above_ma20 = curr > ma20
    golden_cross = (macd > sig) and (prev_macd <= prev_sig)
    dead_cross = (macd < sig) and (prev_macd >= prev_sig)
    oversold = rsi < 35
    overbought = rsi > 70
    dist_to_ma20 = (curr - ma20) / ma20
    dip_buy = trend_up and abs(dist_to_ma20) <= 0.02

    score = 50
    reasons = []

    if curr > ma60:
        score += 15
        if curr > ma20: score += 10
        else: score -= 5 
    else:
        score -= 20
        if curr < ma20: score -= 10

    if dip_buy:
        score += 25
        reasons.append("💎 황금 눌림목 (상승장 속 조정)")
    
    if curr <= bb_lower * 1.02:
        score += 15
        reasons.append("📉 볼린저 밴드 하단 (저점 매수)")
    
    if curr >= bb_upper * 0.98:
        score -= 10
        reasons.append("⚠️ 볼린저 밴드 상단 (고점)")

    if macd > sig and prev_macd <= prev_sig:
        score += 15
        reasons.append("⚡ MACD 골든크로스")
    elif macd > sig: score += 5
    elif macd < sig and prev_macd >= prev_sig:
        score -= 15
        reasons.append("💧 MACD 데드크로스")

    if vol > vol_ma * 1.5 and curr > df['Open'].iloc[-1]:
        score += 10
        reasons.append("🔥 거래량 폭발")

    if rsi < 30:
        score += 15
        reasons.append("zzZ 과매도 (반등 기대)")
    elif rsi > 75:
        score -= 20
        reasons.append("🔥 RSI 과열")
    elif 30 <= rsi <= 50: score += 5

    score = max(0, min(100, score))

    category = "관망 (Neutral)"
    color_name = "gray"

    if score >= 80:
        category = "🚀 강력 매수 (Strong Buy)"
        color_name = "green"
    elif score >= 60:
        category = "📈 매수 (Buy)"
        color_name = "blue"
    elif score <= 20:
        category = "💥 강력 매도 (Strong Sell)"
        color_name = "red"
    elif score <= 40:
        category = "📉 매도 (Sell)"
        color_name = "red"
    else:
        category = "👀 관망 (Neutral)"
        color_name = "gray"
        if not reasons: reasons.append("방향성 탐색 중")

    return category, color_name, ", ".join(reasons), score

# ---------------------------------------------------------
# 4. UI
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2, tab3 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)", "📘 알고리즘 설명서"])

with tab1:
    st.markdown("### 📋 AI 정밀 스캐너")
    st.caption("안전성(저점 매수)과 수익성(추세/모멘텀)을 종합 평가하여 점수를 매깁니다.")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 분석 새로고침", type="primary"):
            st.session_state['scan_result_df'] = None 
            st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 리스트 정밀 분석 시작"):
            with st.spinner('데이터 수집 및 분석 중... (NXT 반영)'):
                raw_data_dict = get_hybrid_data(USER_WATCHLIST)
                scan_results = []
                progress_bar = st.progress(0)
                
                for i, ticker_code in enumerate(USER_WATCHLIST):
                    if ticker_code not in raw_data_dict: continue
                    try:
                        df_tk = raw_data_dict[ticker_code].dropna(how='all')
                        if df_tk.empty: continue
                        
                        df_indi = calculate_indicators(df_tk)
                        if df_indi is None: continue

                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)
                        
                        # 스캐너에서는 속도를 위해 df 마지막 값을 쓰되, NXT는 포트폴리오에서 집중
                        curr_price = df_indi['Close'].iloc[-1]
                        rsi_val = df_indi['RSI'].iloc[-1]
                        name = TICKER_MAP.get(ticker_code, ticker_code)
                        
                        is_kr = ticker_code.endswith(".KS") or ticker_code.endswith(".KQ")
                        sym = "₩" if is_kr else "$"
                        
                        # 미국 주식일 경우 실시간 가격(NXT) 시도 (스캐너에서는 느려질 수 있어 생략하거나 옵션 처리)
                        # 여기서는 속도를 위해 기존 로직 유지, 포트폴리오에서만 정밀 적용
                        
                        fmt_price = f"{sym}{curr_price:,.0f}" if is_kr else f"{sym}{curr_price:,.2f}"

                        scan_results.append({
                            "종목명": f"{name} ({ticker_code})",
                            "점수": score,
                            "현재가": fmt_price,
                            "RSI": rsi_val,
                            "AI 등급": cat,
                            "핵심 요약": reasoning
                        })
                    except: continue
                    progress_bar.progress((i + 1) / len(USER_WATCHLIST))
                
                if scan_results:
                    df_res = pd.DataFrame(scan_results)
                    df_res = df_res.sort_values('점수', ascending=False)
                    st.session_state['scan_result_df'] = df_res
                    st.success("완료!")
                    st.rerun()
                else:
                    st.error("데이터 수집 실패.")
    
    if st.session_state['scan_result_df'] is not None:
        st.dataframe(
            st.session_state['scan_result_df'],
            use_container_width=True,
            height=700,
            column_config={
                "종목명": st.column_config.TextColumn("종목명 (코드)", width="medium"),
                "점수": st.column_config.ProgressColumn("AI 점수", format="%d점", min_value=0, max_value=100),
                "현재가": st.column_config.TextColumn("현재가"), 
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "AI 등급": st.column_config.TextColumn("AI 판단"),
                "핵심 요약": st.column_config.TextColumn("분석 내용", width="large"),
            },
            hide_index=True
        )

with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오")
    st.caption("NXT(After Market) 가격 적용 | 수수료/세금 적용")
    
    db = get_db()
    if not db:
        st.warning("⚠️ Firebase 설정 필요")
    else:
        col_u1, col_u2 = st.columns([1, 3])
        with col_u1:
            user_id = st.text_input("닉네임", value="장동진")
        doc_ref = db.collection('portfolios').document(user_id)
        try:
            doc = doc_ref.get()
            pf_data = doc.to_dict().get('stocks', []) if doc.exists else []
        except: pf_data = []

        with st.container():
            st.markdown("#### ➕ 종목 추가")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                selected_item = st.selectbox("종목 검색", ["선택하세요"] + SEARCH_LIST)
            with c2:
                input_price = st.number_input("내 평단가", min_value=0.0, format="%.2f")
            with c3:
                st.write("")
                st.write("")
                if st.button("추가하기", type="primary"):
                    if selected_item != "선택하세요":
                        target_code = SEARCH_MAP[selected_item]
                        new_pf_data = [p for p in pf_data if p['ticker'] != target_code]
                        new_pf_data.append({"ticker": target_code, "price": input_price})
                        doc_ref.set({'stocks': new_pf_data})
                        st.success("추가 완료!")
                        time.sleep(0.5)
                        st.rerun()

        st.divider()

        if pf_data:
            st.subheader(f"{user_id}님의 보유 종목 진단")
            my_tickers = [p['ticker'] for p in pf_data]
            with st.spinner("실시간(NXT) 시세 조회 중..."):
                raw_data_dict = get_hybrid_data(my_tickers)
            
            display_list = []
            for item in pf_data:
                tk = item['ticker']
                avg = item['price']
                name = TICKER_MAP.get(tk, tk)
                
                # 데이터가 없어도 카드는 표시하되, 분석 불가로 처리
                df_tk = None
                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                
                cat, col_name, reasoning, score = "데이터 로딩 중", "gray", "잠시 후 다시 시도", 0
                curr = 0
                
                # 1. 지표 분석 (History 사용)
                if df_tk is not None and not df_tk.empty:
                    df_indi = calculate_indicators(df_tk)
                    if df_indi is not None:
                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)
                        curr = df_indi['Close'].iloc[-1] # 기본값

                # 2. 가격 보정 (미국 주식 NXT 적용)
                is_kr = tk.endswith(".KS") or tk.endswith(".KQ")
                if not is_kr:
                    nxt_price = get_realtime_price_us(tk)
                    if nxt_price:
                        curr = nxt_price # NXT 가격으로 덮어쓰기

                if curr > 0:
                    profit_pct, profit_amt, currency = calculate_net_profit(tk, avg, curr)
                    
                    display_list.append({
                        "name": TICKER_MAP.get(tk, tk), "tk": tk, "avg": avg, "curr": curr,
                        "cat": cat, "col_name": col_name, "reasoning": reasoning,
                        "profit_pct": profit_pct, "profit_amt": profit_amt,
                        "currency": currency, "score": score
                    })
                else:
                    # 데이터 로딩 실패 시에도 목록에는 띄워줌
                    display_list.append({
                        "name": TICKER_MAP.get(tk, tk), "tk": tk, "avg": avg, "curr": avg,
                        "cat": "로딩 실패", "col_name": "gray", "reasoning": "데이터 수신 불가",
                        "profit_pct": 0.0, "profit_amt": 0.0,
                        "currency": "$" if not tk.endswith(".KS") else "₩", "score": 0
                    })
            
            display_list.sort(key=lambda x: x['score'], reverse=True)

            for item in display_list:
                with st.container():
                    c1, c2, c3 = st.columns([1.5, 1.5, 4])
                    with c1:
                        st.markdown(f"### {item['name']}")
                        st.caption(f"{item['tk']}")
                    with c2:
                        sym = item['currency'].replace("$", "\$")
                        fmt_curr = f"{item['curr']:,.0f}" if item['currency'] == "₩" else f"{item['curr']:,.2f}"
                        fmt_avg = f"{item['avg']:,.0f}" if item['currency'] == "₩" else f"{item['avg']:,.2f}"
                        fmt_diff = f"{item['profit_amt']:,.0f}" if item['currency'] == "₩" else f"{item['profit_amt']:,.2f}"
                        
                        st.metric("순수익률", f"{item['profit_pct']:.2f}%", delta=f"{sym}{fmt_diff}")
                        st.markdown(f"<small style='color: gray'>평단: {sym}{fmt_avg} / 현재: {sym}{fmt_curr}</small>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"**AI 점수: {item['score']}점**")
                        st.markdown(f"**판단:** :{item['col_name']}[{item['cat']}]")
                        st.info(f"💡 {item['reasoning']}")
                    st.divider()

            if st.button("🗑️ 포트폴리오 전체 삭제"):
                doc_ref.delete()
                st.rerun()

with tab3:
    st.markdown("## 📘 AI 투자 전략 알고리즘 가이드 (V2.0)")
    st.markdown("단순한 지표 합산이 아닌, **'수익은 길게, 손실은 짧게'** 가져가는 프로 트레이더의 로직을 구현했습니다.")
    st.divider()
    st.subheader("1. 💯 점수 산정 로직 (Total 100점)")
    score_table = pd.DataFrame({
        "평가 요소": ["추세 (Trend)", "지지 (Support)", "모멘텀 (Momentum)", "거래량 (Volume)", "리스크 (Risk)"],
        "내용": ["60일선/20일선 위에 있는가?", "싸게 살 수 있는 자리인가? (눌림목/볼린저 하단)", "상승 에너지가 강한가? (MACD)", "세력이 들어왔는가?", "너무 비싸진 않은가? (과열)"],
        "배점": ["±15~25점", "+15~25점 (가산점)", "±15점", "+10점", "±10~20점"]
    })
    st.table(score_table)
