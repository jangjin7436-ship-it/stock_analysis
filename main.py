import streamlit as st
import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import datetime
import time
import json
import concurrent.futures
import requests
from bs4 import BeautifulSoup

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
# 2. 데이터 수집 (네이버 금융 크롤링 + YF Fast Info)
# ---------------------------------------------------------
def fetch_kr_realtime(ticker):
    """
    한국 주식 실시간 가격 크롤링 (네이버 금융)
    - FDR보다 정확하고 토스와 일치율이 높음
    """
    try:
        code = ticker.split('.')[0]
        url = f"https://finance.naver.com/item/sise.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 현재가 추출 (strong 태그의 id='_nowVal')
        price_str = soup.select_one('#_nowVal').text.replace(',', '')
        return (ticker, float(price_str))
    except:
        # 실패 시 FDR로 백업 시도
        try:
            df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
            if not df.empty:
                return (ticker, float(df['Close'].iloc[-1]))
        except:
            pass
        return (ticker, None)

def fetch_us_realtime(ticker):
    """미국 주식: 실시간/애프터마켓 가격 (fast_info)"""
    try:
        # fast_info는 가장 최신의 체결가(장후 포함)를 제공함
        price = yf.Ticker(ticker).fast_info['last_price']
        return (ticker, price)
    except:
        return (ticker, None)

def fetch_history_data(ticker):
    """지표 분석용 과거 데이터 (2년치)"""
    try:
        if ticker.endswith('.KS') or ticker.endswith('.KQ'):
            df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        else:
            df = yf.download(ticker, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                # 단일 종목 MultiIndex 처리
                try: df = df.xs(ticker, axis=1, level=0)
                except: pass
            if 'Close' not in df.columns and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
        return (ticker, df)
    except:
        return (ticker, None)

@st.cache_data(ttl=5) # 5초 캐시
def get_hybrid_data_v3(tickers_list):
    """
    실시간 가격(크롤링/FastInfo) + 과거 차트 데이터 병합
    """
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]
    
    final_dfs = {} 

    # 병렬 처리로 모든 데이터 수집
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 1. 실시간 가격 조회 작업
        future_realtime = []
        for t in kr_tickers:
            future_realtime.append(executor.submit(fetch_kr_realtime, t))
        for t in us_tickers:
            future_realtime.append(executor.submit(fetch_us_realtime, t))
            
        # 2. 과거 차트 조회 작업
        future_history = []
        for t in tickers_list:
            future_history.append(executor.submit(fetch_history_data, t))
            
        # 결과 수집
        realtime_map = {}
        for f in concurrent.futures.as_completed(future_realtime):
            tk, price = f.result()
            if price is not None: realtime_map[tk] = price
            
        history_map = {}
        for f in concurrent.futures.as_completed(future_history):
            tk, df = f.result()
            if df is not None and not df.empty: history_map[tk] = df

    # 3. 데이터 병합 (History 마지막 값을 Realtime으로 업데이트)
    for t in tickers_list:
        if t in history_map:
            df = history_map[t].copy()
            if t in realtime_map:
                latest_price = realtime_map[t]
                # 마지막 종가를 실시간 가격으로 덮어씀 (AI 분석 정확도 UP)
                df.iloc[-1, df.columns.get_loc('Close')] = latest_price
            final_dfs[t] = df
        elif t in realtime_map:
            # 차트 데이터는 없지만 실시간 가격은 있는 경우 (신규 상장 등)
            # 분석은 못하지만 가격 표시는 가능하게 더미 DF 생성 가능 (생략)
            pass

    return final_dfs, realtime_map

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

def calculate_total_profit(ticker, avg_price, current_price, quantity):
    """
    수수료/세금 반영한 순수익 및 순평가금 계산
    """
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
    if is_kr: fee_tax_rate = 0.0018 
    else: fee_tax_rate = 0.002
    
    # 1. 총 매수 금액 (원금)
    total_buy = avg_price * quantity
    
    # 2. 단순 평가 금액 (현재가 * 수량)
    raw_eval = current_price * quantity
    
    # 3. 매도 시 수수료/세금
    total_fee = raw_eval * fee_tax_rate
    
    # 4. 순 평가 금액 (세후 수령 예상액) - 사용자 요청: 평가금에도 수수료 제외
    net_eval = raw_eval - total_fee
    
    # 5. 순수익 (세후 평가금 - 원금)
    net_profit_amt = net_eval - total_buy
    
    # 6. 순수익률
    if total_buy > 0:
        net_profit_pct = (net_profit_amt / total_buy) * 100
    else:
        net_profit_pct = 0.0
    
    currency = "₩" if is_kr else "$"
    
    return {
        "pct": net_profit_pct,
        "profit_amt": net_profit_amt,
        "net_eval_amt": net_eval, # 세후 평가금
        "currency": currency
    }

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
    st.caption("실시간(After Market) 가격을 반영하여 AI가 분석합니다.")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 분석 새로고침", type="primary"):
            st.session_state['scan_result_df'] = None 
            st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 리스트 정밀 분석 시작"):
            with st.spinner('실시간 가격 반영 및 AI 분석 중...'):
                raw_data_dict, realtime_map = get_hybrid_data_v3(USER_WATCHLIST)
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
                        
                        curr_price = df_indi['Close'].iloc[-1]
                        rsi_val = df_indi['RSI'].iloc[-1]
                        name = TICKER_MAP.get(ticker_code, ticker_code)
                        
                        is_kr = ticker_code.endswith(".KS") or ticker_code.endswith(".KQ")
                        sym = "₩" if is_kr else "$"
                        
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
    st.caption("네이버페이/토스증권 실시간 기준 | 총 평가금도 세후(순수익) 기준으로 표시")
    
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
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                selected_item = st.selectbox("종목 검색", ["선택하세요"] + SEARCH_LIST)
            with c2:
                input_price = st.number_input("내 평단가", min_value=0.0, format="%.2f")
            with c3:
                input_qty = st.number_input("보유 수량(주)", min_value=0, value=1)
            with c4:
                st.write("")
                st.write("")
                if st.button("추가하기", type="primary"):
                    if selected_item != "선택하세요":
                        target_code = SEARCH_MAP[selected_item]
                        new_pf_data = [p for p in pf_data if p['ticker'] != target_code]
                        new_pf_data.append({
                            "ticker": target_code, 
                            "price": input_price,
                            "qty": input_qty
                        })
                        doc_ref.set({'stocks': new_pf_data})
                        st.success("추가 완료!")
                        time.sleep(0.5)
                        st.rerun()

        st.divider()

        if pf_data:
            st.subheader(f"{user_id}님의 보유 종목 진단")
            my_tickers = [p['ticker'] for p in pf_data]
            with st.spinner("실시간 시세 조회 중... (국내:네이버크롤링, 해외:YF)"):
                raw_data_dict, _ = get_hybrid_data_v3(my_tickers)
            
            display_list = []
            for item in pf_data:
                tk = item['ticker']
                avg = item['price']
                qty = item.get('qty', 1)
                name = TICKER_MAP.get(tk, tk)
                
                df_tk = None
                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                
                cat, col_name, reasoning, score = "데이터 로딩 중", "gray", "잠시 후 다시 시도", 0
                curr = 0
                
                if df_tk is not None and not df_tk.empty:
                    # 지표 계산
                    df_indi = calculate_indicators(df_tk)
                    # 현재가는 최신으로 업데이트된 마지막 종가 사용
                    curr = df_tk['Close'].iloc[-1]
                    
                    if df_indi is not None:
                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)

                if curr > 0:
                    res = calculate_total_profit(tk, avg, curr, qty)
                    
                    display_list.append({
                        "name": TICKER_MAP.get(tk, tk), "tk": tk, 
                        "avg": avg, "curr": curr, "qty": qty,
                        "cat": cat, "col_name": col_name, "reasoning": reasoning,
                        "profit_pct": res['pct'], 
                        "profit_amt": res['profit_amt'],
                        "eval_amt": res['net_eval_amt'], # 세후 평가금
                        "currency": res['currency'], 
                        "score": score
                    })
                else:
                    display_list.append({
                        "name": TICKER_MAP.get(tk, tk), "tk": tk, 
                        "avg": avg, "curr": avg, "qty": qty,
                        "cat": "로딩 실패", "col_name": "gray", "reasoning": "데이터 수신 불가",
                        "profit_pct": 0.0, "profit_amt": 0.0, "eval_amt": 0.0,
                        "currency": "$" if not tk.endswith(".KS") else "₩", "score": 0
                    })
            
            display_list.sort(key=lambda x: x['score'], reverse=True)

            for item in display_list:
                with st.container():
                    c1, c2, c3 = st.columns([1.5, 1.5, 4])
                    sym = item['currency'] # 역슬래시 제거
                    
                    with c1:
                        st.markdown(f"### {item['name']}")
                        st.caption(f"{item['tk']} | 보유: {item['qty']}주")
                        
                    with c2:
                        fmt_curr = f"{item['curr']:,.0f}" if item['currency'] == "₩" else f"{item['curr']:,.2f}"
                        fmt_avg = f"{item['avg']:,.0f}" if item['currency'] == "₩" else f"{item['avg']:,.2f}"
                        fmt_profit = f"{item['profit_amt']:,.0f}" if item['currency'] == "₩" else f"{item['profit_amt']:,.2f}"
                        fmt_eval = f"{item['eval_amt']:,.0f}" if item['currency'] == "₩" else f"{item['eval_amt']:,.2f}"
                        
                        st.metric("총 순수익 (수수료 제)", f"{item['profit_pct']:.2f}%", delta=f"{sym}{fmt_profit}")
                        
                        # 평가금 표시 (세후)
                        st.markdown(f"**세후 총 평가금:** {sym}{fmt_eval}")
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
    st.markdown("## 📘 AI 투자 전략 알고리즘 상세 백서 (Whitepaper)")
    st.markdown("단순한 지표 합산이 아닌, **'수익은 길게, 손실은 짧게'** 가져가는 프로 트레이더의 로직을 구현했습니다.")
    st.divider()
    st.subheader("1. 💯 점수 산정 로직 (Total 100점)")
    score_table = pd.DataFrame({
        "평가 요소": ["추세 (Trend)", "지지 (Support)", "모멘텀 (Momentum)", "거래량 (Volume)", "리스크 (Risk)"],
        "내용": ["60일선/20일선 위에 있는가?", "싸게 살 수 있는 자리인가? (눌림목/볼린저 하단)", "상승 에너지가 강한가? (MACD)", "세력이 들어왔는가?", "너무 비싸진 않은가? (과열)"],
        "배점": ["±15~25점", "+15~25점 (가산점)", "±15점", "+10점", "±10~20점"]
    })
    st.table(score_table)
