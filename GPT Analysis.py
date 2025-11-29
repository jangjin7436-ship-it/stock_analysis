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

        except Exception:

            return None

    return firestore.client()





# ---------------------------------------------------------

# 1. 설정 및 매핑

# ---------------------------------------------------------

st.set_page_config(page_title="AI 주식 스캐너 Pro", page_icon="📈", layout="wide")



if 'scan_result_df' not in st.session_state:

    st.session_state['scan_result_df'] = None



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



SEARCH_LIST = [f"{name} ({code})" for code, name in TICKER_MAP.items()]

SEARCH_MAP = {f"{name} ({code})": code for code, name in TICKER_MAP.items()}

USER_WATCHLIST = list(TICKER_MAP.keys())





# ---------------------------------------------------------

# 2. 데이터 수집

# ---------------------------------------------------------

@st.cache_data(ttl=60)

def get_bulk_us_data(us_tickers):

    """미국 주식 데이터 수집"""

    if not us_tickers:

        return {}, {}



    hist_map = {}

    realtime_map = {}



    try:

        # auto_adjust=False로 설정하여 실제 체결가 기준 계산 (백테스트 로직과 일치)

        df_hist = yf.download(

            us_tickers,

            period="2y",

            interval="1d",

            progress=False,

            group_by="ticker",

            auto_adjust=False, 

        )

        df_real = yf.download(

            us_tickers,

            period="5d",

            interval="1m",

            progress=False,

            group_by="ticker",

            prepost=True,

        )



        hist_is_multi = isinstance(df_hist.columns, pd.MultiIndex)

        real_is_multi = isinstance(df_real.columns, pd.MultiIndex)



        for t in us_tickers:

            try:

                sub_df = df_hist[t] if hist_is_multi else df_hist

                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:

                    sub_df = sub_df.dropna(how="all")

                    if "Close" in sub_df.columns:

                        hist_map[t] = sub_df

            except Exception:

                pass



            try:

                sub_real = df_real[t] if real_is_multi else df_real

                if isinstance(sub_real, pd.DataFrame) and not sub_real.empty:

                    sub_real = sub_real.dropna(how="all")

                    price_series = sub_real["Close"]

                    if price_series is not None:

                        valid_closes = price_series.dropna()

                        if not valid_closes.empty:

                            realtime_map[t] = float(valid_closes.iloc[-1])

            except Exception:

                pass



    except Exception:

        pass



    return hist_map, realtime_map





def fetch_kr_polling(ticker):

    """국내 주식 실시간 (네이버)"""

    code = ticker.split('.')[0]

    try:

        url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"

        headers = {"User-Agent": "Mozilla/5.0"}

        res = requests.get(url, headers=headers, timeout=2)

        data = res.json()

        item = data['datas'][0]



        close = float(str(item['closePrice']).replace(',', ''))

        return ticker, close

    except Exception:

        return ticker, None





def fetch_kr_history(ticker):

    try:

        df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')

        return ticker, df

    except Exception:

        return ticker, None





@st.cache_data(ttl=0)

def get_precise_data(tickers_list):

    """통합 데이터 수집기"""

    if not tickers_list:

        return {}, {}



    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]

    us_tickers = [t for t in tickers_list if t not in kr_tickers]



    hist_map, realtime_map = get_bulk_us_data(us_tickers)



    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

        fut_real = [executor.submit(fetch_kr_polling, t) for t in kr_tickers]

        fut_hist = [executor.submit(fetch_kr_history, t) for t in kr_tickers]



        for f in fut_real:

            try:

                tk, p = f.result(timeout=3)

                if p: realtime_map[tk] = p

            except: continue



        for f in fut_hist:

            try:

                tk, df = f.result(timeout=5)

                if df is not None and not df.empty:

                    hist_map[tk] = df

            except: continue



    return hist_map, realtime_map



# ---------------------------------------------------------

# 3. 분석 엔진 (NEW: ATR 및 개선된 알고리즘 적용)

# ---------------------------------------------------------



def calculate_indicators(df, realtime_price=None):

    """

    [NEW] 지표 계산 로직 (백테스트 코드와 100% 일치)

    - MA120, Disparity, Slope, ATR, BB, RSI, MACD

    """

    if df is None or len(df) < 120:  # MA120 계산을 위해 최소 데이터 필요

        return None



    if isinstance(df, pd.Series):

        df = df.to_frame()

    

    df = df.copy()



    # 컬럼 정리

    if 'Close' in df.columns:

        df['Close_Calc'] = df['Close']

    elif 'Adj Close' in df.columns:

        df['Close_Calc'] = df['Adj Close']

    else:

        return None

        

    df['Close_Calc'] = df['Close_Calc'].astype(float)

    

    # High/Low 확인 (ATR 계산용)

    if 'High' not in df.columns or 'Low' not in df.columns:

        # High/Low 없으면 Close로 대체 (불완전하지만 에러 방지)

        df['High'] = df['Close_Calc']

        df['Low'] = df['Close_Calc']



    # 실시간 가격 주입 및 High/Low 보정

    if realtime_price is not None:

        try:

            rp = float(realtime_price)

            if rp > 0:

                df['Close_Calc'].iloc[-1] = rp

                # 실시간 가격이 기존 High보다 높거나 Low보다 낮으면 갱신

                if rp > df['High'].iloc[-1]:

                    df['High'].iloc[-1] = rp

                if rp < df['Low'].iloc[-1]:

                    df['Low'].iloc[-1] = rp

        except:

            pass



    # 1. 이동평균

    df['MA5'] = df['Close_Calc'].rolling(5).mean()

    df['MA10'] = df['Close_Calc'].rolling(10).mean()

    df['MA20'] = df['Close_Calc'].rolling(20).mean()

    df['MA60'] = df['Close_Calc'].rolling(60).mean()

    df['MA120'] = df['Close_Calc'].rolling(120).mean()



    # 2. 이격도 및 기울기 (핵심)

    df['Disparity_20'] = df['Close_Calc'] / df['MA20']

    df['MA20_Slope'] = df['MA20'].diff()

    df['MA60_Slope'] = df['MA60'].diff()

    df['MA120_Slope'] = df['MA120'].diff()



    # 3. 볼린저 밴드

    std = df['Close_Calc'].rolling(20).std()

    df['Upper_Band'] = df['MA20'] + (std * 2)

    df['Lower_Band'] = df['MA20'] - (std * 2)

    

    # 4. RSI

    delta = df['Close_Calc'].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss

    df['RSI'] = 100 - (100 / (1 + rs))

    

    # 5. MACD

    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()

    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()

    df['MACD'] = exp12 - exp26

    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)

    

    # 6. ATR (Average True Range) - 변동성 지표

    prev_close = df['Close_Calc'].shift(1)

    tr1 = df['High'] - df['Low']

    tr2 = abs(df['High'] - prev_close)

    tr3 = abs(df['Low'] - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df['ATR'] = tr.rolling(14).mean()



    # 7. 거래량 (Volume Ratio)

    if 'Volume' in df.columns:

        df['Vol_MA20'] = df['Volume'].rolling(20).mean()

        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']

    else:

        df['Vol_Ratio'] = 1.0



    return df.dropna()





def get_ai_score_row(row):

    """

    [NEW] AI 점수 로직 (백테스트 코드 이식)

    - 추세 내 눌림목(Dip Buying) 및 과열 방지 중심

    """

    try:

        score = 50.0

        curr = row['Close_Calc']

        ma5, ma20, ma60 = row['MA5'], row['MA20'], row['MA60']

        rsi = row['RSI']

        

        # 1. 추세 판단

        if row['MA60_Slope'] > 0:

            score += 10.0

        else:

            score -= 10.0

            

        if curr > ma60:

            score += 5.0

        else:

            score -= 5.0

            

        if row['MA120_Slope'] > 0:

            score += 5.0

        elif row['MA120_Slope'] < 0:

            score -= 5.0



        # 2. 진입 타이밍 (눌림목 우대)

        if row['MA20_Slope'] > 0:

            if curr > ma20:

                score += 5.0

                # 눌림목 보너스 (MA5 근처 혹은 아래)

                if curr < ma5 * 1.01: 

                    score += 5.0

        

        # 3. 과열 방지 (이격도 필터)

        disparity = row['Disparity_20']

        if disparity > 1.10: 

            score -= 20.0  # 고점 추격 매수 방지

        elif disparity > 1.05:

            score -= 5.0



        # 4. 보조지표 혼합

        if row['MACD_Hist'] > row['Prev_MACD_Hist']:

            score += 5.0

        

        # RSI: 40~60 선호, 70 이상 감점

        if 40 <= rsi <= 60: 

            score += 5.0

        elif rsi > 70: 

            score -= 10.0



        # 볼린저 밴드 하단 터치

        if curr <= row['Lower_Band'] * 1.02:

            score += 10.0



        # 거래량 실린 양봉

        if row['Vol_Ratio'] >= 1.5 and curr > row['Open']:

            score += 5.0



        return max(0.0, min(100.0, score))

    except:

        return 0.0





def analyze_advanced_strategy(df):

    """

    [NEW] 스캐너 결과 해석 함수

    - 백테스트의 'Candidates' 선정 로직 반영

    - 점수 >= 75점 & ATR 안정성 등 체크

    """

    if df is None or df.empty:

        return "분석 불가", "gray", "데이터 부족", 0.0



    try:

        row = df.iloc[-1]

        score = float(get_ai_score_row(row))



        curr = float(row['Close_Calc'])

        ma20 = float(row['MA20'])

        ma60 = float(row['MA60'])

        rsi = float(row['RSI'])

        atr = float(row['ATR'])

        disparity = float(row['Disparity_20'])

        

    except Exception:

        return "오류", "gray", "계산 실패", 0.0



    reasons = []



    # 1) 추세 상태

    if row['MA60_Slope'] > 0 and curr > ma60:

        reasons.append("상승 추세(60일↑)")

    elif row['MA60_Slope'] < 0:

        reasons.append("하락 추세(60일↓)")



    # 2) 눌림목/과열 여부

    if disparity > 1.1:

        reasons.append("⚠️ 과열(이격도 110%↑)")

    elif 1.0 <= disparity <= 1.03:

        reasons.append("⚡ 20일선 근접(눌림)")

    elif disparity < 0.97:

        reasons.append("📉 과매도 구간")



    # 3) RSI

    if 40 <= rsi <= 60:

        reasons.append("RSI 안정(40-60)")

    elif rsi > 70:

        reasons.append("RSI 과열(70↑)")



    # 4) ATR (변동성 리스크)

    atr_ratio = atr / curr if curr > 0 else 0

    if atr_ratio > 0.05:

        reasons.append("⚠️ 고변동성 주의")

    

    # 5) MACD

    if row['MACD_Hist'] > row['Prev_MACD_Hist']:

        reasons.append("MACD 개선중")



    # ---- AI 등급 판정 (백테스트 기준) ----

    # Filter 1: 고변동성 제외

    is_high_risk = atr_ratio > 0.05

    

    if score >= 75 and not is_high_risk:

        cat = "🚀 AI 스나이퍼 매수 (강력)"

        col = "green"

    elif score >= 60 and not is_high_risk:

        cat = "📈 매수 우위 (양호)"

        col = "blue"

    elif disparity > 1.1 or rsi > 70:

        cat = "📉 이익 실현 / 과열"

        col = "orange"

    elif score < 40:

        cat = "💥 매도 / 관망 권장"

        col = "red"

    else:

        cat = "👀 중립 / 관망"

        col = "gray"



    reasoning = " / ".join(reasons[:3]) if reasons else "지표 중립"

    return cat, col, reasoning, round(score, 2)





def calculate_total_profit(ticker, avg_price, current_price, quantity):

    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")

    qty, avg, curr = float(quantity), float(avg_price), float(current_price)



    total_buy = avg * qty

    gross_eval = curr * qty



    fee_rate = 0.000295 if is_kr else 0.001965

    tax_rate = 0.0015 if is_kr else 0.0



    sell_fee = gross_eval * fee_rate

    sell_tax = gross_eval * tax_rate

    net_eval = gross_eval - sell_fee - sell_tax

    net_profit = net_eval - total_buy

    pct = (net_profit / total_buy) * 100 if total_buy > 0 else 0.0



    return {

        "pct": pct,

        "profit_amt": net_profit,

        "net_eval_amt": net_eval,

        "currency": "₩" if is_kr else "$",

    }





# ---------------------------------------------------------

# 4. UI

# ---------------------------------------------------------

st.title("🎯 AI 주식 스캐너 by GPT")



tab1, tab2, tab3 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)", "📘 알고리즘 백서"])



# TAB 1: 스캐너

with tab1:

    st.markdown("### 📋 AI 정밀 스캐너")

    st.caption("초정밀 실시간/AfterMarket 데이터 기반 AI 분석 (AI 스나이퍼 기준)")



    col_btn, col_info = st.columns([1, 4])

    with col_btn:

        if st.button("🔄 분석 새로고침", type="primary"):

            st.session_state['scan_result_df'] = None

            st.rerun()



    if st.session_state['scan_result_df'] is None:

        if st.button("🔍 전체 리스트 정밀 분석 시작"):

            with st.spinner('초정밀 데이터 수집 및 분석 중...'):

                raw_data_dict, realtime_map = get_precise_data(USER_WATCHLIST)

                scan_results = []

                progress_bar = st.progress(0)



                for i, ticker_code in enumerate(USER_WATCHLIST):

                    if ticker_code not in raw_data_dict:

                        continue

                    try:

                        df_tk = raw_data_dict[ticker_code].dropna(how='all')

                        if df_tk.empty:

                            continue



                        curr_price = realtime_map.get(ticker_code)

                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)



                        if df_indi is None or df_indi.empty:

                            continue



                        # 🔥 레버리지 종목 필터링 (3X, 2X 등은 알고리즘상 불리할 수 있음 표시)

                        name = TICKER_MAP.get(ticker_code, ticker_code)

                        is_leverage = any(x in name for x in ["3X", "2X", "1.5X"])

                        

                        # 🔥 백테스트와 동일한 AI_Score/스나이퍼 기준으로 매수/매도 해석

                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)



                        # 레버리지 종목 별도 표기

                        if is_leverage and score >= 70:

                            reasoning += " (레버리지 주의)"



                        final_price = float(df_indi['Close_Calc'].iloc[-1])

                        rsi_val = float(df_indi['RSI'].iloc[-1])

                        vol_ratio = float(df_indi['Vol_Ratio'].iloc[-1]) if 'Vol_Ratio' in df_indi.columns else 0



                        is_kr = ticker_code.endswith(".KS") or ticker_code.endswith(".KQ")

                        sym = "₩" if is_kr else "$"

                        fmt_price = f"{sym}{final_price:,.0f}" if is_kr else f"{sym}{final_price:,.2f}"



                        scan_results.append({

                            "종목명": f"{name} ({ticker_code})",

                            "점수": score,

                            "현재가": fmt_price,

                            "RSI": rsi_val,

                            "AI 등급": cat,

                            "핵심 요약": reasoning,

                            "거래량비율": vol_ratio,

                        })

                    except Exception:

                        continue

                    progress_bar.progress((i + 1) / len(USER_WATCHLIST))



                if scan_results:

                    df_res = pd.DataFrame(scan_results)

                    df_res = df_res.sort_values('점수', ascending=False)

                    st.session_state['scan_result_df'] = df_res

                    st.success("완료! (결과는 '분석 새로고침' 전까지 고정됩니다)")

                    st.rerun()

                else:

                    st.error("데이터 수집 실패.")

    if st.session_state['scan_result_df'] is not None:

        df_scan = st.session_state['scan_result_df']



        try:

            if "점수" in df_scan.columns:

                df_high = df_scan[df_scan["점수"] >= 80.0]

                if not df_high.empty:

                    st.markdown("#### 🔥 강력 매수 시그널 (Score 80+)")

                    st.dataframe(

                        df_high[["종목명", "점수", "현재가", "RSI", "AI 등급", "핵심 요약"]],

                        use_container_width=True,

                        hide_index=True,

                    )

        except Exception:

            pass



        st.dataframe(

            df_scan,

            use_container_width=True,

            height=700,

            column_config={

                "종목명": st.column_config.TextColumn("종목명 (코드)", width="medium"),

                "점수": st.column_config.ProgressColumn("AI 점수", format="%.1f점", min_value=0, max_value=100),

                "현재가": st.column_config.TextColumn("현재가"),

                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),

                "AI 등급": st.column_config.TextColumn("AI 판단"),

                "핵심 요약": st.column_config.TextColumn("분석 내용", width="large"),

                "거래량비율": st.column_config.NumberColumn("Vol Ratio", format="%.2f"),

            },

            hide_index=True,

        )



# TAB 2: 포트폴리오

with tab2:

    st.markdown("### ☁️ 내 자산 포트폴리오")

    st.caption("네이버페이(국내) / 1분봉(해외) 실시간 기반 | ATR 기반 리스크 관리")



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

        except Exception:

            pf_data = []



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

                            "qty": input_qty,

                        })

                        doc_ref.set({'stocks': new_pf_data})

                        st.success("추가 완료!")

                        time.sleep(0.5)

                        st.rerun()



        st.divider()



        if pf_data:

            # 수정 섹션

            st.markdown("#### ✏️ 보유 종목 정보 수정")

            edit_options = [f"{TICKER_MAP.get(p['ticker'], p['ticker'])} ({p['ticker']})" for p in pf_data]

            selected_edit = st.selectbox("수정할 종목 선택", options=["선택하세요"] + edit_options, key="edit_select")



            if selected_edit != "선택하세요":

                edit_ticker = selected_edit.split("(")[-1].rstrip(")")

                target = next((p for p in pf_data if p["ticker"] == edit_ticker), None)

                if target:

                    new_avg = st.number_input(

                        "새 평단가",

                        min_value=0.0,

                        value=float(target["price"]),

                        format="%.4f",

                        key="edit_avg_price",

                    )

                    new_qty = st.number_input(

                        "새 보유 수량(주)",

                        min_value=0,

                        value=int(target.get("qty", 1)),

                        key="edit_qty",

                    )



                    if st.button("변경 내용 저장", type="primary", key="edit_save"):

                        new_pf_data = []

                        for p in pf_data:

                            if p["ticker"] == edit_ticker:

                                new_pf_data.append({"ticker": edit_ticker, "price": new_avg, "qty": new_qty})

                            else:

                                new_pf_data.append(p)

                        doc_ref.set({"stocks": new_pf_data})

                        st.success("수정 완료!")

                        time.sleep(0.5)

                        st.rerun()



            st.divider()



        if pf_data:

            st.subheader(f"{user_id}님의 보유 종목 진단 (AI 스나이퍼 기준)")

            my_tickers = [p['ticker'] for p in pf_data]

            with st.spinner("초정밀 실시간 데이터 수집 중..."):

                raw_data_dict, realtime_map = get_precise_data(my_tickers)



            display_list = []

            for item in pf_data:

                tk = item['ticker']

                avg = item['price']

                qty = item.get('qty', 1)

                name = TICKER_MAP.get(tk, tk)



                curr = 0.0

                df_indi = None



                if tk in raw_data_dict:

                    df_tk = raw_data_dict[tk].dropna(how='all')

                    if not df_tk.empty:

                        curr_price = realtime_map.get(tk)

                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)



                if df_indi is not None and not df_indi.empty:

                    curr = float(df_indi['Close_Calc'].iloc[-1])

                    cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)

                else:

                    curr = avg 

                    cat, col_name, reasoning, score = "데이터 로딩 중", "gray", "잠시 후 다시 시도", 0.0



                if curr > 0:

                    res = calculate_total_profit(tk, avg, curr, qty)

                    display_list.append({

                        "name": name,

                        "tk": tk,

                        "avg": avg,

                        "curr": curr,

                        "qty": qty,

                        "cat": cat,

                        "col_name": col_name,

                        "reasoning": reasoning,

                        "profit_pct": res['pct'],

                        "profit_amt": res['profit_amt'],

                        "eval_amt": res['net_eval_amt'],

                        "currency": res['currency'],

                        "score": score,

                    })



            display_list.sort(key=lambda x: x['score'], reverse=True)



            for item in display_list:

                with st.container():

                    c1, c2, c3 = st.columns([1.5, 1.5, 4])

                    sym = item['currency']

                    safe_sym = sym if sym != "$" else "&#36;"



                    with c1:

                        st.markdown(f"### {item['name']}")

                        st.caption(f"{item['tk']} | 보유: {item['qty']}주")



                    with c2:

                        fmt_curr = f"{item['curr']:,.0f}" if item['currency'] == "₩" else f"{item['curr']:,.2f}"

                        fmt_avg = f"{item['avg']:,.0f}" if item['currency'] == "₩" else f"{item['avg']:,.2f}"

                        fmt_eval = f"{item['eval_amt']:,.0f}" if item['currency'] == "₩" else f"{item['eval_amt']:,.2f}"



                        st.metric(

                            "총 순수익",

                            f"{item['profit_pct']:.2f}%",

                            delta=f"{sym}{item['profit_amt']:,.0f}" if sym == "₩"

                            else f"{sym}{item['profit_amt']:,.2f}",

                        )

                        st.markdown(

                            f"<small style='color: gray'>평단: {safe_sym}{fmt_avg} / 현재: {safe_sym}{fmt_curr}</small>",

                            unsafe_allow_html=True,

                        )



                    with c3:

                        st.markdown(f"**AI 점수: {item['score']:.1f}점**")

                        st.markdown(f"**판단:** :{item['col_name']}[{item['cat']}]")

                        st.info(f"💡 {item['reasoning']}")

                    st.divider()



            if st.button("🗑️ 포트폴리오 전체 삭제"):

                doc_ref.delete()

                st.rerun()



# TAB 3: 알고리즘 백서

with tab3:

    st.markdown("## 📘 AI 투자 전략 알고리즘 백서 (Ver. Sniper)")

    st.markdown("""

본 서비스는 **'AI 스나이퍼 전략'**을 기반으로 종목을 분석합니다.

단순한 지표의 나열이 아닌, **ATR(변동성)**과 **이격도(Disparity)**를 결합하여 

'확실한 추세' 속의 '안전한 눌림목'을 찾아냅니다.

""")



    st.divider()



    st.subheader("1. 🎯 AI 종합 점수 가이드 (Scoring Guide)")

    score_guide_data = [

        {"점수 구간": "75점 ~ 100점", "등급": "🚀 강력 매수 (Sniper Entry)", "설명": "상승 추세 + 완벽한 눌림목 + 변동성 안정. 스나이퍼 전략의 핵심 진입 구간."},

        {"점수 구간": "60점 ~ 74점", "등급": "📈 매수 우위 (Good)", "설명": "상승 추세이나, 눌림목 위치가 애매하거나 단기 모멘텀이 부족함."},

        {"점수 구간": "40점 ~ 59점", "등급": "👀 관망 (Hold)", "설명": "방향성이 불분명하거나, 쉬어가는 구간. 신규 진입 보류."},

        {"점수 구간": "0점 ~ 39점", "등급": "💥 매도/회피 (Exit)", "설명": "하락 추세 전환, 과열(이격도 110%↑), 또는 고변동성 리스크 발생."},

    ]

    st.table(score_guide_data)



    st.header("2. 🧠 5대 핵심 분석 로직")



    with st.expander("① 추세 (Trend) - 60일선 & 120일선의 조화", expanded=True):

        st.markdown("""

**"추세가 꺾이면 모든 기법은 무용지물이다."**

- **MA60 기울기:** 60일선이 우상향 중인가? (+10점)

- **가격 위치:** 현재가가 60일선 위에 있는가? (+5점)

- **장기 추세:** 120일선까지 우상향이면 대세 상승장으로 간주 (+5점)

""")



    with st.expander("② 눌림목 & 과열 방지 (Disparity & Slope)", expanded=True):

        st.markdown("""

**"달리는 말에 타되, 잠시 멈췄을 때 타라."**

- **MA20 기울기:** 20일선이 상승 중일 때만 진입을 고려합니다.

- **눌림목 보너스:** 가격이 MA20 위에 있으면서 MA5 근처까지 내려왔을 때(건강한 조정) 가산점 부여.

- **이격도 과열 필터:** MA20 대비 **110% 이상 급등**하면 즉시 -20점 페널티를 부여하여 추격 매수를 원천 차단합니다.

""")



    with st.expander("③ ATR (Average True Range) - 변동성 통제", expanded=True):

        st.markdown("""

**"감당할 수 있는 흔들림인가?"**

- 단순히 많이 오른다고 좋은 것이 아닙니다.

- **ATR(변동폭) / 주가 비율**이 5%를 넘어가면 '고위험군'으로 분류하여 매수 추천에서 제외합니다.

- 스나이퍼 전략은 변동성이 안정된 상태에서의 꾸준한 우상향을 목표로 합니다.

""")



    with st.expander("④ 보조지표 (MACD & RSI)", expanded=True):

        st.markdown("""

- **MACD 히스토그램:** 어제보다 오늘 상승 에너지가 강해졌는가? (가속도 체크)

- **RSI (40~60):** 과열(70↑)도 아니고 침체(30↓)도 아닌, 가장 안정적으로 상승하는 '허리' 구간을 선호합니다.

- **볼린저 밴드:** 하단 밴드를 터치하고 반등할 때 기술적 반등 점수를 부여합니다.

""")
