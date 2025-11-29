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
import re  # 국내 애프터마켓 가격 파싱용

# ---------------------------------------------------------
# 0. 파이어베이스(DB) 설정
# ---------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, firestore

def _now_kst():
    """UTC 기준 현재 시간을 KST(UTC+9)로 변환."""
    now_utc = datetime.datetime.utcnow()
    return now_utc + datetime.timedelta(hours=9)

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

# 스캔 결과 영구 보존을 위한 세션 초기화
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
# 2. 데이터 수집 혁신 (New Method)
# ---------------------------------------------------------

def fetch_kr_polling(ticker):
    """국내 주식 실시간/시간외 가격 (네이버 API)"""
    code = ticker.split('.')[0]
    try:
        url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.naver.com/"
        }
        res = requests.get(url, headers=headers, timeout=3)
        res.raise_for_status()
        data = res.json()
        datas = data.get("datas", [])
        if not datas: raise ValueError("no datas")

        item = datas[0]
        
        # 1. 가격 파싱
        over_info = item.get("overMarketPriceInfo") or {}
        over_price_str = str(over_info.get("overPrice", "")).replace(",", "").strip()
        close_price_str = str(item.get("closePrice", "")).replace(",", "").strip()

        over_price = float(over_price_str) if over_price_str not in ("", "0") else None
        close_price = float(close_price_str) if close_price_str not in ("", "0") else None

        # 2. 시간 파싱
        def _parse_dt(s):
            try: return datetime.datetime.fromisoformat(s) if s else None
            except: return None
        
        base_time = _parse_dt(item.get("localTradedAt", ""))
        over_time = _parse_dt(over_info.get("localTradedAt", ""))

        # 3. 최신 가격 선택
        chosen_price = None
        chosen_time = None

        if close_price is not None:
            chosen_price, chosen_time = close_price, base_time
        
        if over_price is not None:
            if over_time and chosen_time:
                if over_time > chosen_time:
                    chosen_price, chosen_time = over_price, over_time
            elif chosen_price is None:
                chosen_price, chosen_time = over_price, over_time

        if chosen_price is not None:
            return (ticker, float(chosen_price))
        
        raise ValueError("no usable price")

    except Exception:
        # 실패 시 FDR 종가 폴백
        try:
            df = fdr.DataReader(code, "2023-01-01")
            if not df.empty:
                return (ticker, float(df["Close"].iloc[-1]))
        except:
            pass
        return (ticker, None)

def fetch_us_1m_candle(ticker):
    """미국 주식 1분봉(장전/장후 포함)"""
    try:
        df = yf.download(ticker, period="5d", interval="1m", prepost=True, progress=False)
        if not df.empty:
            return (ticker, float(df['Close'].iloc[-1]))
        return (ticker, None)
    except:
        return (ticker, None)

def fetch_history_data(ticker):
    """지표 분석용 일봉 데이터 (정규장 종가 기준)"""
    try:
        if ticker.endswith('.KS') or ticker.endswith('.KQ'):
            df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        else:
            df = yf.download(ticker, period="2y", interval="1d", progress=False, prepost=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
        return (ticker, df)
    except:
        return (ticker, None)

@st.cache_data(ttl=0)
def get_precise_data(tickers_list):
    """실시간 가격과 일봉 히스토리를 병렬 수집"""
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]

    realtime_prices = {}
    hist_map = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        fut_real = []
        for t in kr_tickers: fut_real.append(executor.submit(fetch_kr_polling, t))
        for t in us_tickers: fut_real.append(executor.submit(fetch_us_1m_candle, t))
        fut_hist = [executor.submit(fetch_history_data, t) for t in tickers_list]

        for f in concurrent.futures.as_completed(fut_real):
            tk, p = f.result()
            if p is not None: realtime_prices[tk] = p

        for f in concurrent.futures.as_completed(fut_hist):
            tk, df = f.result()
            if df is not None and not df.empty:
                # 데이터 전처리
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.sort_index()
                hist_map[tk] = df

    return hist_map, realtime_prices

def calculate_indicators(df):
    """기술적 지표 계산 (MACD, RSI, Boll, MA 등)"""
    if len(df) < 60: return None
    df = df.copy()

    # Close 처리
    if 'Close' in df.columns:
        close = df['Close']
        close_series = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    else:
        return None
    
    close_series = close_series.ffill()
    df['Close_Calc'] = close_series

    # MA
    df['MA5'] = df['Close_Calc'].rolling(window=5).mean()
    df['MA10'] = df['Close_Calc'].rolling(window=10).mean()
    df['MA20'] = df['Close_Calc'].rolling(window=20).mean()
    df['MA60'] = df['Close_Calc'].rolling(window=60).mean()

    # Volatility / Momentum
    df['STD20'] = df['Close_Calc'].rolling(window=20).std()
    df['RET1'] = df['Close_Calc'].pct_change()
    df['MOM10'] = df['Close_Calc'] / df['Close_Calc'].shift(10) - 1

    # Volume
    if 'Volume' in df.columns:
        vol = df['Volume']
        if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]
        df['Volume_Calc'] = vol
        df['VolMA20'] = vol.rolling(window=20).mean()
    else:
        df['Volume_Calc'] = 0
        df['VolMA20'] = 0

    # RSI
    delta = df['Close_Calc'].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)

    return df.dropna()

def calculate_total_profit(ticker, avg_price, current_price, quantity):
    """순수익 계산 (토스증권 수수료 체계 반영)"""
    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
    qty = float(quantity)
    avg_price = float(avg_price)
    current_price = float(current_price)

    total_buy = avg_price * qty
    gross_eval = current_price * qty

    if is_kr:
        fee_rate = 0.000295
        tax_rate = 0.0015
    else:
        fee_rate = 0.001965
        tax_rate = 0.0

    sell_fee = gross_eval * fee_rate
    sell_tax = gross_eval * tax_rate

    net_eval = gross_eval - sell_fee - sell_tax
    net_profit_amt = net_eval - total_buy
    
    net_profit_pct = (net_profit_amt / total_buy) * 100 if total_buy > 0 else 0.0
    currency = "₩" if is_kr else "$"

    return {
        "pct": net_profit_pct,
        "profit_amt": net_profit_amt,
        "net_eval_amt": net_eval,
        "currency": currency
    }

def analyze_advanced_strategy(df, curr_override=None):
    """
    [핵심 AI 엔진] 2~4주 스윙 전략 스코어링
    Scanner와 Portfolio 양쪽에서 동일하게 사용됨.
    """
    if df is None or df.empty:
        return "분석 불가", "gray", "데이터 부족", 0

    try:
        # 기본값: 일봉 종가
        curr = float(df['Close_Calc'].iloc[-1])
        
        # 🔑 실시간 가격(curr_override)이 있으면 최우선 적용 (포트폴리오 로직)
        if curr_override is not None and curr_override > 0:
            curr = float(curr_override)

        ma5  = float(df['MA5'].iloc[-1])
        ma10 = float(df['MA10'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        ma60 = float(df['MA60'].iloc[-1])

        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        sig  = float(df['Signal_Line'].iloc[-1])
        prev_macd = float(df['MACD'].iloc[-2])
        prev_sig  = float(df['Signal_Line'].iloc[-2])

        std20 = float(df['STD20'].iloc[-1])
        mom10 = float(df['MOM10'].iloc[-1]) if 'MOM10' in df.columns else 0.0

        vol    = float(df['Volume_Calc'].iloc[-1]) if 'Volume_Calc' in df.columns else 0.0
        vol_ma = float(df['VolMA20'].iloc[-1]) if 'VolMA20' in df.columns else 0.0
        prev_close = float(df['Close_Calc'].iloc[-2])
    except Exception:
        return "데이터 오류", "gray", "지표 계산 실패", 0

    score = 50
    reasons = []

    # 1) 추세 필터
    if curr > ma60 and ma20 > ma60:
        score += 20
        reasons.append("📈 중기 상승 추세 (60일선 위)")
    elif curr > ma60:
        score += 5
        reasons.append("↗ 60일선 위 (추세 형성 중)")
    else:
        score -= 25
        reasons.append("⚠ 하락 추세 (60일선 아래)")

    # 2) 위치 (눌림목)
    dist_ma20 = (curr - ma20) / ma20 if ma20 > 0 else 0
    
    if (curr >= ma20) and (curr >= ma60) and (-0.03 <= dist_ma20 <= 0.02):
        score += 20
        reasons.append("💎 황금 눌림목 (20일선 근접)")
    elif 0.02 < dist_ma20 <= 0.07:
        score += 5
        reasons.append("🙂 상승 유지 (과열 아님)")
    elif dist_ma20 > 0.07:
        score -= 15
        reasons.append("🔥 단기 과열 (20일선 이격 과다)")

    # 3) RSI
    if 40 <= rsi <= 60:
        score += 15
        reasons.append("⚖ RSI 균형 (스윙 적합)")
    elif 30 <= rsi < 40:
        score += 5
        reasons.append("반등 기대 (약한 과매도)")
    elif rsi < 30:
        score += 15
        reasons.append("심한 과매도 (역발상 기회)")
    elif rsi > 70:
        score -= 20
        reasons.append("🚨 RSI 과열 (조정 주의)")

    # 4) 모멘텀
    if 0.03 <= mom10 <= 0.15:
        score += 10
        reasons.append("📊 건강한 상승 모멘텀")
    elif mom10 > 0.25:
        score -= 15
        reasons.append("급등 피로감 (차익 실현 주의)")
    elif mom10 < -0.10:
        score -= 10
        reasons.append("낙폭 과대")

    # 5) MACD
    if macd > sig and prev_macd <= prev_sig:
        score += 15
        reasons.append("⚡ MACD 골든크로스")
    elif macd > sig:
        score += 5
        reasons.append("MACD 상방")
    elif macd < sig and prev_macd >= prev_sig:
        score -= 10
        reasons.append("💧 MACD 데드크로스")

    # 6) 변동성 & 거래량
    vol_ratio = std20 / curr if curr > 0 else 0
    if vol_ratio > 0.08:
        score -= 15
        reasons.append("🎢 변동성 매우 큼")
    elif vol_ratio < 0.03:
        score += 5
        reasons.append("⚙ 안정적 변동성")

    if vol_ma > 0 and vol > vol_ma * 1.5 and curr > prev_close:
        score += 10
        reasons.append("🔥 거래량 실린 상승")

    score = max(0, min(100, score))

    if score >= 80:
        category = "🚀 단기 강력 매수"
        color_name = "green"
    elif score >= 65:
        category = "📈 매수 우위"
        color_name = "blue"
    elif score >= 45:
        category = "👀 관망"
        color_name = "gray"
    elif score >= 25:
        category = "📉 매도/비중 축소"
        color_name = "red"
    else:
        category = "💥 강력 매도"
        color_name = "red"

    if not reasons:
        reasons.append("관망 (특이사항 없음)")

    summary = " / ".join(reasons[:4])
    return category, color_name, summary, score

# ---------------------------------------------------------
# 4. UI
# ---------------------------------------------------------
st.title("📈 AI 주식 스캐너 & 포트폴리오 Pro")

tab1, tab2, tab3 = st.tabs(["🚀 전체 종목 스캐너", "💼 내 포트폴리오 (서버 저장)", "📘 알고리즘 설명서"])

# =========================================================
# TAB 1: 스캐너 (수정됨: 포트폴리오 로직 적용)
# =========================================================
with tab1:
    st.markdown("### 📋 AI 정밀 스캐너")
    st.caption("포트폴리오와 동일한 정밀 알고리즘 적용 (실시간/AfterMarket)")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 분석 새로고침", type="primary"):
            st.session_state['scan_result_df'] = None 
            st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 리스트 정밀 분석 시작"):
            with st.spinner('초정밀 데이터 수집 및 AI 분석 중... (15~20초 소요)'):
                # 1. 데이터 수집 (포트폴리오와 동일 함수 사용)
                raw_data_dict, realtime_map = get_precise_data(USER_WATCHLIST)
                scan_results = []
                progress_bar = st.progress(0)
                
                for i, ticker_code in enumerate(USER_WATCHLIST):
                    if ticker_code not in raw_data_dict: continue
                    try:
                        df_tk = raw_data_dict[ticker_code].dropna(how='all')
                        if df_tk.empty: continue
                        
                        df_indi = calculate_indicators(df_tk)
                        if df_indi is None: continue

                        # -------------------------------------------------
                        # ⚡ 포트폴리오와 완벽히 동일한 가격 로직 적용 (수정됨)
                        # -------------------------------------------------
                        curr_price = 0
                        
                        # 1순위: 실시간/애프터마켓 데이터
                        if ticker_code in realtime_map:
                            curr_price = float(realtime_map[ticker_code])
                        # 2순위: 실시간 실패 시 일봉 종가
                        elif df_indi is not None and not df_indi.empty:
                            curr_price = float(df_indi['Close_Calc'].iloc[-1])

                        # AI 엔진 호출 (포트폴리오와 동일 인자 전달)
                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi, curr_override=curr_price)

                        # 결과 정리
                        rsi_val = float(df_indi['RSI'].iloc[-1])
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
                    st.success("완료! 포트폴리오와 동일한 로직으로 분석되었습니다.")
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

# =========================================================
# TAB 2: 포트폴리오 (기준 로직)
# =========================================================
with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오")
    st.caption("네이버페이(국내) / 1분봉(해외) 실시간 기반 | 세후 순수익 계산")
    
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
            # ✏️ 보유 종목 정보 수정 섹션
            st.markdown("#### ✏️ 보유 종목 정보 수정")
            edit_options = [f"{TICKER_MAP.get(p['ticker'], p['ticker'])} ({p['ticker']})" for p in pf_data]
            selected_edit = st.selectbox("수정할 종목 선택", options=["선택하세요"] + edit_options, key="edit_select")

            if selected_edit != "선택하세요":
                edit_ticker = selected_edit.split("(")[-1].rstrip(")")
                target = next((p for p in pf_data if p["ticker"] == edit_ticker), None)
                if target:
                    new_avg = st.number_input("새 평단가", min_value=0.0, value=float(target["price"]), format="%.4f", key="edit_avg_price")
                    new_qty = st.number_input("새 보유 수량(주)", min_value=0, value=int(target.get("qty", 1)), key="edit_qty")

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
            st.subheader(f"{user_id}님의 보유 종목 진단")
            my_tickers = [p['ticker'] for p in pf_data]
            with st.spinner("초정밀 실시간 데이터 수집 중..."):
                raw_data_dict, realtime_map = get_precise_data(my_tickers)
            
            display_list = []
            for item in pf_data:
                tk = item['ticker']
                avg = item['price']
                qty = item.get('qty', 1)
                name = TICKER_MAP.get(tk, tk)
                
                df_indi = None
                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                    if not df_tk.empty:
                        df_indi = calculate_indicators(df_tk)

                # -------------------------------------------------
                # ⚡ 포트폴리오 가격 로직 (Scanner와 동일)
                # -------------------------------------------------
                curr = 0
                if tk in realtime_map:
                    curr = float(realtime_map[tk])
                elif df_indi is not None and not df_indi.empty:
                    curr = float(df_indi['Close_Calc'].iloc[-1])

                cat, col_name, reasoning, score = "데이터 로딩 중", "gray", "잠시 후 다시 시도", 0

                if df_indi is not None:
                    cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi, curr_override=curr)
                
                if curr > 0:
                    res = calculate_total_profit(tk, avg, curr, qty)
                    display_list.append({
                        "name": name, "tk": tk, "avg": avg, "curr": curr, "qty": qty,
                        "cat": cat, "col_name": col_name, "reasoning": reasoning,
                        "profit_pct": res['pct'], "profit_amt": res['profit_amt'],
                        "eval_amt": res['net_eval_amt'], "currency": res['currency'], "score": score
                    })
                else:
                    display_list.append({
                        "name": name, "tk": tk, "avg": avg, "curr": avg, "qty": qty,
                        "cat": "로딩 실패", "col_name": "gray", "reasoning": "데이터 수신 불가",
                        "profit_pct": 0.0, "profit_amt": 0.0, "eval_amt": 0.0,
                        "currency": "$" if not tk.endswith(".KS") else "₩", "score": 0
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
                        fmt_avg  = f"{item['avg']:,.0f}"  if item['currency'] == "₩" else f"{item['avg']:,.2f}"
                        fmt_eval = f"{item['eval_amt']:,.0f}"   if item['currency'] == "₩" else f"{item['eval_amt']:,.2f}"
                        
                        st.metric("총 순수익 (수수료 제)", f"{item['profit_pct']:.2f}%", delta=f"{sym}{item['profit_amt']:,.0f}" if sym=="₩" else f"{sym}{item['profit_amt']:,.2f}")
                        st.markdown(f"**세후 총 평가금:** {safe_sym}{fmt_eval}", unsafe_allow_html=True)
                        st.markdown(f"<small style='color: gray'>평단: {safe_sym}{fmt_avg} / 현재: {safe_sym}{fmt_curr}</small>", unsafe_allow_html=True)
                        
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
    st.markdown("""
    본 서비스에 탑재된 AI 알고리즘은 **'추세 추종(Trend Following)'** 전략과 **'평균 회귀(Mean Reversion)'** 이론을 
    결합하여 설계되었습니다. 스캐너와 포트폴리오 탭 모두 동일한 로직을 사용하여 점수를 계산합니다.
    """)
    st.divider()
    st.markdown("...(이전과 동일한 설명 내용)...")
