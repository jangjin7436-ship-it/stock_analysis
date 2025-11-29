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
        except Exception as e:
            return None
    return firestore.client()

# ---------------------------------------------------------
# 1. 설정 및 매핑
# ---------------------------------------------------------
st.set_page_config(page_title="AI 스나이퍼 스캐너", page_icon="🎯", layout="wide")

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
    if not us_tickers:
        return {}, {}
    
    hist_map = {}
    realtime_map = {}

    # 1개일 때
    if len(us_tickers) == 1:
        ticker = us_tickers[0]
        try:
            df_hist = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            if not df_hist.empty:
                if 'Close' in df_hist.columns:
                    hist_map[ticker] = df_hist

            df_real = yf.download(ticker, period="5d", interval="1m", progress=False, prepost=True)
            if not df_real.empty:
                if 'Close' in df_real.columns:
                    last_p = float(df_real['Close'].iloc[-1])
                    realtime_map[ticker] = last_p
        except:
            pass
        return hist_map, realtime_map

    # 여러 개일 때 (Bulk)
    try:
        df_hist = yf.download(us_tickers, period="2y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
        df_real = yf.download(us_tickers, period="5d", interval="1m", progress=False, group_by='ticker', prepost=True)

        for t in us_tickers:
            try:
                sub_df = df_hist[t]
                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                    sub_df = sub_df.dropna(how='all') 
                    if 'Close' in sub_df.columns:
                        hist_map[t] = sub_df
            except: 
                pass

            try:
                sub_real = df_real[t]
                if isinstance(sub_real, pd.DataFrame) and not sub_real.empty:
                    sub_real = sub_real.dropna(how='all')
                    if 'Close' in sub_real.columns:
                        valid_closes = sub_real['Close'].dropna()
                        if not valid_closes.empty:
                            realtime_map[t] = float(valid_closes.iloc[-1])
            except: 
                pass
    except:
        pass

    return hist_map, realtime_map

def fetch_kr_polling(ticker):
    code = ticker.split('.')[0]
    try:
        url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=2)
        data = res.json()
        item = data['datas'][0]
        
        close = float(str(item['closePrice']).replace(',', ''))
        
        over_info = item.get('overMarketPriceInfo', {})
        over_price_str = str(over_info.get('overPrice', '')).replace(',', '').strip()
        if over_price_str and over_price_str != '0':
            return (ticker, float(over_price_str))
            
        return (ticker, close)
    except:
        return (ticker, None)

def fetch_kr_history(ticker):
    try:
        df = fdr.DataReader(ticker.split('.')[0], '2023-01-01')
        return (ticker, df)
    except:
        return (ticker, None)

@st.cache_data(ttl=0)
def get_precise_data(tickers_list):
    if not tickers_list:
        return {}, {}
        
    kr_tickers = [t for t in tickers_list if t.endswith('.KS') or t.endswith('.KQ')]
    us_tickers = [t for t in tickers_list if t not in kr_tickers]

    hist_map, realtime_map = get_bulk_us_data(us_tickers)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        fut_real = [executor.submit(fetch_kr_polling, t) for t in kr_tickers]
        fut_hist = [executor.submit(fetch_kr_history, t) for t in kr_tickers]

        for f in concurrent.futures.as_completed(fut_real):
            try:
                tk, p = f.result()
                if p: realtime_map[tk] = p
            except: pass
            
        for f in concurrent.futures.as_completed(fut_hist):
            try:
                tk, df = f.result()
                if df is not None and not df.empty:
                    hist_map[tk] = df
            except: pass

    return hist_map, realtime_map

# ---------------------------------------------------------
# 3. 분석 엔진
# ---------------------------------------------------------

def calculate_indicators(df, realtime_price=None):
    if df is None or len(df) < 60:
        return None

    df = df.copy()
    
    # [차원 오류 방지] 중복 제거 및 시리즈 변환
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[~df.index.duplicated(keep='last')]

    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    if 'Close' not in df.columns:
        return None

    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # 실시간 가격 주입
    if realtime_price is not None and realtime_price > 0:
        try:
            close.iloc[-1] = realtime_price
        except Exception:
            pass

    df['Close_Calc'] = close

    # 지표 계산
    df['MA5'] = df['Close_Calc'].rolling(5).mean()
    df['MA10'] = df['Close_Calc'].rolling(10).mean() 
    df['MA20'] = df['Close_Calc'].rolling(20).mean()
    df['MA60'] = df['Close_Calc'].rolling(60).mean()
    
    std = df['Close_Calc'].rolling(20).std()
    df['Upper_Band'] = df['MA20'] + (std * 2)
    df['Lower_Band'] = df['MA20'] - (std * 2)
    df['Band_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
    
    delta = df['Close_Calc'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close_Calc'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close_Calc'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)
    
    if 'Volume' in df.columns:
        vol = df['Volume']
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        df['Vol_MA20'] = vol.rolling(20).mean()
        denom = df['Vol_MA20'].replace(0, np.nan)
        df['Vol_Ratio'] = vol / denom
        df['Vol_Ratio'] = df['Vol_Ratio'].fillna(0)
    else:
        df['Vol_Ratio'] = 1.0 

    df['STD20'] = std
    # 최종 결과 중복 컬럼 제거
    return df.loc[:, ~df.columns.duplicated()].dropna()

def get_scalar(val):
    try:
        if isinstance(val, (pd.Series, pd.DataFrame)):
            if val.empty: return 0.0
            return float(val.iloc[0])
        return float(val)
    except:
        return 0.0

def get_ai_score_row(row):
    """AI 스나이퍼 스코어링"""
    try:
        score = 50.0
        curr = get_scalar(row['Close_Calc'])
        ma5  = get_scalar(row['MA5'])
        ma10 = get_scalar(row['MA10'])
        ma20 = get_scalar(row['MA20'])
        ma60 = get_scalar(row['MA60'])
        rsi  = get_scalar(row['RSI'])
        macd_hist = get_scalar(row['MACD_Hist'])
        prev_hist = get_scalar(row['Prev_MACD_Hist'])
        u_band    = get_scalar(row['Upper_Band'])
        band_width= get_scalar(row['Band_Width'])
        vol_ratio = get_scalar(row['Vol_Ratio'])
        std20     = get_scalar(row['STD20'])
        
        # 1. 추세
        if curr > ma10:
            score += 15.0
            if ma5 > ma10 > ma20: score += 5.0
        else:
            score -= 10.0
        if curr > ma60: score += 5.0
        else: score -= 5.0

        # 2. 모멘텀
        if macd_hist > 0:
            score += 5.0
            if macd_hist > prev_hist: score += 5.0
        elif macd_hist > prev_hist and macd_hist > -0.5:
             score += 5.0

        # 3. RSI
        if 50 <= rsi <= 70: score += 10.0
        elif rsi > 75: score -= 5.0
        elif rsi < 35: score += 5.0

        # 4. 볼린저 밴드
        if curr >= u_band * 0.98: score += 10.0
        if band_width < 0.15 and ma5 > ma10: score += 5.0

        # 5. 거래량
        if vol_ratio >= 1.2 and curr > ma5: score += 5.0

        # 6. 안정성
        v_ratio = std20 / curr if curr > 0 else 0
        score -= (v_ratio * 100.0)

        return max(0.0, min(100.0, score))
    except:
        return 0.0

def analyze_advanced_strategy(df):
    """
    [AI 스나이퍼 매수 진입 판단]
    """
    if df is None or df.empty:
        return "분석 불가", "gray", "데이터 부족", 0.0

    try:
        row = df.iloc[-1]
        score = get_ai_score_row(row)
        
        curr = get_scalar(row['Close_Calc'])
        ma10 = get_scalar(row['MA10'])
        ma60 = get_scalar(row['MA60'])
        rsi  = get_scalar(row['RSI'])
        macd_hist = get_scalar(row['MACD_Hist'])
        prev_hist = get_scalar(row['Prev_MACD_Hist'])
        u_band    = get_scalar(row['Upper_Band'])
        
    except Exception:
        return "오류", "gray", "계산 실패", 0.0

    reasons = []
    if curr > ma10: reasons.append("10일선 위")
    else: reasons.append("10일선 이탈")
    if curr >= u_band * 0.99: reasons.append("밴드 돌파")
    elif curr > ma60: reasons.append("정배열")
    if rsi > 75: reasons.append("과열권")
    elif rsi < 35: reasons.append("과매도")
    if macd_hist > 0 and macd_hist > prev_hist: reasons.append("에너지 가속")

    # 등급 산정
    if score >= 70:
        if rsi > 75:
            cat = "🔥 매수 주의 (과열권)"
            col = "orange"
            reasons.insert(0, "단기 고점 위험")
        else:
            cat = "🎯 스나이퍼 매수 (진입 타점)"
            col = "green"
    elif score < 40:
        cat = "💥 매도/손절 (추세 이탈)"
        col = "red"
    elif 40 <= score < 50:
        cat = "📉 비중 축소 (약세)"
        col = "orange"
    else: 
        cat = "👀 관망 (Hold)"
        col = "blue"

    reasoning = " / ".join(reasons[:3])
    return cat, col, reasoning, round(score, 3)

# ---------------------------------------------------------
# 매도/홀딩 판단 로직
# ---------------------------------------------------------
def get_sell_advice(df, buy_price, buy_date_str):
    if df is None or df.empty:
        return "분석 대기", "gray", "데이터 부족"

    try:
        row = df.iloc[-1]
        curr_price = get_scalar(row['Close_Calc']) 
        score = get_ai_score_row(row) 
        
        buy_date = pd.to_datetime(buy_date_str).date()
        today = datetime.date.today()
        held_days = (today - buy_date).days
        
        df_held = df[df.index.date >= buy_date]
        
        if not df_held.empty:
            max_price_since_buy = df_held['Close_Calc'].max()
        else:
            max_price_since_buy = curr_price 

        cur_profit_pct = (curr_price - buy_price) / buy_price * 100
        max_profit_pct = (max_price_since_buy - buy_price) / buy_price * 100
        
        if max_price_since_buy > 0:
            drawdown_from_peak = (curr_price - max_price_since_buy) / max_price_since_buy
        else:
            drawdown_from_peak = 0.0

    except Exception as e:
        return "계산 오류", "gray", f"날짜/가격 정보 확인 필요 ({e})"

    # 판단 로직
    if cur_profit_pct <= -3.0:
        return "⚡ 칼손절 (-3%)", "red", f"손절 원칙 도달(현재 {cur_profit_pct:.1f}%). 즉시 자르세요."

    if held_days >= 14:
        return "⏱️ 타임컷 매도", "orange", f"보유 14일 경과(현재 {held_days}일). 원칙대로 전량 매도."

    if max_profit_pct >= 5.0:
        if drawdown_from_peak <= -0.03:
            return "📉 트레일링 익절", "orange", f"최고점({max_profit_pct:.1f}%) 찍고 -3% 하락. 이익 확정하세요."
        else:
            return "💎 슈퍼 홀딩 (Riding)", "green", f"수익 극대화 중! (현재 +{cur_profit_pct:.1f}% / 고점 대비 {drawdown_from_peak*100:.1f}%)"

    if score < 40:
        return "📉 추세 이탈", "red", f"AI 점수 급락({score:.0f}점). 상승 동력 상실."

    return "⏳ 홀딩 (Waiting)", "blue", f"목표 +5% 대기 중. (현재 {cur_profit_pct:.1f}% / 보유 {held_days}일)"

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
        "pct": pct, "profit_amt": net_profit, 
        "net_eval_amt": net_eval, "currency": "₩" if is_kr else "$"
    }

# ---------------------------------------------------------
# 4. UI
# ---------------------------------------------------------
st.title("🎯 AI 스나이퍼 스캐너 Pro")

tab1, tab2, tab3 = st.tabs(["🚀 AI 스나이퍼 스캔", "💼 내 포트폴리오(매도 알림)", "📘 전략 백서"])

# TAB 1: 스캐너
with tab1:
    st.markdown("### 📋 AI 스나이퍼 종목 발굴")
    st.caption("전략: 2주 단기 스윙 | 선정: 조건 만족(70점↑) 종목 전부 매수 (분산 투자 권장)")

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 시장 정밀 스캔", type="primary"):
            st.session_state['scan_result_df'] = None 
            st.rerun()

    if st.session_state['scan_result_df'] is None:
        if st.button("🔍 전체 유니버스 분석 시작"):
            with st.spinner('AI 스나이퍼 알고리즘 가동 중... (10일선/볼린저/모멘텀 분석)'):
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
                        
                        if df_indi is None: 
                            continue

                        # AI 스나이퍼 분석 실행
                        cat, col_name, reasoning, score = analyze_advanced_strategy(df_indi)

                        final_price = float(df_indi['Close_Calc'].iloc[-1])
                        rsi_val = float(df_indi['RSI'].iloc[-1])
                        # ★ 정렬용 데이터
                        macd_hist_val = float(df_indi['MACD_Hist'].iloc[-1])
                        vol_ratio_val = float(df_indi['Vol_Ratio'].iloc[-1])

                        name = TICKER_MAP.get(ticker_code, ticker_code)
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
                            "MACD_Hist": macd_hist_val,
                            "Vol_Ratio": vol_ratio_val  # 거래량 비율 추가 (정렬용)
                        })
                    except: 
                        continue
                    progress_bar.progress((i + 1) / len(USER_WATCHLIST))
                
                if scan_results:
                    df_res = pd.DataFrame(scan_results)
                    df_res = df_res.sort_values('점수', ascending=False)
                    st.session_state['scan_result_df'] = df_res
                    st.success("스캔 완료! 70점 이상인 종목들을 확인하세요.")
                    st.rerun()
                else:
                    st.error("데이터 수집 실패.")
    
    if st.session_state['scan_result_df'] is not None:
        # ★ [오류 수정 핵심] 기존 세션 데이터에 MACD_Hist/Vol_Ratio가 없는 경우 자동 재설정
        if 'Vol_Ratio' not in st.session_state['scan_result_df'].columns:
            st.warning("⚠️ 데이터 업데이트가 필요하여 재스캔을 준비합니다...")
            st.session_state['scan_result_df'] = None
            time.sleep(1)
            st.rerun()
        
        else:
            # 기본 필터링: 70점 이상
            base_df = st.session_state['scan_result_df'][st.session_state['scan_result_df']['점수'] >= 70]
            
            # ★ 100점 만점 종목 과다 시 Top 5 추천 로직
            perfect_candidates = base_df[base_df['점수'] >= 100]
            
            display_df = base_df # 기본값
            
            if len(perfect_candidates) > 5:
                st.toast(f"💎 100점 만점 종목이 {len(perfect_candidates)}개 발견되었습니다!", icon="🔥")
                st.info(f"💡 **AI 추천:** 100점 종목이 너무 많아, 거래량 급증(Volume Ratio)이 가장 강력한 **상위 5개**를 엄선했습니다.")
                
                # 1. 100점짜리 중 Vol_Ratio(거래량 비율)가 높은 순으로 5개 선정
                top5_perfect = perfect_candidates.sort_values(by='Vol_Ratio', ascending=False).head(5)
                
                # 2. 100점 미만 70점 이상 종목들은 그대로 유지
                others = base_df[base_df['점수'] < 100]
                
                # 3. 데이터프레임 재구성
                display_df = pd.concat([top5_perfect, others])
                display_df = display_df.sort_values(by=['점수', 'Vol_Ratio'], ascending=[False, False])
            
            count = len(display_df)
            
            if count > 0:
                st.markdown(f"✨ **매수 추천 리스트 ({count}개)**")
            else:
                st.warning("현재 매수 조건을 만족하는 종목이 없습니다. (관망 권장)")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=700,
                column_config={
                    "종목명": st.column_config.TextColumn("종목명 (코드)", width="medium"),
                    "점수": st.column_config.ProgressColumn("AI 점수", format="%.1f점", min_value=0, max_value=100),
                    "현재가": st.column_config.TextColumn("현재가"), 
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                    "AI 등급": st.column_config.TextColumn("AI 판단"),
                    "핵심 요약": st.column_config.TextColumn("분석 내용", width="large"),
                    "MACD_Hist": st.column_config.NumberColumn("에너지(Momentum)", format="%.2f"),
                },
                hide_index=True
            )

# TAB 2: 포트폴리오
with tab2:
    st.markdown("### ☁️ 내 자산 포트폴리오 (매도 알리미)")
    st.caption("AI 스나이퍼 규칙: 2주 타임컷 / -3% 손절 / +5% 후 트레일링 익절")
    
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
        except: 
            pf_data = []

        with st.container():
            st.markdown("#### ➕ 종목 추가 (매수일 필수)")
            c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
            with c1:
                selected_item = st.selectbox("종목 검색", ["선택하세요"] + SEARCH_LIST)
            with c2:
                input_price = st.number_input("매수 단가", min_value=0.0, format="%.2f")
            with c3:
                input_date = st.date_input("매수 날짜", datetime.date.today())
            with c4:
                input_qty = st.number_input("수량", min_value=1, value=1)
            
            if st.button("포트폴리오에 추가", type="primary"):
                if selected_item != "선택하세요":
                    target_code = SEARCH_MAP[selected_item]
                    new_pf_data = [p for p in pf_data if p['ticker'] != target_code]
                    new_pf_data.append({
                        "ticker": target_code, 
                        "price": input_price,
                        "qty": input_qty,
                        "date": str(input_date) 
                    })
                    doc_ref.set({'stocks': new_pf_data})
                    st.success("추가 완료!")
                    time.sleep(0.5)
                    st.rerun()

        st.divider()

        if pf_data:
            # 수정 섹션
            with st.expander("✏️ 종목 수정/삭제"):
                edit_options = [f"{TICKER_MAP.get(p['ticker'], p['ticker'])} ({p['ticker']})" for p in pf_data]
                selected_edit = st.selectbox("수정할 종목", options=["선택하세요"] + edit_options)

                if selected_edit != "선택하세요":
                    edit_ticker = selected_edit.split("(")[-1].rstrip(")")
                    target = next((p for p in pf_data if p["ticker"] == edit_ticker), None)
                    if target:
                        c_e1, c_e2, c_e3 = st.columns(3)
                        with c_e1:
                            new_avg = st.number_input("수정 단가", value=float(target["price"]), format="%.2f")
                        with c_e2:
                            try:
                                def_date = pd.to_datetime(target.get("date", str(datetime.date.today()))).date()
                            except:
                                def_date = datetime.date.today()
                            new_date_val = st.date_input("수정 매수일", value=def_date)
                        with c_e3:
                            new_qty_val = st.number_input("수정 수량", value=int(target.get("qty", 1)))

                        if st.button("변경 저장", type="primary"):
                            new_pf_data = []
                            for p in pf_data:
                                if p["ticker"] == edit_ticker:
                                    new_pf_data.append({
                                        "ticker": edit_ticker, "price": new_avg, 
                                        "qty": new_qty_val, "date": str(new_date_val)
                                    })
                                else:
                                    new_pf_data.append(p)
                            doc_ref.set({"stocks": new_pf_data})
                            st.rerun()
        
            st.divider()
            
            st.subheader(f"📊 {user_id}님의 포트폴리오 진단")
            my_tickers = [p['ticker'] for p in pf_data]
            with st.spinner("AI가 스나이퍼 규칙을 대입 중..."):
                raw_data_dict, realtime_map = get_precise_data(my_tickers)
            
            display_list = []
            for item in pf_data:
                tk = item['ticker']
                avg = item['price']
                qty = item.get('qty', 1)
                b_date = item.get('date', str(datetime.date.today()))
                name = TICKER_MAP.get(tk, tk)
                
                curr = 0
                df_indi = None
                
                if tk in raw_data_dict:
                    df_tk = raw_data_dict[tk].dropna(how='all')
                    if not df_tk.empty:
                        curr_price = realtime_map.get(tk)
                        df_indi = calculate_indicators(df_tk, realtime_price=curr_price)
                
                if df_indi is not None:
                    curr = float(df_indi['Close_Calc'].iloc[-1])
                    action, color, advice = get_sell_advice(df_indi, avg, b_date)
                else:
                    action, color, advice = "데이터 로딩 중", "gray", "잠시 후 다시 시도"

                if curr > 0:
                    res = calculate_total_profit(tk, avg, curr, qty)
                    display_list.append({
                        "name": name, "tk": tk, "avg": avg, "curr": curr, "qty": qty,
                        "action": action, "color": color, "advice": advice,
                        "profit_pct": res['pct'], "profit_amt": res['profit_amt'],
                        "eval_amt": res['net_eval_amt'], "currency": res['currency']
                    })
                else:
                    display_list.append({
                        "name": name, "tk": tk, "avg": avg, "curr": avg, "qty": qty,
                        "action": "로딩 실패", "color": "gray", "advice": "데이터 없음",
                        "profit_pct": 0.0, "profit_amt": 0.0, "eval_amt": 0.0,
                        "currency": "$" if not tk.endswith(".KS") else "₩"
                    })
            
            priority = {"⚡ 칼손절 (-3%)": 0, "⏱️ 타임컷 매도": 1, "📉 트레일링 익절": 2, "📉 추세 이탈": 3, "💎 슈퍼 홀딩 (Riding)": 4, "⏳ 홀딩 (Waiting)": 5}
            display_list.sort(key=lambda x: priority.get(x['action'], 99))

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
                        
                        st.metric(
                            "수익률", 
                            f"{item['profit_pct']:.2f}%", 
                            delta=f"{sym}{item['profit_amt']:,.0f}" if sym=="₩" else f"{sym}{item['profit_amt']:,.2f}"
                        )
                        st.markdown(
                            f"<small style='color: gray'>평단: {safe_sym}{fmt_avg} / 현재: {safe_sym}{fmt_curr}</small>",
                            unsafe_allow_html=True
                        )
                        
                    with c3:
                        st.markdown(f"##### AI 추천: :{item['color']}[{item['action']}]")
                        st.info(f"{item['advice']}")
                        
                    st.divider()

            if st.button("🗑️ 포트폴리오 전체 삭제"):
                doc_ref.delete()
                st.rerun()

with tab3:
    st.markdown("## 📘 AI 스나이퍼 전략 백서 (Sniper Mode v2.0)")
    
    st.info("""
    **핵심 철학:** "손실은 짧게, 수익은 길게 (Cut losses short, Let profits run)"
    """)
    
    with st.expander("⚔️ 매매 규칙 (Rules of Engagement)", expanded=True):
        st.markdown("""
        1.  **진입 (Entry):** AI 점수 **70점 이상** (확실한 추세만 탑승)
        2.  **손절 (Stop Loss):** **-3%** 도달 시 즉시 매도 (계좌 방어 최우선)
        3.  **타임컷 (Time Cut):** 매수 후 **14일(2주)** 경과 시 조건 없이 매도 (기회비용 확보)
        4.  **익절 (Take Profit):** **트레일링 스탑** 적용
            * 수익률 +5% 미만: 잔파동 무시하고 홀딩
            * 수익률 **+5% 돌파 후**: 고점 대비 **-3%** 하락 시 전량 매도
        """)

    st.header("🧠 스나이퍼 핵심 3요소")
    with st.expander("① 10일선 생명선 매매", expanded=True):
        st.markdown("스윙에서 20일선은 느리고 5일선은 빠릅니다. **10일선**을 생명선으로 삼아 추세를 추적합니다.")
    with st.expander("② 볼린저 밴드 스퀴즈 & 돌파", expanded=True):
        st.markdown("에너지가 응축(스퀴즈)된 후 폭발(돌파)하는 시점을 노립니다.")
    with st.expander("③ MACD 가속도", expanded=True):
        st.markdown("단순 양수가 아니라, 상승 에너지가 '가속'되는 구간에 높은 점수를 부여합니다.")
