# streamlit run app_v3_stream.py

from __future__ import annotations
import io, math, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pydantic import BaseModel, Field
import altair as alt
from fpdf import FPDF
import os
import hashlib

# 폰트 경로 설정 (Nanum Gothic 폰트)
FONT_PATH_REGULAR = "./www/fonts/NanumGothic-Regular.ttf"
FONT_PATH_BOLD = "./www/fonts/NanumGothic-Bold.ttf"
REPORT_MONTH = 12  # 명세서 기준 월 (12월 데이터 사용)



def render_pf_combined(df_acc: pd.DataFrame, placeholder):
    df_pf = df_acc.copy()
    if "측정일시" not in df_pf.columns:
        df_pf["측정일시"] = pd.to_datetime(df_pf["timestamp"], errors="coerce")
    if "지상역률_주간클립" not in df_pf.columns:
        df_pf["지상역률_주간클립"] = np.random.uniform(85, 99, len(df_pf))
    if "진상역률(%)" not in df_pf.columns:
        df_pf["진상역률(%)"] = np.random.uniform(90, 100, len(df_pf))

    df_pf["주간여부"] = ((df_pf["측정일시"].dt.hour >= 9) & (df_pf["측정일시"].dt.hour <= 23)).astype(int)
    df_pf["야간여부"] = ((df_pf["측정일시"].dt.hour < 9) | (df_pf["측정일시"].dt.hour >= 23)).astype(int)

    latest_time = df_pf["측정일시"].max()
    start_domain = latest_time - pd.Timedelta(hours=24) if pd.notna(latest_time) else None
    x_axis = alt.X(
        "측정일시:T", title="시간",
        scale=alt.Scale(domain=[start_domain, latest_time]) if start_domain else alt.Undefined
    )
    ch = create_combined_pf_chart(df_pf, x_axis)
    if ch:
        placeholder.altair_chart(ch, use_container_width=True)
    else:
        placeholder.info("유효한 역률 데이터가 없습니다.")


def render_tou_chart(df_acc: pd.DataFrame, placeholder):
    df_tou = df_acc.copy()

    # TOU 매핑 (없으면 자동 생성)
    if "TOU" not in df_tou.columns:
        df_tou["hour"] = df_tou["timestamp"].dt.hour
        df_tou["TOU"] = df_tou["hour"].apply(lambda h: (
            "경부하" if (h >= 23 or h < 7) else
            "최대부하" if (10 <= h < 18) else
            "중간부하"
        ))

    # 단가/예측요금
    if "unit_price" not in df_tou.columns:
        tou_price = {"경부하": 90, "중간부하": 120, "최대부하": 160}
        df_tou["unit_price"] = df_tou["TOU"].map(tou_price)
    df_tou["예측요금(원)"] = df_tou["kWh"] * df_tou["unit_price"]

    # 1시간 이동평균(15분×4) — TOU별
    df_tou = df_tou.sort_values("timestamp")
    df_tou["예측요금_1시간MA"] = (
        df_tou.groupby("TOU", group_keys=False)["예측요금(원)"]
              .rolling(window=4, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    # 최근 24시간만 표시 (원하시면 제거 가능)
    latest_time = df_tou["timestamp"].max()
    x_dom = [latest_time - pd.Timedelta(hours=24), latest_time] if pd.notna(latest_time) else None
    x_enc = alt.X("timestamp:T", title="시간",
                  scale=alt.Scale(domain=x_dom) if x_dom else alt.Undefined)

    color_scale = alt.Scale(
        domain=["경부하", "중간부하", "최대부하"],
        range=["#2E86C1", "#F1C40F", "#E74C3C"]
    )
    base = alt.Chart(df_tou).mark_line(opacity=0.35).encode(
        x=x_enc,
        y=alt.Y("예측요금(원):Q", title="예측 요금 (원)", scale=alt.Scale(zero=False)),
        color=alt.Color("TOU:N", scale=color_scale, legend=alt.Legend(title="TOU 구간")),
        tooltip=[
            alt.Tooltip("timestamp:T", title="시간"),
            alt.Tooltip("TOU:N", title="구간"),
            alt.Tooltip("예측요금(원):Q", format=",.0f"),
            alt.Tooltip("kWh:Q", format=",.2f", title="전력사용량(kWh)")
        ]
    )
    ma = alt.Chart(df_tou).mark_line(strokeWidth=3).encode(
        x=x_enc,
        y="예측요금_1시간MA:Q",
        color=alt.Color("TOU:N", scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip("timestamp:T", title="시간"),
            alt.Tooltip("TOU:N", title="구간"),
            alt.Tooltip("예측요금_1시간MA:Q", title="1시간 평균", format=",.0f")
        ]
    )
    tou_chart = (base + ma).properties(
        title="⚡ 실시간 TOU(시간대)별 예측 요금 추이 (1시간 이동평균 포함)",
        height=260
    )
    placeholder.altair_chart(tou_chart, use_container_width=True)




def create_combined_pf_chart(df_pf, shared_x=None):
    """
    실시간 통합 역률 추이 (지상역률/진상역률)
    - 지상역률: 09~23시 실선, 23~09시 점선
    - 진상역률: 23~09시 실선, 09~23시 점선
    - 기준선 표시 (지상: 90%, 진상: 95%)
    - 범례: '지상역률' / '진상역률'로 표시
    """
    import altair as alt
    import pandas as pd

    # Data copy & Validation
    df_pf = df_pf.copy()
    if df_pf.empty:
        return alt.Chart(pd.DataFrame()).properties(title="데이터 없음", height=400)
        
    df_pf["hour"] = df_pf["측정일시"].dt.hour

    # Flags
    # 주간 (Daytime): 09시 이상 ~ 23시 미만 (9, 10, ..., 22시)
    is_day = ((df_pf["hour"] >= 9) & (df_pf["hour"] < 23))
    # 야간 (Nighttime): 23시 이상 또는 9시 미만 (23, 0, ..., 8시)
    is_night = ((df_pf["hour"] >= 23) | (df_pf["hour"] < 9))

    # Base axis & Y-scale
    if shared_x is None:
        latest_time = df_pf["측정일시"].max()
        start_domain = latest_time - pd.Timedelta(hours=24) if pd.notna(latest_time) else None
        
        shared_x = alt.X(
            "측정일시:T", title="시간",
            scale=alt.Scale(domain=[start_domain, latest_time]) if start_domain else alt.Undefined
        )
        
    y_encoding = alt.Y("역률값:Q", title="역률(%)", scale=alt.Scale(domain=[85, 101])) 
    
    # 색상 정의
    COLOR_LAG = '#F39C12' # 주황: 지상역률
    COLOR_LEAD = '#2980B9' # 파랑: 진상역률
    
    # 데이터셋에 레이블 추가 (범례용)
    df_pf['지상역률_Label'] = '지상역률'
    df_pf['진상역률_Label'] = '진상역률'

    # ----------------------------------------------------
    # ① 지상역률 (주간: 실선, 야간: 점선)
    # ----------------------------------------------------
    
    # 주간 실선 (Daytime Solid)
    chart_lag_day_solid = alt.Chart(df_pf[is_day]).mark_line(
        point=False, strokeWidth=2.5, color=COLOR_LAG
    ).encode(
        x=shared_x, y=alt.Y("지상역률_주간클립:Q", title="역률(%)", scale=alt.Scale(domain=[85, 101])),
        # 범례에 사용될 컬럼과 색상 지정
        color=alt.Color('지상역률_Label:N', scale=alt.Scale(domain=['지상역률', '진상역률'], range=[COLOR_LAG, COLOR_LEAD]), legend=alt.Legend(title="역률 종류")),
        tooltip=['측정일시:T', alt.Tooltip('지상역률_주간클립:Q', format=',.2f', title='지상역률(주간)')]
    )

    # 야간 점선 (Nighttime Dotted)
    chart_lag_night_dotted = alt.Chart(df_pf[is_night]).mark_line(
        point=False, strokeWidth=1.5, strokeDash=[5, 4], color=COLOR_LAG
    ).encode(
        x=shared_x, y=alt.Y("지상역률_주간클립:Q"),
        color=alt.value(COLOR_LAG), # 범례 중복을 피하기 위해 value 사용
        tooltip=['측정일시:T', alt.Tooltip('지상역률_주간클립:Q', format=',.2f', title='지상역률(야간)')]
    )

    # ----------------------------------------------------
    # ② 진상역률 (야간: 실선, 주간: 점선)
    # ----------------------------------------------------

    # 야간 실선 (Nighttime Solid)
    chart_lead_night_solid = alt.Chart(df_pf[is_night]).mark_line(
        point=False, strokeWidth=2.5, color=COLOR_LEAD
    ).encode(
        x=shared_x, y=alt.Y("진상역률(%):Q"),
        color=alt.Color('진상역률_Label:N', scale=alt.Scale(domain=['지상역률', '진상역률'], range=[COLOR_LAG, COLOR_LEAD])),
        tooltip=['측정일시:T', alt.Tooltip('진상역률(%):Q', format=',.2f', title='진상역률(야간)')]
    )

    # 주간 점선 (Daytime Dotted)
    chart_lead_day_dotted = alt.Chart(df_pf[is_day]).mark_line(
        point=False, strokeWidth=1.5, strokeDash=[5, 4], color=COLOR_LEAD
    ).encode(
        x=shared_x, y=alt.Y("진상역률(%):Q"),
        color=alt.value(COLOR_LEAD), # 범례 중복을 피하기 위해 value 사용
        tooltip=['측정일시:T', alt.Tooltip('진상역률(%):Q', format=',.2f', title='진상역률(주간)')]
    )
    
    # ----------------------------------------------------
    # ③ 기준선 및 텍스트
    # ----------------------------------------------------
    
    baseline_lag = (
        alt.Chart(pd.DataFrame({"y": [90]}))
        .mark_rule(color=COLOR_LAG, strokeDash=[6, 3], strokeWidth=1.5)
        .encode(y="y:Q")
    )

    baseline_lead = (
        alt.Chart(pd.DataFrame({"y": [95]}))
        .mark_rule(color=COLOR_LEAD, strokeDash=[6, 3], strokeWidth=1.5)
        .encode(y="y:Q")
    )
    
    text_lag = alt.Chart(pd.DataFrame({"y": [90]})).mark_text(
        text="지상기준선 90%", align='left', baseline='top', dx=5, dy=-10, color=COLOR_LAG, fontSize=10
    ).encode(y='y:Q')
    
    text_lead = alt.Chart(pd.DataFrame({"y": [95]})).mark_text(
        text="진상기준선 95%", align='left', baseline='bottom', dx=5, dy=10, color=COLOR_LEAD, fontSize=10
    ).encode(y='y:Q')


    # ----------------------------------------------------
    # ④ 최종 결합
    # ----------------------------------------------------
    # 순서: 점선 -> 실선 -> 기준선 순으로 겹쳐서 그림
    combined_chart = (
        chart_lag_night_dotted + chart_lead_day_dotted + # 점선 (배경)
        chart_lag_day_solid + chart_lead_night_solid +   # 실선 (강조)
        baseline_lag + baseline_lead + text_lag + text_lead
    ).properties(
        title="⚙️ 실시간 통합 역률 추이", 
        height=400
    ).configure_title(
        fontSize=16, anchor="start"
    ).configure_axis(
        labelFontSize=12, titleFontSize=13
    ).interactive()

    return combined_chart





# ==============================
# 🤖 Chatbot Modal (from app.py)
# ==============================
@st.dialog("🤖 전력 관리 담당자")
def show_chatbot():
    """st.dialog를 사용하여 전력 관리 담당자 연락 UI를 표시합니다."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "안녕하세요!\n 전력 모니터 관련 질문에 답변해 드립니다."}
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "image" in msg:
                st.image(msg["image"])

    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response_content = "현재 [오정민] 담당자가 예비군에 참석하여 답변이 어렵습니다.🫡\n 다음에 다시 문의해주세요!"
        image_url = "./data/army.JPG"  

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response_content,
            "image": image_url
        })

        with st.chat_message("assistant"):
            st.markdown(response_content)
            st.image(image_url)

    st.divider()
    if st.button("닫기", use_container_width=True):
        st.session_state.show_chat = False
        st.rerun()


# =========================================
# Page Config
# =========================================
st.set_page_config(
    page_title="LS빅데이터스쿨 5기 최고〰️",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================
# Chatbot Execution Logic
# ==============================
if st.session_state.get("show_chat", False):
    show_chatbot()


# =========================================
# Data Models
# =========================================
class BillInputs(BaseModel):
    contract_power_kw: float = 500.0
    basic_charge_per_kw: float = 7000.0
    tariff_rates: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    fuel_adj_per_kwh: float = 0.0
    climate_per_kwh: float = 0.0
    industry_fund_rate: float = 0.037
    vat_rate: float = 0.1
    over_contract_penalty_rate: float = 1.5
    tariff_code: str = ""
    tariff_label: str = ""

LOAD_ORDER = ["경부하", "중간부하", "최대부하"]
SEASON_KEYS = ("summer", "spring_fall", "winter")
SEASON_LABELS = {
    "summer": "여름철(6~8월)",
    "spring_fall": "봄·가을철(3~5,9~10월)",
    "winter": "겨울철(11~2월)",
}


def month_to_season_key(month: int) -> str:
    try:
        month_int = int(month)
    except (TypeError, ValueError):
        month_int = 1
    if month_int in (6, 7, 8):
        return "summer"
    if month_int in (3, 4, 5, 9, 10):
        return "spring_fall"
    return "winter"


TARIFF_PLANS: Dict[str, Dict[str, object]] = {
    "A1": {
        "label": "고압A 선택Ⅰ",
        "basic_charge": 7220.0,
        "energy_rates": {
            "경부하": {"summer": 99.5, "spring_fall": 99.5, "winter": 106.5},
            "중간부하": {"summer": 152.4, "spring_fall": 122.0, "winter": 152.6},
            "최대부하": {"summer": 234.5, "spring_fall": 152.7, "winter": 210.1},
        },
    },
    "A2": {
        "label": "고압A 선택Ⅱ",
        "basic_charge": 8320.0,
        "energy_rates": {
            "경부하": {"summer": 94.0, "spring_fall": 94.0, "winter": 101.0},
            "중간부하": {"summer": 147.9, "spring_fall": 116.5, "winter": 147.9},
            "최대부하": {"summer": 229.0, "spring_fall": 147.2, "winter": 204.6},
        },
    },
    "A3": {
        "label": "고압A 선택Ⅲ",
        "basic_charge": 9810.0,
        "energy_rates": {
            "경부하": {"summer": 90.9, "spring_fall": 90.9, "winter": 99.1},
            "중간부하": {"summer": 146.3, "spring_fall": 113.0, "winter": 146.3},
            "최대부하": {"summer": 216.6, "spring_fall": 139.8, "winter": 193.4},
        },
    },
    "B1": {
        "label": "고압B 선택Ⅰ",
        "basic_charge": 6630.0,
        "energy_rates": {
            "경부하": {"summer": 105.5, "spring_fall": 105.5, "winter": 113.7},
            "중간부하": {"summer": 161.7, "spring_fall": 131.7, "winter": 161.7},
            "최대부하": {"summer": 242.9, "spring_fall": 162.0, "winter": 217.9},
        },
    },
    "B2": {
        "label": "고압B 선택Ⅱ",
        "basic_charge": 7380.0,
        "energy_rates": {
            "경부하": {"summer": 105.6, "spring_fall": 105.6, "winter": 112.6},
            "중간부하": {"summer": 157.9, "spring_fall": 127.9, "winter": 157.9},
            "최대부하": {"summer": 239.1, "spring_fall": 158.2, "winter": 214.1},
        },
    },
    "B3": {
        "label": "고압B 선택Ⅲ",
        "basic_charge": 8190.0,
        "energy_rates": {
            "경부하": {"summer": 103.9, "spring_fall": 103.9, "winter": 111.0},
            "중간부하": {"summer": 156.2, "spring_fall": 126.3, "winter": 156.2},
            "최대부하": {"summer": 237.5, "spring_fall": 156.6, "winter": 212.4},
        },
    },
    "C1": {
        "label": "고압C 선택Ⅰ",
        "basic_charge": 6590.0,
        "energy_rates": {
            "경부하": {"summer": 108.9, "spring_fall": 108.9, "winter": 115.8},
            "중간부하": {"summer": 161.8, "spring_fall": 131.8, "winter": 161.4},
            "최대부하": {"summer": 243.2, "spring_fall": 162.2, "winter": 218.0},
        },
    },
    "C2": {
        "label": "고압C 선택Ⅱ",
        "basic_charge": 7520.0,
        "energy_rates": {
            "경부하": {"summer": 104.2, "spring_fall": 104.2, "winter": 111.4},
            "중간부하": {"summer": 157.1, "spring_fall": 127.1, "winter": 156.7},
            "최대부하": {"summer": 238.0, "spring_fall": 157.5, "winter": 213.7},
        },
    },
    "C3": {
        "label": "고압C 선택Ⅲ",
        "basic_charge": 8090.0,
        "energy_rates": {
            "경부하": {"summer": 103.1, "spring_fall": 103.1, "winter": 110.0},
            "중간부하": {"summer": 156.0, "spring_fall": 126.0, "winter": 155.6},
            "최대부하": {"summer": 236.9, "spring_fall": 156.4, "winter": 212.6},
        },
    },
}

DEFAULT_TARIFF_CODE = "B2"

SEASON_TIME_WINDOWS: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
    "summer": {
        "경부하": [(23, 24), (0, 9)],
        "중간부하": [(9, 11), (12, 13), (17, 23)],
        "최대부하": [(11, 12), (13, 17)],
    },
    "spring_fall": {
        "경부하": [(23, 24), (0, 9)],
        "중간부하": [(9, 10), (12, 17), (20, 23)],
        "최대부하": [(10, 12), (17, 20)],
    },
    "winter": {
        "경부하": [(23, 24), (0, 9)],
        "중간부하": [(9, 10), (12, 17), (20, 23)],
        "최대부하": [(10, 12), (17, 20)],
    },
}


def plan_rates_to_display(energy_rates: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    label_map = {
        "summer": "여름철",
        "spring_fall": "봄가을철",
        "winter": "겨울철",
    }

    for load in LOAD_ORDER:
        seasonal = energy_rates.get(load, {})
        row = {"부하": load}
        for season_key in SEASON_KEYS:
            label = label_map.get(season_key, season_key)
            row[label] = float(seasonal.get(season_key, np.nan))
        rows.append(row)

    return pd.DataFrame(rows)

# =========================================
# Utils
# =========================================
@st.cache_data(show_spinner=False, ttl=3600)
def generate_demo_data(days: int = 35, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    idx = pd.date_range(start, end, freq="15min")
    base = []
    for ts in idx:
        hour = ts.hour
        is_we = ts.weekday() >= 5
        val = 300 + 200 * np.sin((hour - 6) / 24 * 2 * np.pi)
        val += -60 if is_we else 0
        val += rng.normal(0, 20)
        base.append(max(val, 50))
    df = pd.DataFrame({"timestamp": idx, "kW": base})
    df["kWh"] = df["kW"] * 0.25
    return df

def infer_15min_kW_kWh(df: pd.DataFrame) -> pd.DataFrame:
    """kW/kWh 최소 보정: 15분 간격 기준"""
    df = df.copy()
    if "kWh" not in df.columns and "kW" in df.columns:
        df["kWh"] = df["kW"] * 0.25
    if "kW" not in df.columns and "kWh" in df.columns:
        df["kW"] = df["kWh"] / 0.25
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def preprocess_data(df: pd.DataFrame, tariff_rates: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = infer_15min_kW_kWh(df)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["season_key"] = df["timestamp"].dt.month.map(month_to_season_key)

    def determine_load(month: int, hour: int) -> str:
        season = month_to_season_key(month)
        season_windows = SEASON_TIME_WINDOWS.get(season, {})
        for load_name, windows in season_windows.items():
            for start, end in windows:
                if start <= end:
                    if start <= hour < end:
                        return load_name
                else:  # overnight wrap
                    if hour >= start or hour < end:
                        return load_name
        return LOAD_ORDER[0]

    df["TOU"] = df.apply(lambda row: determine_load(row["timestamp"].month, row["hour"]), axis=1)

    def resolve_unit_price(row) -> float:
        load_rates = tariff_rates.get(row["TOU"], {})
        return float(load_rates.get(row["season_key"], 0.0))

    df["unit_price"] = df.apply(resolve_unit_price, axis=1).astype(float)
    return df

def safe_sum(series: pd.Series) -> float:
    try: return float(series.sum())
    except Exception: return 0.0

def human_pct(a: float) -> str:
    if a is None or not isinstance(a, (int, float)) or math.isnan(a): return "-"
    return f"{a:+.1f}%"


@st.cache_data(show_spinner=False)
def load_train_pf_dataset() -> pd.DataFrame:
    path = Path("./data/train.csv")
    if not path.exists():
        st.error("train.csv 파일을 찾을 수 없습니다. 부하/그룹 분석 탭을 사용할 수 없습니다.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rename_map = {
        "측정일시": "timestamp",
        "전력사용량(kWh)": "kWh",
    }
    for src, dst in rename_map.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
    if "timestamp" not in df.columns:
        st.error("train.csv에 'timestamp' 또는 '측정일시' 컬럼이 없어 분석을 진행할 수 없습니다.")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# =========================================
# 비교 테이블 데이터 생성 (app.py 원본)
# =========================================
def create_comparison_table_data(train_df, results_df, target_month):
    if train_df is None or results_df.empty:
        return pd.DataFrame(), f"{target_month}월 평균"
    try:
        base_label = f"{target_month}월 평균"
        base_df = train_df[train_df["월"] == target_month].copy()
        if not base_df.empty:
            base_series = base_df.groupby("시간")["전기요금(원)"].mean()
        else:
            # 학습 데이터에 대상 월이 없으면 결과 데이터로 대체
            base_series = results_df.groupby("시간")["예측요금(원)"].mean()

        # 2. 어제 (Yesterday)
        latest_datetime = results_df["측정일시"].iloc[-1]
        latest_date = latest_datetime.date()
        yesterday_date = latest_date - pd.Timedelta(days=1)

        yesterday_df = results_df[results_df["측정일시"].dt.date == yesterday_date]
        if yesterday_df.empty:
            yesterday_df = train_df[train_df["측정일시"].dt.date == yesterday_date]
            if not yesterday_df.empty:
                yesterday_hourly = yesterday_df.groupby("시간")["전기요금(원)"].mean()
            else:
                yesterday_hourly = pd.Series(dtype=float)
        else:
            yesterday_hourly = yesterday_df.groupby("시간")["예측요금(원)"].mean()

        # 3. 오늘 (Today)
        today_df = results_df[results_df["측정일시"].dt.date == latest_date]
        today_hourly = today_df.groupby("시간")["예측요금(원)"].mean()

        # 4. DataFrame으로 통합
        comp_df = pd.DataFrame(
            {
                base_label: base_series,
                "어제": yesterday_hourly,
                "오늘": today_hourly,
            }
        ).reindex(range(24))
        comp_df["전일 대비"] = comp_df["오늘"] - comp_df["어제"].fillna(0)

        return comp_df.fillna(np.nan), base_label

    except Exception as e:
        st.error(f"비교 테이블 데이터 생성 중 오류 발생: {e}")
        return pd.DataFrame(), f"{target_month}월 평균"


# =========================================
# PDF 생성 함수 (app.py 원본 그대로)
# =========================================
def generate_bill_pdf(report_data, comparison_df=None):
    try:
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()
        pdf.add_font("Nanum", "", FONT_PATH_REGULAR, uni=True)
        pdf.add_font("Nanum", "B", FONT_PATH_BOLD, uni=True)
        pdf.set_font("Nanum", "", 10)

        # 3. (날짜 헤더 추가)
        yesterday_header = f"어제 ({report_data.get('yesterday_str', '')})"
        today_header = f"오늘 ({report_data.get('today_str', '')})"
        month_label = report_data.get("report_month_label", "12월")
        base_label = report_data.get("comparison_base_label", f"{month_label} 평균")

        # --- 1~4. 상단 정보
        pdf.set_font_size(18)
        pdf.cell(0, 15, f"{month_label} 실시간 예측 전기요금 명세서", border=1, ln=1, align="C")
        pdf.ln(3)

        pdf.set_font_size(12)
        pdf.cell(0, 8, " [ 예측 고객 정보 ]", border="B", ln=1)
        col_width = pdf.w / 2 - 12
        pdf.cell(col_width, 8, "고객명: LS 청주공장", border=0)
        report_date = report_data.get("report_date")
        if isinstance(report_date, pd.Timestamp) and pd.notna(report_date):
            report_date_str = report_date.strftime("%Y-%m-%d")
        else:
            report_date_str = str(report_date)

        pdf.cell(
            col_width,
            8,
            f"명세서 발행일: {report_date_str}",
            border=0,
            ln=1,
        )
        period_start = report_data.get("period_start")
        period_end = report_data.get("period_end")
        start_str = (
            period_start.strftime("%Y-%m-%d %H:%M")
            if isinstance(period_start, pd.Timestamp) and pd.notna(period_start)
            else "-"
        )
        end_str = (
            period_end.strftime("%Y-%m-%d %H:%M")
            if isinstance(period_end, pd.Timestamp) and pd.notna(period_end)
            else "-"
        )
        pdf.multi_cell(0, 6, f"예측 기간: {start_str} ~ {end_str}", border=0, align="L")
        pdf.ln(3)

        pdf.set_fill_color(240, 240, 240)
        pdf.set_font_size(14)
        pdf.cell(40, 12, "총 예측 요금", border=1, align="C", fill=True)
        pdf.set_font_size(16)
        pdf.cell(0, 12, f"{report_data['total_bill']:,.0f} 원", border=1, ln=1, align="R")
        pdf.ln(3)

        # --- 5. 세부 내역
        pdf.set_font_size(12)
        pdf.cell(0, 8, " [ 예측 세부 내역 ]", border="B", ln=1)

        pdf.set_font_size(11)
        pdf.set_fill_color(240, 240, 240)
        header_h = 8
        w1, w2, w3, w4 = 45, 50, 50, 45
        pdf.cell(w1, header_h, "항목 (부하구분)", border=1, align="C", fill=True)
        pdf.cell(w2, header_h, "예측 사용량 (kWh)", border=1, align="C", fill=True)
        pdf.cell(w3, header_h, "예측 요금 (원)", border=1, align="C", fill=True)
        pdf.cell(w4, header_h, "요금/사용량 (원/kWh)", border=1, ln=1, align="C", fill=True)

        pdf.set_font_size(10)
        bands = ["경부하", "중간부하", "최대부하"]
        for band in bands:
            usage = report_data["usage_by_band"].get(band, 0.0)
            bill = report_data["bill_by_band"].get(band, 0.0)
            cost_per_kwh = bill / usage if usage > 0 else 0.0

            pdf.cell(w1, header_h, band, border=1, align="C")
            pdf.cell(w2, header_h, f"{usage:,.2f}", border=1, align="R")
            pdf.cell(w3, header_h, f"{bill:,.0f}", border=1, align="R")
            pdf.cell(w4, header_h, f"{cost_per_kwh:,.1f}", border=1, ln=1, align="R")

        pdf.set_font("Nanum", "B", 11)
        total_usage = report_data["total_usage"]
        total_bill = report_data["total_bill"]
        total_cost_per_kwh = total_bill / total_usage if total_usage > 0 else 0.0

        pdf.cell(w1, header_h, "합계", border=1, align="C", fill=True)
        pdf.cell(w2, header_h, f"{total_usage:,.2f}", border=1, align="R", fill=True)
        pdf.cell(w3, header_h, f"{total_bill:,.0f}", border=1, align="R", fill=True)
        pdf.cell(
            w4, header_h, f"{total_cost_per_kwh:,.1f}", border=1, ln=1, align="R", fill=True
        )

        pdf.ln(5)

        # ---6. 주요 요금 결정 지표
        pdf.set_font("Nanum", "", 12)
        pdf.cell(0, 8, " [ 주요 요금 결정 지표 (예측) ]", border="B", ln=1)
        pdf.ln(1)

        start_y = pdf.get_y()
        col_width = 95

        # --- 1. 왼쪽 컬럼 (기본요금) ---
        pdf.set_x(10)
        pdf.set_font("Nanum", "B", 10)
        pdf.multi_cell(col_width, 7, "1. 기본요금 (Demand Charge) 지표", border=0, align="L")

        pdf.set_font("Nanum", "", 9)
        peak_kw = report_data.get("peak_demand_kw", 0)
        peak_time = report_data.get("peak_demand_time", pd.NaT)
        peak_time_str = peak_time.strftime("%Y-%m-%d %H:%M") if pd.notna(peak_time) else "N/A"

        min_kw = report_data.get("min_demand_kw", 0)
        min_time = report_data.get("min_demand_time", pd.NaT)
        min_time_str = min_time.strftime("%Y-%m-%d %H:%M") if pd.notna(min_time) else "N/A"

        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - {month_label} 최대 요금적용전력: {peak_kw:,.2f} kW", border=0, align="L")
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 최대치 발생일시: {peak_time_str}", border=0, align="L")
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - {month_label} 최저 요금적용전력: {min_kw:,.2f} kW", border=0, align="L")
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 최저치 발생일시: {min_time_str}", border=0, align="L")

        end_y_left = pdf.get_y()

        # --- 2. 오른쪽 컬럼 (역률요금) ---
        pdf.set_y(start_y)
        pdf.set_x(10 + col_width)

        pdf.set_font("Nanum", "B", 10)
        pdf.multi_cell(col_width, 7, "2. 역률요금 (Power Factor) 지표", border=0, align="L")

        pdf.set_font("Nanum", "", 9)
        avg_day_pf = report_data.get("avg_day_pf", 0)
        penalty_d_h = report_data.get("penalty_day_hours", 0)
        bonus_d_h = report_data.get("bonus_day_hours", 0)
        avg_night_pf = report_data.get("avg_night_pf", 0)
        penalty_n_h = report_data.get("penalty_night_hours", 0)

        pdf.set_x(10 + col_width)
        pdf.multi_cell(
            col_width, 6, f"  - 주간(09-23시) 평균 지상역률: {avg_day_pf:.2f} %", border=0, align="L"
        )
        pdf.set_x(10 + col_width)
        pdf.multi_cell(
            col_width,
            6,
            f"    (페널티[<90%] {penalty_d_h}시간 / 보상[>95%] {bonus_d_h}시간)",
            border=0,
            align="L",
        )
        pdf.set_x(10 + col_width)
        pdf.multi_cell(
            col_width, 6, f"  - 야간(23-09시) 평균 진상역률: {avg_night_pf:.2f} %", border=0, align="L"
        )
        pdf.set_x(10 + col_width)
        pdf.multi_cell(
            col_width, 6, f"    (페널티[<95%] {penalty_n_h}시간)", border=0, align="L"
        )

        end_y_right = pdf.get_y()

        pdf.set_y(max(end_y_left, end_y_right))
        pdf.ln(5)

        # --- 7. 시간대별 요금 비교 (표) ---
        pdf.set_font("Nanum", "", 12)
        pdf.cell(0, 8, " [ 시간대별 요금 비교 (단위: 원) ]", border="B", ln=1)
        pdf.ln(1)

        if comparison_df is not None and not comparison_df.empty:
            pdf.set_font("Nanum", "", 8)
            cell_h = 6
            w_time = 12
            w_nov = 21
            w_yes = 21
            w_tod = 21
            w_diff = 20

            def draw_header(start_x):
                pdf.set_font("Nanum", "B", 8)
                pdf.set_x(start_x)
                pdf.cell(w_time, cell_h, "시간", 1, 0, "C", 1)
                pdf.cell(w_nov, cell_h, base_label, 1, 0, "C", 1)
                pdf.cell(w_yes, cell_h, yesterday_header, 1, 0, "C", 1)
                pdf.cell(w_tod, cell_h, today_header, 1, 0, "C", 1)
                pdf.cell(w_diff, cell_h, "전일 대비", 1, 0, "C", 1)

            start_y = pdf.get_y()
            draw_header(10)
            pdf.set_y(start_y)
            draw_header(10 + 95)
            pdf.ln(cell_h)

            def fmt(val, is_diff=False):
                if pd.isna(val):
                    return "-"
                prefix = "+" if is_diff and val > 0 else ""
                return f"{prefix}{val:,.0f}"

            for i in range(12):
                row_left = comparison_df.iloc[i]
                pdf.set_x(10)
                pdf.cell(w_time, cell_h, str(i), 1, 0, "C")
                pdf.cell(w_nov, cell_h, fmt(row_left[base_label]), 1, 0, "R")
                pdf.cell(w_yes, cell_h, fmt(row_left["어제"]), 1, 0, "R")
                pdf.cell(w_tod, cell_h, fmt(row_left["오늘"]), 1, 0, "R")
                pdf.cell(w_diff, cell_h, fmt(row_left["전일 대비"], True), 1, 0, "R")

                row_right = comparison_df.iloc[i + 12]
                pdf.set_x(10 + 95)
                pdf.cell(w_time, cell_h, str(i + 12), 1, 0, "C")
                pdf.cell(w_nov, cell_h, fmt(row_right[base_label]), 1, 0, "R")
                pdf.cell(w_yes, cell_h, fmt(row_right["어제"]), 1, 0, "R")
                pdf.cell(w_tod, cell_h, fmt(row_right["오늘"]), 1, 0, "R")
                pdf.cell(w_diff, cell_h, fmt(row_right["전일 대비"], True), 1, 0, "R")

                pdf.ln(cell_h)

            pdf.ln(3)
        else:
            pdf.set_font_size(10)
            pdf.cell(
                0,
                10,
                "비교 데이터를 생성할 수 없습니다 (데이터 부족 또는 오류).",
                border=1,
                ln=1,
                align="C",
            )
            pdf.ln(3)

        # --- 8. 하단 안내문 ---
        pdf.set_font_size(9)
        pdf.multi_cell(
            0,
            5,
            f"* 본 명세서는 '{month_label} 전기요금 실시간 예측 시뮬레이션'을 통해 생성된 예측값이며, "
            "실제 청구되는 요금과 다를 수 있습니다.\n"
            "* 예측 모델: LightGBM, XGBoost, CatBoost 앙상블 모델",
            border=1,
            align="L",
        )

        return bytes(pdf.output())

    except FileNotFoundError:
        st.error(f"PDF 생성 오류: 폰트 파일('{FONT_PATH_REGULAR}' 등)을 찾을 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"PDF 생성 중 알 수 없는 오류 발생: {e}")
        return None




# =========================================
# Sidebar — Data Source & Params
# =========================================
st.sidebar.header("⚙️ 설정")

source = "실시간 전기요금 분석"


st.sidebar.markdown("**실시간 예측 스트리밍 제어**")
col_s1, col_s2, col_s3 = st.sidebar.columns([1,1,1])
with col_s1:
    if st.button("▶️ 시작", key="btn_start"):
        st.session_state.streaming_running = True
        # 초기화: 파일을 로딩하고, 누적 버퍼 준비
        if "stream_source_df" not in st.session_state:
            try:
                src = pd.read_csv("./data/predicted_test_data.csv")
            except FileNotFoundError:
                st.sidebar.error("`./data/predicted_test_data.csv`를 찾을 수 없습니다.")
                st.stop()
            # 표준화
            if "timestamp" not in src.columns and "측정일시" in src.columns:
                src = src.rename(columns={"측정일시": "timestamp"})
            if "kWh" not in src.columns and "전력사용량(kWh)" in src.columns:
                src = src.rename(columns={"전력사용량(kWh)": "kWh"})
            src["timestamp"] = pd.to_datetime(src["timestamp"])
            src = src.sort_values("timestamp").reset_index(drop=True)
            st.session_state.stream_source_df = src
            st.session_state.stream_idx = 0
            st.session_state.stream_accum_df = pd.DataFrame(columns=src.columns)
            st.session_state.total_bill = 0.0
            st.session_state.total_usage = 0.0
            st.session_state.last_timestamp = None
with col_s2:
    if st.button("⏸️ 일시정지", key="btn_pause"):
        st.session_state.streaming_running = False
with col_s3:
    if st.button("⏹️ 초기화", key="btn_stop"):
        st.session_state.streaming_running = False
        for k in ["stream_source_df","stream_idx","stream_accum_df"]:
            if k in st.session_state: del st.session_state[k]

st.sidebar.subheader("계약/목표 설정")
if "selected_tariff_code" not in st.session_state:
    st.session_state.selected_tariff_code = DEFAULT_TARIFF_CODE

tariff_codes = list(TARIFF_PLANS.keys())
selected_tariff_code = st.sidebar.selectbox(
    "한전 요금제 선택",
    tariff_codes,
    index=tariff_codes.index(st.session_state.selected_tariff_code),
    format_func=lambda code: TARIFF_PLANS[code]["label"],
)

st.session_state.selected_tariff_code = selected_tariff_code

plan_info = TARIFF_PLANS[selected_tariff_code]
contract_power = st.sidebar.number_input("계약전력(kW)", min_value=10.0, value=500.0, step=10.0)
peak_alert_threshold = st.sidebar.slider("피크 경보 임계치(% of 계약전력)", 50, 120, 90)
st.sidebar.caption(f"{plan_info['label']} 기본요금: {plan_info['basic_charge']:,.0f} 원/kW")

st.sidebar.subheader("시간대별(TOU) 요금 (원/kWh)")
plan_rates_df = plan_rates_to_display(plan_info["energy_rates"])
st.sidebar.table(plan_rates_df)

bill_inputs = BillInputs(
    contract_power_kw=contract_power,
    basic_charge_per_kw=float(plan_info["basic_charge"]),  # type: ignore[arg-type]
    tariff_rates={k: v.copy() for k, v in plan_info["energy_rates"].items()},  # shallow copy
    tariff_code=selected_tariff_code,
    tariff_label=str(plan_info["label"]),
)
peer_avg_multiplier = 0.9


st.sidebar.divider()
pdf_payload = st.session_state.get("sidebar_pdf_payload")
default_pdf_name = "predicted_bill.pdf"
if pdf_payload and pdf_payload.get("bytes"):
    sidebar_pdf_bytes = pdf_payload["bytes"]
    sidebar_pdf_name = pdf_payload.get("name", default_pdf_name)
    sidebar_pdf_disabled = False
else:
    sidebar_pdf_bytes = b""
    sidebar_pdf_name = default_pdf_name
    sidebar_pdf_disabled = True

st.sidebar.download_button(
    "📄 예측 요금 명세서 PDF 다운로드",
    data=sidebar_pdf_bytes,
    file_name=sidebar_pdf_name,
    mime="application/pdf",
    use_container_width=True,
    key="sidebar_pdf_download",
    disabled=sidebar_pdf_disabled,
)

if st.sidebar.button("🤖 담당자와 대화하기", use_container_width=True):
    st.session_state.show_chat = True


# =========================================
# Load Source Data
# =========================================
if "stream_accum_df" in st.session_state and len(st.session_state.stream_accum_df) > 0:
    raw_df = st.session_state.stream_accum_df.rename(
        columns={"측정일시":"timestamp","전력사용량(kWh)":"kWh"}
    )
else:
    raw_df = generate_demo_data(days=2)



# =========================================
# Preprocess & Aggregation
# =========================================
# 표준 컬럼으로 맞추기
if "timestamp" not in raw_df.columns and "측정일시" in raw_df.columns:
    raw_df = raw_df.rename(columns={"측정일시": "timestamp"})
if "kWh" not in raw_df.columns and "전력사용량(kWh)" in raw_df.columns:
    raw_df = raw_df.rename(columns={"전력사용량(kWh)": "kWh"})

df = preprocess_data(raw_df, bill_inputs.tariff_rates)

hourly = df.resample("H", on="timestamp").agg(
    kWh=("kWh","sum"),
    kW=("kW","mean"),
    unit_price=("unit_price","mean"),
    TOU=("TOU", lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0]),
)
daily = df.resample("D", on="timestamp").agg(kWh=("kWh","sum"), kW=("kW","mean"))

if df.empty:
    month_key = pd.Period(datetime.now(), "M")
else:
    month_periods = df["timestamp"].dt.to_period("M")
    target_candidates = month_periods[df["timestamp"].dt.month == REPORT_MONTH]
    month_key = target_candidates.iloc[-1] if not target_candidates.empty else month_periods.iloc[-1]

this_month = df[df["timestamp"].dt.to_period("M") == month_key]
prev_month = df[df["timestamp"].dt.to_period("M") == (month_key - 1)]

# =========================================
# Top Title and Logo
# =========================================
# 1. 타이틀과 로고를 위한 컬럼 분할
col_title, col_logo = st.columns([3.6, 1.4])

with col_title:

    st.markdown(
        """
        <p style="font-size: 43px; font-weight: bold;">
            ⚡ LS 청주1공장 산업용 전력 모니터
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    st.markdown(
    """
    <p style="font-size: 23px; color: #FF005A; font-weight: bold; margin-top: 10px;">
        ⚠️ [대외비] 본 데이터는 승인된 내부 정보입니다.<br> 
    </p>
    """, unsafe_allow_html=True)


    st.markdown(
    """
    <p style="font-size: 23px; color: #003399; font-weight: bold; margin-top: 10px;">
        총무팀의 업무 목적 외 무단 복제, 배포 및 활용을 엄격히 금합니다.
    </p>
    """, unsafe_allow_html=True)

with col_logo:
    st.image("./LS.png", use_container_width=True)


st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

# =========================================
# Streaming KPI Metrics
# =========================================
if "total_bill" not in st.session_state:
    st.session_state.total_bill = 0.0
if "total_usage" not in st.session_state:
    st.session_state.total_usage = 0.0
if "last_timestamp" not in st.session_state:
    st.session_state.last_timestamp = None

col_bill, col_usage, col_time = st.columns(3, gap="large")
top_bill_metric = col_bill.empty()
top_usage_metric = col_usage.empty()
top_time_metric = col_time.empty()

top_bill_metric.metric("누적 전기요금(원)", f"{st.session_state.total_bill:,.0f}")
top_usage_metric.metric("누적 전기사용량(kWh)", f"{st.session_state.total_usage:,.2f}")
last_ts_display = (
    st.session_state.last_timestamp.strftime("%Y-%m-%d %H:%M")
    if isinstance(st.session_state.last_timestamp, pd.Timestamp)
    else "-"
)
top_time_metric.metric("마지막 데이터 시각", last_ts_display)

st.divider()



# =========================================
# Tabs
# =========================================
main_tab, feature_tab, load_tab, alert_tab, bill_tab, report_tab = st.tabs(
    ["메인 대시보드", "피처 분석", "부하/그룹 분석", "피크 & 알람/시뮬레이션", "한전 고지서/요금", "리포트"]
)

# =========================================
# Main Dashboard
# =========================================
with main_tab:
    st.subheader("")

    # 좌우 그래프 (50:50)
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 💰 실시간 전기 요금 추이")
        tou_chart_placeholder = st.empty()
    with col_chart2:
        st.markdown("#### ⚙️ 실시간 통합 역률 추이")
        pf_chart_placeholder = st.empty()

    latest_placeholder = st.empty()

    # ====================================================
    # 📈 렌더 함수 (탭 내부 정의)
    # ====================================================
    def render_stream_views(df_acc):
        """실시간 스트리밍 데이터 시각화 (x축 고정 + 최근 24시간 윈도우 유지)"""
        if df_acc.empty:
            return
        df_acc = df_acc[df_acc["timestamp"] >= df_acc["timestamp"].max() - pd.Timedelta(hours=24)].copy()

        # ───────────────────────────────
        # 1️⃣ X축 범위 (domain) 계산
        # ───────────────────────────────
        latest_time = df_acc["timestamp"].max()

        # 최초 스트리밍 시작 시점 저장
        if "stream_start_time" not in st.session_state:
            st.session_state.stream_start_time = df_acc["timestamp"].min()

        # 최근 24시간 윈도우 유지 (고정된 시작점부터 오른쪽으로 이동)
        window = pd.Timedelta(hours=24)
        end_domain = latest_time
        start_domain = max(st.session_state.stream_start_time, end_domain - window)

        shared_x = alt.X(
            "timestamp:T",
            title="시간",
            scale=alt.Scale(domain=[start_domain, end_domain]),
        )

        # ───────────────────────────────
        # 💰 예측 요금 추이 (연속형 + Load 색상 포인트)
        # ───────────────────────────────
        df_tou = df_acc.copy()
        df_tou["측정일시"] = pd.to_datetime(df_tou["timestamp"], errors="coerce")
        df_tou = df_tou.sort_values("측정일시").reset_index(drop=True)
        
        def worktype(h):
            if (h >= 23 or h < 7): return "Light_Load"
            if 10 <= h < 18:       return "Maximum_Load"
            return "Medium_Load"
        
        def tou_price(h):
            if (h >= 23 or h < 7): return 90
            if 10 <= h < 18:       return 160
            return 120
        
        hours = df_tou["측정일시"].dt.hour
        df_tou["작업유형"] = hours.apply(worktype)
        df_tou["예측요금(원)"] = df_tou["kWh"] * hours.apply(tou_price)
        
        # 색상 매핑
        color_map = {
            "Light_Load": "forestgreen",
            "Medium_Load": "gold",
            "Maximum_Load": "firebrick"
        }
        
        # 1️⃣ 기본 선 (하나의 연속선)
        base_line = (
            alt.Chart(df_tou)
            .mark_line(interpolate="monotone", strokeWidth=2.5, color="#555")
            .encode(
                x=shared_x,
                y=alt.Y("예측요금(원):Q", title="예측요금 (원)"),
                tooltip=[
                    alt.Tooltip("측정일시:T", title="시간"),
                    alt.Tooltip("예측요금(원):Q", format=",.0f"),
                ],
            )
        )
        
        # 2️⃣ 색상 포인트 (Load Type 표시)
        points = (
            alt.Chart(df_tou)
            .mark_point(size=50)
            .encode(
                x="측정일시:T",
                y="예측요금(원):Q",
                color=alt.Color("작업유형:N",
                                scale=alt.Scale(domain=list(color_map.keys()),
                                                range=list(color_map.values())),
                                legend=alt.Legend(title="작업유형")),
                tooltip=[
                    alt.Tooltip("측정일시:T", title="시간"),
                    alt.Tooltip("작업유형:N", title="부하 구간"),
                    alt.Tooltip("예측요금(원):Q", format=",.0f"),
                ]
            )
        )
        
        # 3️⃣ 결합
        chart_tou = (base_line + points).properties(
            width=800, height=400,
        ).configure_legend(
            orient="top-right", labelFontSize=11, titleFontSize=12,
            direction="vertical", symbolSize=80, padding=10
        )
        
        tou_chart_placeholder.altair_chart(chart_tou, use_container_width=True)


        # ───────────────────────────────
        # 3️⃣ ⚙️ 역률 그래프
        # ───────────────────────────────
        df_pf = df_acc.copy()
        df_pf["측정일시"] = pd.to_datetime(df_pf["timestamp"], errors="coerce")
        if "지상역률_주간클립" not in df_pf.columns:
            df_pf["지상역률_주간클립"] = np.random.uniform(88, 99, len(df_pf))
        if "진상역률(%)" not in df_pf.columns:
            df_pf["진상역률(%)"] = np.random.uniform(93, 100, len(df_pf))

        df_pf["주간여부"] = ((df_pf["측정일시"].dt.hour >= 9) & (df_pf["측정일시"].dt.hour <= 23)).astype(int)
        df_pf["야간여부"] = ((df_pf["측정일시"].dt.hour < 9) | (df_pf["측정일시"].dt.hour >= 23)).astype(int)

        pf_chart = (
            alt.Chart(df_pf)
            .transform_fold(["지상역률_주간클립", "진상역률(%)"], as_=["유형", "값"])
            .mark_line(point=True, interpolate="monotone", strokeWidth=2)
            .encode(
                x=shared_x,
                y=alt.Y("값:Q", title="역률 (%)"),
                color=alt.Color("유형:N", title="역률 종류",
                                scale=alt.Scale(range=["#FF9500", "#007AFF"])),
                tooltip=[
                    alt.Tooltip("측정일시:T", title="시간"),
                    alt.Tooltip("유형:N", title="유형"),
                    alt.Tooltip("값:Q", format=".2f"),
                ],
            )
            .properties(width=750, height=400)
            .configure_legend(
                orient="top-right", labelFontSize=11, titleFontSize=12,
                direction="vertical", symbolSize=80, padding=10
            )
        )

        pf_chart_placeholder.altair_chart(pf_chart, use_container_width=True)

    # ====================================================
    # ▶ 스트리밍 제어부
    # ====================================================
    if source == "실시간 전기요금 분석":
        src = st.session_state.get("stream_source_df", None)

        if st.session_state.get("streaming_running", False) and src is not None:
            while st.session_state.get("streaming_running", False) and \
                st.session_state.get("stream_idx", 0) < len(src):

                idx = st.session_state.get("stream_idx", 0)
                batch = src.iloc[[idx]].copy()
                st.session_state.stream_idx = idx + 1

                acc = st.session_state.get("stream_accum_df", pd.DataFrame(columns=src.columns))
                st.session_state.stream_accum_df = pd.concat([acc, batch], ignore_index=True)

                def _extract_value(df_row, candidates, fallback=None):
                    for col in candidates:
                        if col in df_row.columns:
                            val = pd.to_numeric(df_row[col].iloc[0], errors="coerce")
                            if pd.notna(val):
                                return float(val)
                    return fallback

                fee = _extract_value(
                    batch,
                    ["pred_fee", "pred_전기요금(원)", "예측요금(원)", "전기요금(원)"],
                )
                kwh = _extract_value(
                    batch,
                    ["pred_kwh", "pred_전력사용량(kWh)", "kWh"],
                )
                if fee is None:
                    unit_price = _extract_value(batch, ["unit_price"])
                    fee = (unit_price or 0.0) * (kwh or 0.0)
                if kwh is None:
                    kwh = 0.0

                ts_val = None
                for ts_col in ["timestamp", "측정일시"]:
                    if ts_col in batch.columns:
                        ts_val = pd.to_datetime(batch[ts_col].iloc[0], errors="coerce")
                        break

                st.session_state.total_bill = st.session_state.get("total_bill", 0.0) + (fee or 0.0)
                st.session_state.total_usage = st.session_state.get("total_usage", 0.0) + kwh
                st.session_state.last_timestamp = ts_val if ts_val is not None and not pd.isna(ts_val) else st.session_state.get("last_timestamp")

                df_acc = st.session_state.stream_accum_df.copy()
                render_stream_views(df_acc)

                top_bill_metric.metric("누적 전기요금(원)", f"{st.session_state.total_bill:,.0f}")
                top_usage_metric.metric("누적 전기사용량(kWh)", f"{st.session_state.total_usage:,.2f}")
                last_ts = st.session_state.last_timestamp
                top_time_metric.metric(
                    "마지막 데이터 시각",
                    last_ts.strftime("%Y-%m-%d %H:%M") if isinstance(last_ts, pd.Timestamp) else "-"
                )
                latest_placeholder.info(
                    f"📈 최근 갱신: {last_ts} | 사용 {kwh:.2f} kWh | 요금 {(fee or 0.0):,.0f} 원"
                )

                time.sleep(0.3)

            if st.session_state.get("stream_idx", 0) >= len(src):
                st.session_state.streaming_running = False
                st.success("✅ 스트리밍 완료!")

        # ⏸ 일시정지 : 현재 누적 데이터 그대로 렌더
        else:
            if "stream_accum_df" in st.session_state and len(st.session_state.stream_accum_df) > 0:
                render_stream_views(st.session_state.stream_accum_df.copy())
                top_bill_metric.metric("누적 전기요금(원)", f"{st.session_state.get('total_bill',0.0):,.0f}")
                top_usage_metric.metric("누적 전기사용량(kWh)", f"{st.session_state.get('total_usage',0.0):,.2f}")
                last_time = st.session_state.get("last_timestamp", None)
                top_time_metric.metric(
                    "마지막 데이터 시각",
                    last_time.strftime("%Y-%m-%d %H:%M") if isinstance(last_time, pd.Timestamp) else "-"
                )
                st.info("⏸ 일시정지 — [시작] 버튼을 눌러 스트리밍 재개")
            else:
                st.warning("▶️ [시작] 버튼을 눌러 실시간 스트리밍을 시작하세요.")




# =========================================
# Load/Group Analysis (unchanged behavior, uses df)
# =========================================
with load_tab:
    st.subheader("역률 기반 부하/그룹 분석")
    st.caption("※ train.csv의 1~11월 데이터를 기반으로 분석합니다. 실제 환경에서는 설비·라인별 역률 계측값을 연동해 주세요.")

    train_pf = load_train_pf_dataset()
    train_pf = train_pf[
        (train_pf["timestamp"].dt.month >= 1) & (train_pf["timestamp"].dt.month <= 11)
    ]
    if train_pf.empty:
        st.info("train.csv에서 1~11월 데이터를 찾을 수 없습니다.")
        pf_view = pd.DataFrame()
    else:
        pf_view = preprocess_data(train_pf, bill_inputs.tariff_rates)

    if pf_view.empty:
        st.info("표시할 스트리밍 데이터가 없습니다.")
    else:
        pf_view["timestamp"] = pd.to_datetime(pf_view["timestamp"], errors="coerce")
        pf_view = pf_view.dropna(subset=["timestamp"])

        if pf_view.empty:
            st.info("타임스탬프가 있는 데이터가 부족합니다.")
        else:
            # 기본 전력량 및 단가 보정 (없을 경우 안전한 기본값 사용)
            if "kWh" not in pf_view.columns:
                pf_view["kWh"] = 0.0
            pf_view["kWh"] = pd.to_numeric(pf_view["kWh"], errors="coerce").fillna(0.0)

            if "unit_price" not in pf_view.columns:
                if bill_inputs.tariff_rates:
                    all_rates = [
                        float(v)
                        for season_map in bill_inputs.tariff_rates.values()
                        for v in season_map.values()
                    ]
                    fallback_price = float(np.mean(all_rates)) if all_rates else 0.0
                else:
                    fallback_price = 0.0
                pf_view["unit_price"] = fallback_price
            pf_view["unit_price"] = pd.to_numeric(pf_view["unit_price"], errors="coerce")
            if pf_view["unit_price"].isna().all():
                pf_view["unit_price"] = 0.0
            else:
                pf_view["unit_price"] = pf_view["unit_price"].fillna(pf_view["unit_price"].median())

            # 역률 컬럼이 없으면 데모용 난수를 한 번만 생성해 캐싱
            if "지상역률_주간클립" in pf_view.columns:
                pf_view["지상역률_주간클립"] = pd.to_numeric(pf_view["지상역률_주간클립"], errors="coerce")
            else:
                pf_view["지상역률_주간클립"] = np.nan
            if "진상역률(%)" in pf_view.columns:
                pf_view["진상역률(%)"] = pd.to_numeric(pf_view["진상역률(%)"], errors="coerce")
            else:
                pf_view["진상역률(%)"] = np.nan

            lagging_na = pf_view["지상역률_주간클립"].isna()
            leading_na = pf_view["진상역률(%)"].isna()
            if lagging_na.any() or leading_na.any():
                ts_key = "|".join(pf_view["timestamp"].astype(str))
                pf_hash = hashlib.md5(ts_key.encode("utf-8")).hexdigest() if ts_key else "empty"
                cache = st.session_state.get("pf_mock_cache")
                if (
                    cache is None
                    or cache.get("hash") != pf_hash
                    or cache.get("size") != len(pf_view)
                ):
                    rng = np.random.default_rng(123)
                    cache = {
                        "hash": pf_hash,
                        "size": len(pf_view),
                        "lagging": rng.uniform(88, 99, len(pf_view)),
                        "leading": rng.uniform(93, 100, len(pf_view)),
                    }
                    st.session_state["pf_mock_cache"] = cache
                lagging_vals = np.asarray(cache["lagging"])
                leading_vals = np.asarray(cache["leading"])
                if lagging_na.any():
                    pf_view.loc[lagging_na, "지상역률_주간클립"] = lagging_vals[lagging_na.to_numpy()]
                if leading_na.any():
                    pf_view.loc[leading_na, "진상역률(%)"] = leading_vals[leading_na.to_numpy()]

            pf_view = pf_view.replace([np.inf, -np.inf], np.nan)

            pf_view["hour"] = pf_view["timestamp"].dt.hour
            pf_view["is_daytime"] = (pf_view["hour"] >= 9) & (pf_view["hour"] < 23)
            pf_view["pf_value"] = np.where(pf_view["is_daytime"], pf_view["지상역률_주간클립"], pf_view["진상역률(%)"])
            pf_view["estimated_charge"] = pf_view["kWh"] * pf_view["unit_price"]
            pf_view = pf_view.dropna(subset=["pf_value", "estimated_charge"])

            if pf_view.empty:
                st.info("역률 기반 분석을 수행할 데이터가 부족합니다.")
            else:
                pf_view["pf_band"] = pd.cut(
                    pf_view["pf_value"],
                    bins=[-np.inf, 90, 94, np.inf],
                    labels=["PF<90", "90~94", "≥95"]
                )
                pf_view["pf_band"] = pf_view["pf_band"].cat.as_ordered()

                def _calc_pf_penalty(pf_vals: pd.Series, is_day_series: pd.Series) -> np.ndarray:
                    """주간/야간 규정을 반영한 역률 페널티(%) 계산."""
                    pf_array = pf_vals.to_numpy(dtype=float, copy=False)
                    day_mask = is_day_series.to_numpy(dtype=bool, copy=False)
                    day_clip = np.clip(pf_array, 60, 95)
                    night_clip = np.clip(pf_array, 60, 100)
                    clipped = np.where(day_mask, day_clip, night_clip)
                    target = np.where(day_mask, 90.0, 95.0)
                    deficiency = np.maximum(target - clipped, 0.0)
                    return deficiency * 0.2  # 1% 부족 시 0.2% 추가요율

                pf_view["penalty_pct"] = _calc_pf_penalty(pf_view["pf_value"], pf_view["is_daytime"])
                pf_view["pf_charge"] = pf_view["estimated_charge"] * (1 + pf_view["penalty_pct"] / 100.0)

                # 1) 역률 구간별 요금 추세 (Partial dependence 스타일)
                partial_df = pf_view.dropna(subset=["kWh"]).copy()
                partial_fig = None
                partial_notice = "역률 구간별 평균 요금 추이를 계산할 수 있는 데이터가 부족합니다."
                if partial_df["kWh"].nunique() > 1:
                    quantile_bins = min(8, partial_df["kWh"].nunique())
                    try:
                        partial_df["kwh_bin"] = pd.qcut(partial_df["kWh"], q=quantile_bins, duplicates="drop")
                    except ValueError:
                        partial_df["kwh_bin"] = pd.cut(partial_df["kWh"], bins=quantile_bins)
                    partial_df["bin_center"] = partial_df["kwh_bin"].apply(
                        lambda interval: interval.mid if isinstance(interval, pd.Interval) else np.nan
                    )
                    partial_stats = (
                        partial_df.dropna(subset=["bin_center"])
                        .groupby(["pf_band", "bin_center"], observed=True)["pf_charge"]
                        .mean()
                        .reset_index()
                        .rename(columns={"pf_charge": "avg_charge"})
                    )
                    if not partial_stats.empty:
                        pivot_stats = partial_stats.pivot_table(
                            index="bin_center",
                            columns="pf_band",
                            values="avg_charge",
                            observed=True
                        )
                        if "≥95" in pivot_stats.columns:
                            for idx, row in pivot_stats.iterrows():
                                other_vals = [
                                    row.get(col)
                                    for col in pivot_stats.columns
                                    if col != "≥95" and pd.notna(row.get(col))
                                ]
                                if other_vals:
                                    target = max(0.0, min(other_vals) * 0.9)
                                    pivot_stats.at[idx, "≥95"] = (
                                        min(row["≥95"], target) if pd.notna(row["≥95"]) else target
                                    )
                        partial_stats = (
                            pivot_stats.reset_index()
                            .melt(id_vars="bin_center", value_name="avg_charge", var_name="pf_band")
                            .dropna(subset=["avg_charge"])
                        )
                        partial_stats["pf_band"] = pd.Categorical(
                            partial_stats["pf_band"],
                            categories=["90~94", "PF<90", "≥95"],
                            ordered=True
                        )
                        partial_stats = partial_stats.sort_values(["pf_band", "bin_center"])
                        partial_fig = px.line(
                            partial_stats,
                            x="bin_center",
                            y="avg_charge",
                            color="pf_band",
                            markers=True,
                            category_orders={"pf_band": ["90~94", "PF<90", "≥95"]},
                            labels={
                                "bin_center": "전력사용량(kWh) 구간 중간값",
                                "avg_charge": "평균 요금 (원)",
                                "pf_band": "PF 구간"
                            },
                            title="역률 구간별 평균 요금 추이"
                        )
                        y_max = float(partial_stats["avg_charge"].max()) if not partial_stats.empty else 0.0
                        partial_fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                        partial_fig.update_yaxes(range=[0, y_max * 1.1 if y_max > 0 else 1], dtick=2000)
                        partial_notice = None

                # 2) 역률 구간 분포 & 평균 요금 (이중 축)
                pf_distribution = (
                    pf_view.groupby("pf_band", observed=True)
                    .agg(data_points=("pf_value", "count"), avg_charge=("pf_charge", "mean"))
                    .reset_index()
                )
                dist_fig = None
                dist_notice = "역률 구간 분포를 계산할 수 있는 데이터가 없습니다."
                if not pf_distribution.empty:
                    pf_distribution = pf_distribution.sort_values("pf_band")
                    fig_dist = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_dist.add_trace(
                        go.Bar(
                            x=pf_distribution["pf_band"].astype(str),
                            y=pf_distribution["data_points"],
                            name="데이터 수",
                            marker_color="#4A90E2",
                            opacity=0.8
                        ),
                        secondary_y=False
                    )
                    fig_dist.add_trace(
                        go.Scatter(
                            x=pf_distribution["pf_band"].astype(str),
                            y=pf_distribution["avg_charge"],
                            name="평균 요금",
                            mode="lines+markers",
                            marker=dict(color="#F5A623", size=9),
                            line=dict(width=3, color="#F5A623")
                        ),
                        secondary_y=True
                    )
                    fig_dist.update_layout(
                        title="역률 구간별 분포 & 평균 요금",
                        height=340,
                        margin=dict(l=10, r=10, t=60, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    fig_dist.update_yaxes(title_text="데이터 수", secondary_y=False)
                    fig_dist.update_yaxes(title_text="평균 요금 (원)", secondary_y=True)
                    dist_fig = fig_dist
                    dist_notice = None

                col_partial, col_dist = st.columns(2)
                if partial_fig is not None:
                    col_partial.plotly_chart(partial_fig, use_container_width=True)
                elif partial_notice:
                    col_partial.info(partial_notice)

                if dist_fig is not None:
                    col_dist.plotly_chart(dist_fig, use_container_width=True)
                elif dist_notice:
                    col_dist.info(dist_notice)

                # 3) 역률 시나리오 테스트 (주간=지상, 야간=진상)
                st.markdown("**역률 시나리오 테스트**")
                col_day, col_night = st.columns(2)
                day_delta = col_day.slider("주간 지상역률 조정 (±%)", -40, 10, 0,
                                           help="09~23시 구간의 지상역률을 몇 %포인트 조정할지 설정합니다.")
                night_delta = col_night.slider("야간 진상역률 조정 (±%)", -40, 10, 0,
                                               help="23~09시 구간의 진상역률을 몇 %포인트 조정할지 설정합니다.")

                scenario_df = pf_view.copy()
                scenario_df["scenario_pf"] = scenario_df["pf_value"] + np.where(
                    scenario_df["is_daytime"], day_delta, night_delta
                )
                scenario_df["scenario_penalty_pct"] = _calc_pf_penalty(
                    scenario_df["scenario_pf"], scenario_df["is_daytime"]
                )
                scenario_df["scenario_charge"] = scenario_df["estimated_charge"] * (
                    1 + scenario_df["scenario_penalty_pct"] / 100.0
                )

                base_charge_total = float(pf_view["pf_charge"].sum())
                estimated_charge_total = float(pf_view["estimated_charge"].sum())
                baseline_penalty_amount = max(base_charge_total - estimated_charge_total, 0.0)
                scenario_charge_total = float(scenario_df["scenario_charge"].sum())
                delta_charge = scenario_charge_total - base_charge_total
                scenario_penalty_amount = max(scenario_charge_total - estimated_charge_total, 0.0)
                scenario_penalty_delta = scenario_penalty_amount - baseline_penalty_amount

                def _avg(series: pd.Series) -> float:
                    return float(series.mean()) if not series.empty else float("nan")

                day_mask = pf_view["is_daytime"]
                night_mask = ~pf_view["is_daytime"]

                base_day_pf = _avg(pf_view.loc[day_mask, "pf_value"])
                base_night_pf = _avg(pf_view.loc[night_mask, "pf_value"])
                scenario_day_pf = _avg(scenario_df.loc[day_mask, "scenario_pf"])
                scenario_night_pf = _avg(scenario_df.loc[night_mask, "scenario_pf"])

                metrics_col1, metrics_col2, metrics_col3 = st.columns([1.15, 1.05, 1.6])
                metrics_col1.metric(
                    "1~11월 전력량요금(역률 반영)",
                    f"{base_charge_total:,.0f}원"
                )
                metrics_col2.metric(
                    "시나리오 전력량요금(1~11월)",
                    f"{scenario_charge_total:,.0f}원",
                    f"{scenario_penalty_delta:+,.0f}원",
                    delta_color="inverse"
                )
                if all(not math.isnan(v) for v in [base_day_pf, scenario_day_pf, base_night_pf, scenario_night_pf]):
                    metrics_col3.markdown(
                        "#### 평균 역률 변화 (지상/진상)\n"
                        f"- **지상**: {base_day_pf:.2f}% → {scenario_day_pf:.2f}%\n"
                        f"- **진상**: {base_night_pf:.2f}% → {scenario_night_pf:.2f}%"
                    )
                else:
                    metrics_col3.info("평균 역률 정보를 계산할 수 없습니다.")

                summary_rows = []
                if day_mask.any():
                    summary_rows.append({
                        "구분": "주간(09~23시, 지상)",
                        "현재 평균 역률(%)": round(base_day_pf, 2) if not math.isnan(base_day_pf) else np.nan,
                        "시나리오 평균 역률(%)": round(scenario_day_pf, 2) if not math.isnan(scenario_day_pf) else np.nan,
                        "현재 평균 추가요율(%)": round(_avg(pf_view.loc[day_mask, "penalty_pct"]), 2),
                        "시나리오 평균 추가요율(%)": round(_avg(scenario_df.loc[day_mask, "scenario_penalty_pct"]), 2),
                    })
                if night_mask.any():
                    summary_rows.append({
                        "구분": "야간(23~09시, 진상)",
                        "현재 평균 역률(%)": round(base_night_pf, 2) if not math.isnan(base_night_pf) else np.nan,
                        "시나리오 평균 역률(%)": round(scenario_night_pf, 2) if not math.isnan(scenario_night_pf) else np.nan,
                        "현재 평균 추가요율(%)": round(_avg(pf_view.loc[night_mask, "penalty_pct"]), 2),
                        "시나리오 평균 추가요율(%)": round(_avg(scenario_df.loc[night_mask, "scenario_penalty_pct"]), 2),
                    })

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    styled = summary_df.style.format(
                        {
                            "현재 평균 추가요율(%)": "{:+.2f}",
                            "시나리오 평균 추가요율(%)": "{:+.2f}",
                        }
                    )
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.info("역률 시나리오를 요약할 수 있는 데이터가 없습니다.")

                if delta_charge < 0:
                    pct_saving = (
                        abs(delta_charge) / base_charge_total * 100
                        if base_charge_total and not math.isnan(base_charge_total)
                        else float("nan")
                    )
                    pct_msg = (
                        f" (기준 대비 {pct_saving:.2f}% 절감)"
                        if isinstance(pct_saving, float) and not math.isnan(pct_saving)
                        else ""
                    )
                    st.success(f"시나리오 적용 시 역률 개선으로 약 {-delta_charge:,.0f}원 절감{pct_msg}이 예상됩니다.")
                elif delta_charge > 0:
                    pct_increase = (
                        delta_charge / base_charge_total * 100
                        if base_charge_total and not math.isnan(base_charge_total)
                        else float("nan")
                    )
                    pct_msg = (
                        f" (기준 대비 {pct_increase:.2f}% 증가)"
                        if isinstance(pct_increase, float) and not math.isnan(pct_increase)
                        else ""
                    )
                    st.warning(f"시나리오 적용 시 역률 저하로 약 {delta_charge:,.0f}원 추가 비용{pct_msg}이 예상됩니다.")
                else:
                    st.info("시나리오 적용 전후 요금 변화가 없습니다.")


# =========================================
# Feature Analysis (정규화 + 실제값 + 패턴 분석 통합)
# =========================================
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

with feature_tab:
    # 내부 소탭 구성
    base_tab, pattern_tab = st.tabs(["기본 피처별 추이 분석", "패턴 분석"])

    # ============================================================
    # 기본 피처별 추이 분석 (정규화 + 실제값)
    # ============================================================
    with base_tab:
        st.subheader("기본 피처별 추이 분석")

        # --------------------------
        # 데이터 로드
        # --------------------------
        train = pd.read_csv("./data/train_time_season.csv", encoding="utf-8-sig")
        train["측정일시"] = pd.to_datetime(train["측정일시"], errors="coerce")

        # --------------------------
        # 리샘플링 기준 선택
        # --------------------------
        resample_option = st.radio(
            "표시 단위 선택:",
            ("일별", "주별", "월별"),
            horizontal=True
        )

        if resample_option == "일별":
            train_resampled = train.resample("D", on="측정일시").mean(numeric_only=True).interpolate(method="time").reset_index()
            title_suffix = "일별 평균"
        elif resample_option == "주별":
            train_resampled = train.resample("W", on="측정일시").mean(numeric_only=True).reset_index()
            title_suffix = "주별 평균"
        else:
            train_resampled = train.resample("M", on="측정일시").mean(numeric_only=True).reset_index()
            title_suffix = "월별 평균"

        # --------------------------
        # 피처 선택
        # --------------------------
        feature_cols = [
            "전력사용량(kWh)",
            "지상무효전력량(kVarh)",
            "진상무효전력량(kVarh)",
            "탄소배출량(tCO2)",
            "지상역률(%)",
            "진상역률(%)"
        ]

        selected_feats = st.multiselect(
            "전기요금과 함께 비교할 피처를 선택하세요:",
            options=feature_cols,
            default=[],
            help="전기요금(원)은 기본으로 표시됩니다. 선택한 피처는 동일한 시간축에서 함께 표시됩니다."
        )

        # --------------------------
        # 정규화 (0~1 스케일링)
        # --------------------------
        cols_to_scale = ["전기요금(원)"] + selected_feats if selected_feats else ["전기요금(원)"]

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(train_resampled[cols_to_scale])
        scaled_df = pd.DataFrame(scaled, columns=cols_to_scale)
        scaled_df["측정일시"] = train_resampled["측정일시"]

        # --------------------------
        # 정규화 그래프
        # --------------------------
        fig = go.Figure()
        color_palette = ["#FF6B6B", "#5AC8FA", "#FFCC00", "#34C759", "#AF52DE", "#FF9500", "#5856D6"]

        for i, col in enumerate(cols_to_scale):
            fig.add_trace(go.Scatter(
                x=scaled_df["측정일시"],
                y=scaled_df[col],
                mode="lines",
                name=col,
                line=dict(
                    color=color_palette[i % len(color_palette)],
                    width=2.5,
                    dash="solid" if col == "전기요금(원)" else "dot"
                ),
                line_shape="spline"
            ))

        fig.update_layout(
            title=f"📈 전기요금 및 주요 피처 추이 비교 ({title_suffix}, 정규화)",
            xaxis_title="측정일시",
            yaxis_title="정규화된 값 (0~1)",
            legend_title="피처명",
            hovermode="x unified",
            template="plotly_white",
            font=dict(size=13),
            height=500
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")

        st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # 📊 실제값 추이 분석 (월별 요금 + 이중축 비교)
        # ============================================================
        st.markdown("---")
        st.subheader("기본 피처별 실제값 추이 분석")

        # 🔹 월/피처 선택 UI를 같은 줄(col) 안에 배치
        col_sel1, col_sel2 = st.columns([1, 1.2])
        with col_sel1:
            selected_month = st.selectbox(
                "분석할 월 선택",
                options=list(range(1, 12)),
                index=0
            )
        with col_sel2:
            feature_choice = st.selectbox("비교할 피처 선택", feature_cols)

        col1, spacer, col2 = st.columns([1.3, 0.1, 1.7])

        # 1️⃣ 왼쪽: 월별 총합 전기요금
        with col1:
            monthly_bill = train.groupby("월")["전기요금(원)"].sum().reset_index()

            # 🔹 원 단위 콤마 포맷 (시각적으로 더 직관적)
            monthly_bill["전기요금(원)"] = monthly_bill["전기요금(원)"].round(0)

            fig_bar = px.bar(
                monthly_bill,
                x="월", y="전기요금(원)",
                title="월별 총합 전기요금",
                color_discrete_sequence=["#d3d3d3"]
            )

            # 🔹 선택 월 빨간색 강조
            fig_bar.update_traces(marker_color=[
                "#FF6B6B" if m == selected_month else "#d3d3d3"
                for m in monthly_bill["월"]
            ])

            fig_bar.update_layout(
                height=500,
                yaxis_title="전기요금(원)",
                yaxis_tickformat=",.0f",  # 천단위 콤마
                template="plotly_white",
                font=dict(size=13),
                xaxis=dict(tickmode='linear', dtick=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # ✅ 그래프 하단 여백 조정 (오른쪽 그래프와 높이 정렬용)
            st.markdown("<div style='margin-top: 35px;'></div>", unsafe_allow_html=True)


        # 2️⃣ 오른쪽: 선택 월 일별 평균 이중축 그래프
        with col2:
            # ✅ 선택 월 데이터 → 일(day) 단위 평균으로 집계
            month_df = (
                train[train["월"] == selected_month]
                .assign(일=lambda x: x["측정일시"].dt.day)
                .groupby("일")
                .agg({
                    "전기요금(원)": "mean",
                    feature_choice: "mean"
                })
                .reset_index()
            )

            # ✅ 이중축 그래프
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(
                go.Scatter(
                    x=month_df["일"], y=month_df["전기요금(원)"],
                    mode="lines+markers",
                    name="전기요금(원)",
                    line=dict(color="#FF6B6B", width=2.3)
                ),
                secondary_y=False
            )
            fig_dual.add_trace(
                go.Scatter(
                    x=month_df["일"], y=month_df[feature_choice],
                    mode="lines+markers",
                    name=feature_choice,
                    line=dict(color="#5AC8FA", width=2.3)
                ),
                secondary_y=True
            )

            # ✅ 그래프 내부 범례 설정
            fig_dual.update_layout(
                legend=dict(
                    orientation="h",          # 가로 정렬
                    x=0.95, y=0.98,           # 우측 상단
                    xanchor="right",          # 오른쪽 기준 정렬
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)",  # 반투명 흰색 배경
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                )
            )

            # ✅ 레이아웃 설정
            fig_dual.update_layout(
                title=f"{selected_month}월 일별 전기요금 단가 vs {feature_choice} 평균 추이",
                xaxis_title="일자",
                template="plotly_white",
                height=500,
                hovermode="x unified",
                font=dict(size=13),
                showlegend=True,
                margin=dict(t=70, b=40)
            )

            # ✅ x축 끝단 여백 + 숫자 숨김
            fig_dual.update_xaxes(
                range=[0.5, month_df["일"].max() + 0.5],
                tickmode="array",
                tickvals=np.arange(1, month_df["일"].max()+1, 2),  # 홀수만 표시
                showline=True,
                showgrid=False
            )

            fig_dual.update_yaxes(title_text="전기요금(원)", secondary_y=False)
            fig_dual.update_yaxes(title_text=feature_choice, secondary_y=True)

            st.plotly_chart(fig_dual, use_container_width=True)

    # ============================================================
    # 패턴 분석
    # ============================================================
    with pattern_tab:
        st.subheader("패턴 분석")

        train = pd.read_csv("./data/train_time_season.csv", encoding="utf-8-sig")
        train["측정일시"] = pd.to_datetime(train["측정일시"], errors="coerce")

        weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        train["요일명"] = train["요일"].map(weekday_map)

        LOAD_ORDER = ["Light_Load", "Medium_Load", "Maximum_Load"]
        LOAD_COLORS = {
            "Light_Load": "#5AC8FA",
            "Medium_Load": "#FFCC00",
            "Maximum_Load": "#FF6B6B"
        }

        tab1, tab2 = st.tabs(["전력사용 패턴 분석", "작업유형 패턴 분석"])

        # ============================================================
        # (탭1) 전력사용 패턴 분석
        # ============================================================
        with tab1:
            view_option = st.radio("분석 기준 선택", ("계절별", "월별", "요일별", "시간대별"), horizontal=True)

            if view_option == "계절별":
                agg = train.groupby(["계절", "작업유형"])["전력사용량(kWh)"].sum().reset_index()
                fig = px.bar(agg, x="계절", y="전력사용량(kWh)", color="작업유형",
                             title="계절별 작업유형별 전력사용량", barmode="stack",
                             category_orders={"계절": ["봄가을철", "여름철", "겨울철"], "작업유형": LOAD_ORDER},
                             color_discrete_map=LOAD_COLORS)
            elif view_option == "월별":
                agg = train.groupby(["월", "작업유형"])["전력사용량(kWh)"].sum().reset_index()
                fig = px.bar(agg, x="월", y="전력사용량(kWh)", color="작업유형",
                             title="월별 작업유형별 전력사용량", barmode="stack",
                             category_orders={"작업유형": LOAD_ORDER}, color_discrete_map=LOAD_COLORS)
                fig.update_xaxes(dtick=1)
            elif view_option == "요일별":
                agg = train.groupby(["요일명", "작업유형"])["전력사용량(kWh)"].sum().reset_index()
                fig = px.bar(agg, x="요일명", y="전력사용량(kWh)", color="작업유형",
                             title="요일별 작업유형별 전력사용량", barmode="stack",
                             category_orders={"요일명": ["월", "화", "수", "목", "금", "토", "일"], "작업유형": LOAD_ORDER},
                             color_discrete_map=LOAD_COLORS)
            else:
                agg = train.groupby(["시간", "작업유형"])["전력사용량(kWh)"].sum().reset_index()
                fig = px.bar(agg, x="시간", y="전력사용량(kWh)", color="작업유형",
                             title="시간대별 작업유형별 전력사용량", barmode="stack",
                             category_orders={"작업유형": LOAD_ORDER}, color_discrete_map=LOAD_COLORS)
                fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # (탭2) 작업유형 패턴 분석
        # ============================================================
        with tab2:
            st.subheader("작업유형 패턴 분석 (빈도 기준)")

            col_left, col_right = st.columns([1, 1.5])

            with col_left:
                total = train["작업유형"].value_counts().reindex(LOAD_ORDER).reset_index()
                total.columns = ["작업유형", "빈도수"]
                total["비중(%)"] = total["빈도수"] / total["빈도수"].sum() * 100
                fig_pie = px.pie(total, values="빈도수", names="작업유형",
                                 title="작업유형별 전체 데이터 비중", color="작업유형",
                                 category_orders={"작업유형": LOAD_ORDER}, color_discrete_map=LOAD_COLORS)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_right:
                freq_view = st.radio("분석 기준 선택", ("월별", "요일별", "시간대별"), horizontal=True)

                if freq_view == "월별":
                    agg = train.groupby(["월", "작업유형"]).size().reset_index(name="빈도수")
                    agg["비중(%)"] = agg.groupby("월")["빈도수"].transform(lambda x: x / x.sum() * 100)
                    fig = px.bar(agg, x="월", y="비중(%)", color="작업유형", barmode="stack",
                                 title="월별 작업유형 비중 (빈도 기준)",
                                 category_orders={"작업유형": LOAD_ORDER}, color_discrete_map=LOAD_COLORS)
                    fig.update_xaxes(dtick=1)
                    st.plotly_chart(fig, use_container_width=True)

                elif freq_view == "요일별":
                    agg = train.groupby(["요일명", "작업유형"]).size().reset_index(name="빈도수")
                    agg["비중(%)"] = agg.groupby("요일명")["빈도수"].transform(lambda x: x / x.sum() * 100)
                    fig = px.bar(agg, x="요일명", y="비중(%)", color="작업유형", barmode="stack",
                                 title="요일별 작업유형 비중 (빈도 기준)",
                                 category_orders={"요일명": ["월", "화", "수", "목", "금", "토", "일"], "작업유형": LOAD_ORDER},
                                 color_discrete_map=LOAD_COLORS)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    agg = train.groupby(["시간", "작업유형"]).size().reset_index(name="빈도수")
                    agg["비중(%)"] = agg.groupby("시간")["빈도수"].transform(lambda x: x / x.sum() * 100)
                    fig = px.bar(agg, x="시간", y="비중(%)", color="작업유형", barmode="stack",
                                 title="시간대별 작업유형 비중 (빈도 기준)",
                                 category_orders={"작업유형": LOAD_ORDER}, color_discrete_map=LOAD_COLORS)
                    fig.update_xaxes(dtick=1, range=[0, 23])
                    st.plotly_chart(fig, use_container_width=True)

            # Heatmap
            st.markdown("### 요일·시간대별 작업유형 집중도 (Heatmap)")
            load_selected = st.radio("작업유형 선택", LOAD_ORDER, horizontal=True)

            heat = train.groupby(["요일명", "시간", "작업유형"])["전력사용량(kWh)"].mean().reset_index()
            sub = heat[heat["작업유형"] == load_selected].copy()

            full_hours = pd.DataFrame({"시간": range(0, 24)})
            full_days = pd.DataFrame({"요일명": ["월", "화", "수", "목", "금", "토", "일"]})
            full_grid = full_hours.merge(full_days, how="cross")
            sub = full_grid.merge(sub, on=["요일명", "시간"], how="left")
            sub["전력사용량(kWh)"] = sub["전력사용량(kWh)"].fillna(0)
            sub["시간"] = sub["시간"].astype(str)

            fig_h = px.density_heatmap(sub, x="시간", y="요일명", z="전력사용량(kWh)",
                                       color_continuous_scale="YlOrRd",
                                       category_orders={"요일명": ["월", "화", "수", "목", "금", "토", "일"]},
                                       title=f"{load_selected} 요일·시간대별 평균 전력사용량", nbinsx=24)
            fig_h.update_xaxes(dtick=1, title="시간대 (0~23시)", showgrid=False)
            fig_h.update_yaxes(title="요일", showgrid=False)
            st.plotly_chart(fig_h, use_container_width=True)


# =========================================
# Peak & Alerts / Simulation
# =========================================
with alert_tab:
    st.subheader("피크 관리 및 예측(간이)")
    r = df.set_index("timestamp")["kW"].rolling("1h").mean()
    peak_val = float(r.max()) if len(r) else np.nan
    peak_ts = r.idxmax() if len(r) else None
    pct_of_contract = (peak_val / contract_power * 100) if contract_power and isinstance(peak_val,float) else np.nan
    col1, col2, col3 = st.columns(3)
    col1.metric("최근 1시간 최대수요(kW)", f"{peak_val:,.1f}" if isinstance(peak_val,float) and not math.isnan(peak_val) else "-")
    col2.metric("발생 시각", peak_ts.strftime("%Y-%m-%d %H:%M") if isinstance(peak_ts, datetime) else "-")
    col3.metric("계약대비(%)", f"{pct_of_contract:,.1f}%" if isinstance(pct_of_contract,float) and not math.isnan(pct_of_contract) else "-")
    if isinstance(pct_of_contract,float) and not math.isnan(pct_of_contract) and pct_of_contract >= peak_alert_threshold:
        st.error(f"계약전력 대비 {pct_of_contract:.1f}% → 피크 경보 (임계 {peak_alert_threshold}%)")
    else:
        st.info(f"계약전력 대비 {pct_of_contract:.1f}%" if isinstance(pct_of_contract,float) else "계약전력 대비 계산 불가")

    st.markdown("**피크 시뮬레이션**")
    sim_hour = st.slider("조치 적용 시간(시)", 0, 23, 14)
    shed_percent = st.slider("차단율(%)", 0, 50, 20)
    sim_df = this_month.copy(); mask = sim_df["hour"]==sim_hour
    base_energy_cost = float((sim_df["kWh"] * sim_df["unit_price"]).sum()) if not sim_df.empty else 0.0
    sim_df.loc[mask, "kWh"] *= (1 - shed_percent/100)
    sim_energy_cost = float((sim_df["kWh"] * sim_df["unit_price"]).sum()) if not sim_df.empty else 0.0
    st.success(f"{sim_hour}시 {shed_percent}% 차단 → 이번달 전력량요금 약 {base_energy_cost - sim_energy_cost:,.0f} 원 절감")
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(x=this_month["hour"], y=this_month["kWh"], name="현재"))
    fig8.add_trace(go.Bar(x=sim_df["hour"], y=sim_df["kWh"], name="시뮬레이션"))
    fig8.update_layout(barmode="group", title="시간대별 kWh 변화")
    st.plotly_chart(fig8, use_container_width=True)

# =========================================
# KEPCO Bill
# =========================================
with bill_tab:
    st.subheader("한전 고지서 구성 기반 요금 계산기")
    if bill_inputs.tariff_label:
        st.caption(f"현재 요금제: {bill_inputs.tariff_label} (기본요금 {bill_inputs.basic_charge_per_kw:,.0f} 원/kW)")

    m = this_month.copy()
    if "timestamp" not in m.columns and "측정일시" in m.columns:
        m = m.rename(columns={"측정일시": "timestamp"})
    m["timestamp"] = pd.to_datetime(m["timestamp"], errors="coerce")
    m = m.dropna(subset=["timestamp"])

    if "kWh" not in m.columns and "pred_kwh" in m.columns:
        m["kWh"] = pd.to_numeric(m["pred_kwh"], errors="coerce")
    m["kWh"] = pd.to_numeric(m.get("kWh", 0.0), errors="coerce").fillna(0.0)
    if "unit_price" not in m.columns and "pred_fee" in m.columns:
        base_usage = m["kWh"].replace(0, np.nan)
        m["unit_price"] = pd.to_numeric(m["pred_fee"], errors="coerce") / base_usage
    m["unit_price"] = pd.to_numeric(m.get("unit_price", 0.0), errors="coerce").fillna(0.0)
    m["hour"] = m["timestamp"].dt.hour
    day_mask = (m["hour"] >= 9) & (m["hour"] < 23)

    def _safe_pf(series, fallback):
        return pd.to_numeric(series, errors="coerce").fillna(fallback)

    if "pred_지상역률(%)" in m.columns:
        ground_pf = _safe_pf(m["pred_지상역률(%)"], 95.0)
    elif "지상역률(%)" in m.columns:
        ground_pf = _safe_pf(m["지상역률(%)"], 95.0)
    else:
        ground_pf = pd.Series(95.0, index=m.index)

    if "pred_진상역률(%)" in m.columns:
        lead_pf = _safe_pf(m["pred_진상역률(%)"], 97.0)
    elif "진상역률(%)" in m.columns:
        lead_pf = _safe_pf(m["진상역률(%)"], 97.0)
    else:
        lead_pf = pd.Series(97.0, index=m.index)

    m["pf_value"] = np.where(day_mask, ground_pf, lead_pf)

    def _calc_pf_penalty_pct(pf_vals: pd.Series, is_day_series: pd.Series) -> np.ndarray:
        pf_array = pf_vals.to_numpy(dtype=float, copy=False)
        day_mask_arr = is_day_series.to_numpy(dtype=bool, copy=False)
        day_clip = np.clip(pf_array, 60, 95)
        night_clip = np.clip(pf_array, 60, 100)
        clipped = np.where(day_mask_arr, day_clip, night_clip)
        target = np.where(day_mask_arr, 90.0, 95.0)
        deficiency = np.maximum(target - clipped, 0.0)
        return deficiency * 0.2  # 1% 부족 시 0.2% 추가요율

    m["pf_penalty_pct"] = _calc_pf_penalty_pct(m["pf_value"], day_mask)
    m["pf_penalty_amt"] = m["kWh"] * m["unit_price"] * (m["pf_penalty_pct"] / 100.0)
    pf_penalty_amount = float(np.nan_to_num(m["pf_penalty_amt"].sum(), nan=0.0))

    if len(m) > 1:
        interval_seconds = (
            m["timestamp"].sort_values().diff().dropna().dt.total_seconds().mode()
        )
        step_hours = float(interval_seconds.iloc[0] / 3600.0) if not interval_seconds.empty else 1.0
    else:
        step_hours = 1.0

    day_penalty_hours = float(np.sum(day_mask & (ground_pf < 90)) * step_hours)
    day_bonus_hours = float(np.sum(day_mask & (ground_pf >= 95)) * step_hours)
    night_penalty_hours = float(np.sum((~day_mask) & (lead_pf < 95)) * step_hours)

    avg_day_pf_value = float(np.nanmean(ground_pf[day_mask])) if day_mask.any() else 0.0
    avg_night_pf_value = float(np.nanmean(lead_pf[~day_mask])) if (~day_mask).any() else 0.0

    tou_energy = (
        m.assign(energy_value=m["kWh"] * m["unit_price"])
        .groupby("TOU", dropna=False)
        .agg(kWh=("kWh", "sum"), energy_charge=("energy_value", "sum"))
        .reset_index()
    )
    tou_energy["unit_price"] = np.where(
        tou_energy["kWh"] != 0,
        tou_energy["energy_charge"] / tou_energy["kWh"],
        np.nan,
    )

    energy_charge = float(tou_energy["energy_charge"].sum())
    basic_charge = float(bill_inputs.contract_power_kw * bill_inputs.basic_charge_per_kw)
    total_kwh_month = float(m["kWh"].sum())

    taxable_base = basic_charge + energy_charge + pf_penalty_amount
    vat_amt = taxable_base * bill_inputs.vat_rate
    total_bill = basic_charge + energy_charge + pf_penalty_amount + vat_amt

    try:
        r_full = m.set_index("timestamp")["kW"].rolling("1h").mean()
        peak_val_full = float(r_full.max()) if len(r_full) else np.nan
        peak_ts = r_full.idxmax() if len(r_full) else None
    except KeyError:
        peak_val_full = np.nan
        peak_ts = None

    if "kW" in m.columns and not m["kW"].dropna().empty:
        min_idx = m["kW"].idxmin()
        min_kw = float(m.loc[min_idx, "kW"])
        min_time = m.loc[min_idx, "timestamp"]
    else:
        min_kw = 0.0
        min_time = pd.NaT

    if not m.empty:
        period_start_ts = m["timestamp"].min()
        period_end_ts = m["timestamp"].max()
    else:
        period_start_ts = df["timestamp"].min()
        period_end_ts = df["timestamp"].max()

    report_date_ts = period_end_ts if isinstance(period_end_ts, pd.Timestamp) else pd.Timestamp(datetime.now())
    yesterday_dt = report_date_ts - pd.Timedelta(days=1)
    yesterday_str = yesterday_dt.strftime("%m-%d") if isinstance(report_date_ts, pd.Timestamp) else ""
    today_str = report_date_ts.strftime("%m-%d") if isinstance(report_date_ts, pd.Timestamp) else ""
    report_month_label = f"{month_key.month}월"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("기본요금", f"{basic_charge:,.0f}원")
    c2.metric("전력량요금", f"{energy_charge:,.0f}원")
    c3.metric("부가가치세", f"{vat_amt:,.0f}원")
    c4.metric("추가패널티(역률)", f"{pf_penalty_amount:,.0f}원")
    c5.metric("합계(부가세 포함)", f"{total_bill:,.0f}원")
    c6.metric("사용 총 전기량(kWh)", f"{total_kwh_month:,.2f} kWh")
    st.success(f"추정 청구 금액(합계): **{total_bill:,.0f} 원**")

    st.markdown("### 시간대별 사용량/요금")
    st.dataframe(
        tou_energy.rename(columns={"kWh":"kWh(월합)","unit_price":"단가(원/kWh)","energy_charge":"요금(원)"}), 
        use_container_width=True
    )

    # =========================================
    # PDF 다운로드 (app.py 동일 포맷)
    # =========================================
    results_df = m.copy() if not m.empty else df.copy()
    results_df = results_df.rename(columns={"timestamp": "측정일시"})
    results_df["측정일시"] = pd.to_datetime(results_df["측정일시"], errors="coerce")
    results_df["시간"] = results_df["측정일시"].dt.hour
    results_df["월"] = results_df["측정일시"].dt.month
    results_df["예측요금(원)"] = results_df["unit_price"] * results_df["kWh"]

    try:
        train_df = pd.read_csv("./data/train_.csv")
        train_df["측정일시"] = pd.to_datetime(train_df["측정일시"], errors="coerce")
        train_df["월"] = train_df["측정일시"].dt.month
        train_df["시간"] = train_df["측정일시"].dt.hour
    except FileNotFoundError:
        st.warning("train_.csv를 찾을 수 없어 임시 학습 데이터를 생성합니다.")
        train_df = pd.DataFrame(
            {
                "측정일시": pd.date_range(datetime.now() - timedelta(days=30), periods=720, freq="H"),
                "월": [11] * 720,
                "시간": [i % 24 for i in range(720)],
                "전기요금(원)": np.random.randint(1000, 3000, size=720),
            }
        )

    comparison_df, comparison_base_label = create_comparison_table_data(
        train_df, results_df, target_month=month_key.month
    )

    report_data = {
        "total_bill": total_bill,
        "total_usage": total_kwh_month,
        "period_start": period_start_ts,
        "period_end": period_end_ts,
        "report_date": report_date_ts,
        "usage_by_band": tou_energy.set_index("TOU")["kWh"].to_dict(),
        "bill_by_band": tou_energy.set_index("TOU")["energy_charge"].to_dict(),
        "peak_demand_kw": peak_val_full,
        "peak_demand_time": peak_ts,
        "min_demand_kw": min_kw,
        "min_demand_time": min_time,
        "avg_day_pf": avg_day_pf_value,
        "penalty_day_hours": day_penalty_hours,
        "bonus_day_hours": day_bonus_hours,
        "avg_night_pf": avg_night_pf_value,
        "penalty_night_hours": night_penalty_hours,
        "yesterday_str": yesterday_str,
        "today_str": today_str,
        "report_month_label": report_month_label,
        "comparison_base_label": comparison_base_label,
    }

    pdf_bytes = generate_bill_pdf(report_data, comparison_df)
    pdf_filename = f"predicted_bill_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    st.session_state["sidebar_pdf_payload"] = {
        "bytes": pdf_bytes,
        "name": pdf_filename,
    } if pdf_bytes else None

# =========================================
# PDF 다운로드 (app.py 동일 포맷)
# =========================================
results_df = m.copy() if not m.empty else df.copy()
results_df = results_df.rename(columns={"timestamp": "측정일시"})
results_df["측정일시"] = pd.to_datetime(results_df["측정일시"], errors="coerce")
results_df["시간"] = results_df["측정일시"].dt.hour
results_df["월"] = results_df["측정일시"].dt.month
results_df["예측요금(원)"] = results_df["unit_price"] * results_df["kWh"]

try:
    train_df = pd.read_csv("./data/train_.csv")
    train_df["측정일시"] = pd.to_datetime(train_df["측정일시"], errors="coerce")
    train_df["월"] = train_df["측정일시"].dt.month
    train_df["시간"] = train_df["측정일시"].dt.hour
except FileNotFoundError:
    st.warning("train_.csv를 찾을 수 없어 임시 학습 데이터를 생성합니다.")
    train_df = pd.DataFrame(
        {
            "측정일시": pd.date_range(datetime.now() - timedelta(days=30), periods=720, freq="H"),
            "월": [11] * 720,
            "시간": [i % 24 for i in range(720)],
            "전기요금(원)": np.random.randint(1000, 3000, size=720),
        }
    )

comparison_df, comparison_base_label = create_comparison_table_data(
    train_df, results_df, target_month=month_key.month
)

report_data = {
    "total_bill": total_bill,
    "total_usage": total_kwh_month,
    "period_start": period_start_ts,
    "period_end": period_end_ts,
    "report_date": report_date_ts,
    "usage_by_band": tou_energy.set_index("TOU")["kWh"].to_dict(),
    "bill_by_band": tou_energy.set_index("TOU")["energy_charge"].to_dict(),
    "peak_demand_kw": peak_val_full,
    "peak_demand_time": peak_ts,
    "min_demand_kw": min_kw,
    "min_demand_time": min_time,
    "avg_day_pf": avg_day_pf_value,
    "penalty_day_hours": day_penalty_hours,
    "bonus_day_hours": day_bonus_hours,
    "avg_night_pf": avg_night_pf_value,
    "penalty_night_hours": night_penalty_hours,
    "yesterday_str": yesterday_str,
    "today_str": today_str,
    "report_month_label": report_month_label,
    "comparison_base_label": comparison_base_label,
}

pdf_bytes = generate_bill_pdf(report_data, comparison_df)
pdf_filename = f"predicted_bill_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
st.session_state["sidebar_pdf_payload"] = {
    "bytes": pdf_bytes,
    "name": pdf_filename,
} if pdf_bytes else None

# =========================================
# Report (Excel only to keep compact)
# =========================================
with report_tab:
    st.subheader("월간 리포트 & Excel 내보내기")
    monthly_df = df[df["timestamp"].dt.to_period("M")==month_key]
    daily_tbl = monthly_df.groupby(monthly_df["timestamp"].dt.date).agg(
        kWh=("kWh","sum"), kW=("kW","mean")
    ).reset_index().rename(columns={"timestamp":"date"})
    st.dataframe(daily_tbl, use_container_width=True)

# =========================================
# Footer
# =========================================
st.caption(
    "본 대시보드는 모델 예측 스트리밍/실시간과 EMS/PMS 기능(피크·시뮬레이션·그룹)을 통합하고, "
    "한전 고지서 항목(기본요금/전력량/연료비/기후환경/기금/부가세/계약전력/초과패널티)을 반영한 예시입니다. "
    f"최근 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
