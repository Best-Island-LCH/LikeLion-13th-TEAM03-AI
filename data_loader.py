# data_loader.py
import os
import re
import pandas as pd

# 프로젝트 루트 기준 raw_data 폴더 사용
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")

# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------
def _read_csv_select(path, usecols_candidates=None, encodings=("cp949", "utf-8", "utf-8-sig")):
    """
    usecols_candidates:
      - None  -> 전체 컬럼 읽기
      - list  -> 후보가 1개 이상 매칭되면 그 컬럼만, 하나도 없으면 전체
    """
    last_err = None
    for enc in encodings:
        try:
            if usecols_candidates is None:
                return pd.read_csv(path, encoding=enc)

            header = pd.read_csv(path, nrows=0, encoding=enc)
            actual = set(header.columns)
            sel = [c for c in usecols_candidates if c in actual]
            # sel이 비면 전체, 비지 않으면 sel만
            return pd.read_csv(path, usecols=(sel if sel else None), encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read CSV with encodings {encodings}: {path}. Last error: {last_err}")


def _read_excel_safe(path, sheet_name=0, usecols=None):
    """엑셀을 안전하게 읽는다(시트명 오류 시 0번 시트로 폴백)."""
    try:
        return pd.read_excel(path, sheet_name=sheet_name, usecols=usecols)
    except ValueError:
        return pd.read_excel(path, sheet_name=0, usecols=usecols)

def _rename_safe(df, mapping):
    """mapping에 존재하는 키만 안전하게 rename"""
    if not isinstance(df, pd.DataFrame):
        return df
    submap = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=submap)

def _coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    """
    인구수 관련 열 전부 콤마 제거 + 숫자화.
    (직장/유동/상주 공통)
    """
    if not isinstance(df, pd.DataFrame):
        return df
    patt = re.compile(r"(유동인구수|상주인구수|직장인구수)$|^(총_|남성_|여성_|10대_|20대_|30대_|40대_|50대_|60대이상_).*(인구수)$")
    num_cols = [c for c in df.columns if patt.search(str(c))]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return df

# ------------------------------------------------------------
# 직장인구 컬럼 표준화 (핵심!)
# ------------------------------------------------------------
def _normalize_worker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    직장인구 원본의 다양한 표기(예: '총_직장_인구_수', '연령대_20_직장_인구_수')를
    main.py가 기대하는 표준 스키마로 통일:
      - 총/남성/여성:  총_직장인구수, 남성_직장인구수, 여성_직장인구수
      - 연령대:        10대_직장인구수, …, 50대_직장인구수, 60대이상_직장인구수
    """
    if not isinstance(df, pd.DataFrame):
        return df

    rename: dict[str, str] = {}
    for c in df.columns:
        original = str(c)
        s = original.strip().replace(" ", "")
        # 표기 흔들림 정리
        s = s.replace("직장_인구", "직장인구")
        s = s.replace("인구_수", "인구수").replace("_수", "수")

        # 총/성별
        if re.fullmatch(r"총_?직장인구수", s):
            rename[c] = "총_직장인구수"; continue
        if re.fullmatch(r"남성_?직장인구수", s):
            rename[c] = "남성_직장인구수"; continue
        if re.fullmatch(r"여성_?직장인구수", s):
            rename[c] = "여성_직장인구수"; continue

        # 연령대 (연령대_20_직장인구수 / 연령대_60_이상_직장인구수 등)
        m = re.match(r"^연령대_?(10|20|30|40|50|60)(?:_?이상)?_?직장인구수$", s)
        if m:
            age = m.group(1)
            if "이상" in s or age == "60":
                rename[c] = "60대이상_직장인구수"
            else:
                rename[c] = f"{age}대_직장인구수"
            continue

        # 이미 표준형이면 rename 불필요
        # (그 외 컬럼은 rename에 넣지 않아 그대로 유지)
        # rename[c] = original  # <- 굳이 매핑하지 않음

    return df.rename(columns=rename)

# ------------------------------------------------------------
# 임대료 정규화
# ------------------------------------------------------------
def _normalize_rent_df(rent_raw: pd.DataFrame) -> pd.DataFrame:
    """
    임대료 엑셀을 표준화:
    - 자치구/행정동 공란 forward-fill
    - '전체' 동 → 자치구 집계로 간주
    - 분기 컬럼명 -> '임대료_YYYYQn'
    - 집계수준: '자치구' 또는 '행정동'
    - 임대료 숫자: 콤마 제거 후 숫자화
    """
    r = rent_raw.copy()

    # 1) 자치구/행정동 컬럼 찾기 (없으면 앞 두 컬럼 사용)
    gu_candidates = ["자치구명", "자치구", "구명"]
    dong_candidates = ["행정동명", "행정동", "동명"]
    gu_col = next((c for c in gu_candidates if c in r.columns), None) or r.columns[0]
    dong_col = next((c for c in dong_candidates if c in r.columns), None) or r.columns[1]

    # 2) 표준화된 컬럼명으로 rename
    r = r.rename(columns={gu_col: "자치구명", dong_col: "행정동명"})

    # 3) 공란 forward-fill (엑셀 병합셀/빈칸 보정)
    r[["자치구명", "행정동명"]] = r[["자치구명", "행정동명"]].ffill()

    # 4) 분기 컬럼 rename → 임대료_YYYYQn
    quarter_map = {}
    for c in list(r.columns):
        s = str(c).strip()
        m = pd.Series([s]).str.extract(r"^\s*(20\d{2})\s*년\s*(\d)\s*분기\s*$")
        if not m.isna().all(axis=None):
            year = m.iloc[0, 0]
            q = m.iloc[0, 1]
            if pd.notna(year) and pd.notna(q):
                quarter_map[c] = f"임대료_{year}Q{int(q)}"
    if quarter_map:
        r = r.rename(columns=quarter_map)

    # 5) 집계수준
    def _level(row):
        v = str(row.get("행정동명", "")).strip()
        return "자치구" if (v == "" or v in ["전체", "합계"]) else "행정동"

    r["집계수준"] = r.apply(_level, axis=1)
    r.loc[r["행정동명"].isin(["전체", "합계"]), "행정동명"] = pd.NA

    # 6) 숫자화
    rent_cols = [c for c in r.columns if str(c).startswith("임대료_")]
    for c in rent_cols:
        r[c] = pd.to_numeric(r[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    return r

# ------------------------------------------------------------
# 로더 본체
# ------------------------------------------------------------
def load_all_data():
    """
    raw_data 폴더에서 데이터 로드 후 표준 컬럼명 통일.
    반환: resident, flow, mall, worker, rent, one_person
    """

    # ------------------ 1) 상주인구(행정동) ------------------
    resident_path = os.path.join(BASE, "상주인구-행정동.csv")
    resident_src_cols = [
        "자치구명", "자치구", "구명",
        "행정동명", "행정동", "행정동_코드", "행정동_코드_명",
        "기준년분기", "기준_년분기_코드", "기준_년분기",
        "총_상주인구_수", "남성_상주인구_수", "여성_상주인구_수",
        "연령대_10_상주인구_수", "연령대_20_상주인구_수", "연령대_30_상주인구_수",
        "연령대_40_상주인구_수", "연령대_50_상주인구_수", "연령대_60_이상_상주인구_수",
        "총_상주인구수", "남성_상주인구수", "여성_상주인구수",
        "10대_상주인구수", "20대_상주인구수", "30대_상주인구수",
        "40대_상주인구수", "50대_상주인구수", "60대이상_상주인구수",
    ]
    resident = _read_csv_select(resident_path, resident_src_cols)
    resident = _rename_safe(resident, {
        "자치구": "자치구명", "구명": "자치구명",
        "행정동_코드_명": "행정동명",
        "행정동": "행정동명",
        "기준_년분기": "기준년분기",
        "기준_년분기_코드": "기준년분기",
        "행정동_코드": "행정동코드",
        "총_상주인구_수": "총_상주인구수",
        "남성_상주인구_수": "남성_상주인구수",
        "여성_상주인구_수": "여성_상주인구수",
        "연령대_10_상주인구_수": "10대_상주인구수",
        "연령대_20_상주인구_수": "20대_상주인구수",
        "연령대_30_상주인구_수": "30대_상주인구수",
        "연령대_40_상주인구_수": "40대_상주인구수",
        "연령대_50_상주인구_수": "50대_상주인구수",
        "연령대_60_이상_상주인구_수": "60대이상_상주인구수",
    })
    resident = _coerce_numeric_like(resident)

    # ------------------ 2) 유동인구(행정동) ------------------
    flow_path = os.path.join(BASE, "길단위인구-행정동.csv")
    flow_src_cols = [
        "자치구명", "자치구", "구명",
        "행정동명", "행정동", "행정동_코드", "행정동_코드_명",
        "기준년분기", "기준_년분기_코드", "기준_년분기",
        "총_유동인구_수", "남성_유동인구_수", "여성_유동인구_수",
        "연령대_10_유동인구_수", "연령대_20_유동인구_수", "연령대_30_유동인구_수",
        "연령대_40_유동인구_수", "연령대_50_유동인구_수", "연령대_60_이상_유동인구_수",
        "총_유동인구수", "남성_유동인구수", "여성_유동인구수",
        "10대_유동인구수", "20대_유동인구수", "30대_유동인구수",
        "40대_유동인구수", "50대_유동인구수", "60대이상_유동인구수",
    ]
    flow = _read_csv_select(flow_path, flow_src_cols)
    flow = _rename_safe(flow, {
        "자치구": "자치구명", "구명": "자치구명",
        "행정동_코드_명": "행정동명",
        "행정동": "행정동명",
        "기준_년분기_코드": "기준년분기",
        "기준_년분기": "기준년분기",
        "행정동_코드": "행정동코드",
        "총_유동인구_수": "총_유동인구수",
        "남성_유동인구_수": "남성_유동인구수",
        "여성_유동인구_수": "여성_유동인구수",
        "연령대_10_유동인구_수": "10대_유동인구수",
        "연령대_20_유동인구_수": "20대_유동인구수",
        "연령대_30_유동인구_수": "30대_유동인구수",
        "연령대_40_유동인구_수": "40대_유동인구수",
        "연령대_50_유동인구_수": "50대_유동인구수",
        "연령대_60_이상_유동인구_수": "60대이상_유동인구수",
    })
    flow = _coerce_numeric_like(flow)

    # ------------------ 3) 상가(상권) 정보(엑셀) ------------------
    mall_path = os.path.join(BASE, "서울시_상가(상권)정보.xlsx")
    mall = _read_excel_safe(mall_path, sheet_name=0)
    mall = _rename_safe(mall, {
        "행정동": "행정동명",
        "법정동": "행정동명",  # 파일에 따라 다를 수 있어 폴백
    })

    # ------------------ 4) 직장인구(행정동) ------------------
    worker_path = os.path.join(BASE, "직장인구-행정동.csv")
    worker_src_cols = [
        "자치구명", "자치구", "구명",
        "행정동명", "행정동", "행정동_코드", "행정동_코드_명",
        "기준년분기", "기준_년분기_코드", "기준_년분기",
        "총_직장인구_수", "남성_직장인구_수", "여성_직장인구_수",
        "연령대_10_직장인구_수", "연령대_20_직장인구_수", "연령대_30_직장인구_수",
        "연령대_40_직장인구_수", "연령대_50_직장인구_수", "연령대_60_이상_직장인구_수",
        "총_직장인구수", "남성_직장인구수", "여성_직장인구수",
        "10대_직장인구수", "20대_직장인구수", "30대_직장인구수",
        "40대_직장인구수", "50대_직장인구수", "60대이상_직장인구수",
    ]
    worker = _read_csv_select(worker_path, None)  # 전체 컬럼 읽기!
    worker = _normalize_worker_columns(worker)    # 표준 컬럼명으로 통일
    worker = _coerce_numeric_like(worker)         # 콤마 제거 + 숫자화

    worker = _rename_safe(worker, {
        "자치구": "자치구명", "구명": "자치구명",
        "행정동_코드_명": "행정동명",
        "행정동": "행정동명",
        "기준_년분기_코드": "기준년분기",
        "기준_년분기": "기준년분기",
        "행정동_코드": "행정동코드",
        # 총/성별/연령 일부 표준형 우선 매핑
        "총_직장인구_수": "총_직장인구수",
        "남성_직장인구_수": "남성_직장인구수",
        "여성_직장인구_수": "여성_직장인구수",
        "연령대_10_직장인구_수": "10대_직장인구수",
        "연령대_20_직장인구_수": "20대_직장인구수",
        "연령대_30_직장인구_수": "30대_직장인구수",
        "연령대_40_직장인구_수": "40대_직장인구수",
        "연령대_50_직장인구_수": "50대_직장인구수",
        "연령대_60_이상_직장인구_수": "60대이상_직장인구수",
    })
    # 🔧 다양한 표기를 표준 스키마로 추가 정규화
    worker = _normalize_worker_columns(worker)
    # 🔢 숫자화(콤마 제거 등)
    worker = _coerce_numeric_like(worker)

    # ------------------ 5) 임대료(엑셀, 자치구/행정동 혼재 가능) ------------------
    rent_path = os.path.join(BASE, "자치구_행정동별_3분기_임대료.xlsx")
    rent_raw = _read_excel_safe(rent_path, sheet_name=0)
    rent = _normalize_rent_df(rent_raw)

    # ------------------ 6) 1인가구(CSV, 자치구 단위) ------------------
    one_p_path = os.path.join(BASE, "자치구별_1인가구(연령별).csv")
    one_person = None
    for enc in ("cp949", "utf-8", "utf-8-sig"):
        try:
            one_person = pd.read_csv(one_p_path, encoding=enc)
            break
        except Exception:
            continue
    if one_person is None:
        one_person = pd.read_csv(one_p_path, encoding="cp949")
    one_person = _rename_safe(one_person, {"자치구": "자치구명", "구명": "자치구명"})

    return resident, flow, mall, worker, rent, one_person
