# scorer.py
import pandas as pd
from typing import Dict, List, Optional

def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx):
        # 전부 NaN이면 0.5 고정
        return pd.Series([0.5] * len(s), index=s.index)
    if mn == mx:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """컬럼이 없으면 0으로 채운 Series 반환."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([0.0] * len(df), index=df.index)

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_region_score(
    dfs: Dict[str, pd.DataFrame],
    preferred_ages: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    top_n: int = 2
) -> pd.DataFrame:
    """
    지역별 점수 계산 후 상위 N개 반환.
    preferred_ages: 기본 ["20대_유동인구수", "30대_유동인구수"]
    weights: {"flow":0.35, "worker":0.25, "age":0.25, "rent":0.15}
    """
    preferred_ages = preferred_ages or ["20대_유동인구수", "30대_유동인구수"]
    weights = weights or {"flow": 0.35, "worker": 0.25, "age": 0.25, "rent": 0.15}

    flow   = dfs.get("flow",   pd.DataFrame()).copy()
    resident = dfs.get("resident", pd.DataFrame()).copy()
    worker = dfs.get("worker", pd.DataFrame()).copy()
    rent   = dfs.get("rent",   pd.DataFrame()).copy()

    # -------- 1) 표준 컬럼으로 그룹 평균 (기간 전체 평균 사용) --------
    # 유동
    flow_cols = [
        "행정동명", "총_유동인구수",
        "10대_유동인구수","20대_유동인구수","30대_유동인구수",
        "40대_유동인구수","50대_유동인구수","60대이상_유동인구수",
    ]
    flow_exist = [c for c in flow_cols if c in flow.columns]
    flow_grp = pd.DataFrame(columns=flow_cols)
    if flow_exist:
        flow_grp = (flow[flow_exist]
                    .groupby("행정동명", as_index=False)
                    .mean(numeric_only=True))

    # 직장
    worker_cols = [
        "행정동명", "총_직장인구수",
        "10대_직장인구수","20대_직장인구수","30대_직장인구수",
        "40대_직장인구수","50대_직장인구수","60대이상_직장인구수",
    ]
    worker_exist = [c for c in worker_cols if c in worker.columns]
    worker_grp = pd.DataFrame(columns=worker_cols)
    if worker_exist:
        worker_grp = (worker[worker_exist]
                      .groupby("행정동명", as_index=False)
                      .mean(numeric_only=True))

    # 상주
    resident_cols = [
        "행정동명", "총_상주인구수",
        "10대_상주인구수","20대_상주인구수","30대_상주인구수",
        "40대_상주인구수","50대_상주인구수","60대이상_상주인구수",
    ]
    resident_exist = [c for c in resident_cols if c in resident.columns]
    resident_grp = pd.DataFrame(columns=resident_cols)
    if resident_exist:
        resident_grp = (resident[resident_exist]
                        .groupby("행정동명", as_index=False)
                        .mean(numeric_only=True))

    # 임대료: 임대료_* 컬럼들을 평균 (행정동 단위)
    rent_grp = pd.DataFrame(columns=["행정동명"])
    if not rent.empty:
        rent_cols_all = [c for c in rent.columns if str(c).startswith("임대료_")]
        if rent_cols_all:
            rnum = rent[["행정동명"] + rent_cols_all].copy()
            for c in rent_cols_all:
                rnum[c] = pd.to_numeric(rnum[c], errors="coerce")
            rent_grp = (rnum.groupby("행정동명", as_index=False)[rent_cols_all]
                             .mean(numeric_only=True))

    # -------- 2) 병합 --------
    merged = flow_grp.merge(worker_grp, on="행정동명", how="outer", suffixes=("", "_wk"))
    merged = merged.merge(resident_grp, on="행정동명", how="outer", suffixes=("", "_res"))
    merged = merged.merge(rent_grp, on="행정동명", how="left", suffixes=("", "_rent"))

    if merged.empty:
        # 아무 데이터도 없으면 빈 DF 반환
        return merged

    # -------- 3) 점수 계산 --------
    # 3-1) 유동/직장 점수
    merged["flow_score"]   = _minmax(_safe_series(merged, "총_유동인구수"))
    merged["worker_score"] = _minmax(_safe_series(merged, "총_직장인구수"))

    # 3-2) 선호 연령대 점수 (유동 우선, 없으면 상주 대체)
    age_series_list = []
    for ac in preferred_ages:
        if ac in merged.columns:
            age_series_list.append(pd.to_numeric(merged[ac], errors="coerce"))
        else:
            alt = ac.replace("유동인구수", "상주인구수")
            if alt in merged.columns:
                age_series_list.append(pd.to_numeric(merged[alt], errors="coerce"))

    if age_series_list:
        age_base = sum(age_series_list) / len(age_series_list)
        merged["age_score"] = _minmax(age_base)
    else:
        merged["age_score"] = 0.5

    # 3-3) 임대료 점수 (낮을수록 가점)
    rent_cols_all = [c for c in merged.columns if str(c).startswith("임대료_")]
    if rent_cols_all:
        rent_avg = pd.to_numeric(merged[rent_cols_all], errors="coerce").mean(axis=1)
        rent_norm = _minmax(rent_avg)
        merged["rent_score"] = 1 - rent_norm
    else:
        merged["rent_score"] = 0.5

    # 결측 채움
    for c in ["flow_score", "worker_score", "age_score", "rent_score"]:
        if c not in merged.columns:
            merged[c] = 0.5
        merged[c] = merged[c].fillna(0.5)

    # 3-4) 가중합
    merged["score"] = (
        merged["flow_score"]   * float(weights.get("flow", 0.35)) +
        merged["worker_score"] * float(weights.get("worker", 0.25)) +
        merged["age_score"]    * float(weights.get("age", 0.25)) +
        merged["rent_score"]   * float(weights.get("rent", 0.15))
    )

    # 정렬 및 상위 N개
    merged = merged.sort_values("score", ascending=False).reset_index(drop=True)
    return merged.head(top_n)
