# report_builder.py
from __future__ import annotations  # 타입 힌트 지연평가(선택이지만 추천)
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import re

def _safe_mean(series) -> Optional[float]:
    try:
        s = pd.to_numeric(series, errors="coerce")
        v = float(s.dropna().mean())
        return v if pd.notna(v) else None
    except Exception:
        return None

def _friendly_age_label(colname: str) -> Optional[str]:
    """원본 1인가구 컬럼을 '20~24세' 등 라벨로 정규화(해당 없으면 None)"""
    colname = str(colname).strip()
    mapping = {
        "계": None, "20세미만": "20세미만", "20~24세": "20~24세", "25~29세": "25~29세",
        "30~34세": "30~34세", "35~39세": "35~39세", "40~44세": "40~44세",
        "45~49세": "45~49세", "50~54세": "50~54세", "55~59세": "55~59세",
        "60~64세": "60~64세", "65~69세": "65~69세", "70~74세": "70~74세",
        "75~79세": "75~79세", "80~84세": "80~84세", "85세이상": "85세이상",
    }
    if colname in mapping:
        return mapping[colname]
    m = re.match(r"^(\d{2})\s*[~∼-]\s*(\d{2})세$", colname)
    if m: return f"{m.group(1)}~{m.group(2)}세"
    m = re.match(r"^(\d{2})세이상$", colname)
    if m: return f"{m.group(1)}세이상"
    return None

def _find_first(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# 내부 유틸
# -----------------------------

def _safe_float(x) -> Optional[float]:
    """
    값이 숫자/숫자문자열/NaN/공백/기호 등 섞여 있어도
    안전하게 float 또는 None을 반환.
    """
    try:
        v = pd.to_numeric(x, errors="coerce")
        # v가 시리즈/스칼라 모두 올 수 있으니 스칼라로 보정
        if isinstance(v, pd.Series):
            if v.empty:
                return None
            v = v.iloc[0]
        f = float(v)
        return None if pd.isna(f) else f
    except Exception:
        return None

def _safe_sum(series) -> Optional[float]:
    try:
        s = pd.to_numeric(series, errors="coerce")
        v = float(s.fillna(0).sum())
        return v
    except Exception:
        return None

def _aggregate_stores(mall_df: Optional[pd.DataFrame], region_name: str) -> Dict[str, Any]:
    """
    상가 DF에서 해당 행정동의 업종별 점포 수 상위 집계 (컬럼 자동 인식)
    반환 예:
      {
        "총_점포수": 123,
        "점포수_상권업종소분류명_한식음식점": 30,
        "점포수_상권업종소분류명_치킨전문점": 12,
        ...
      }
    """
    out: Dict[str, Any] = {}
    if not isinstance(mall_df, pd.DataFrame) or mall_df.empty:
        return out

    by_col = None
    for c in ["상권업종소분류명", "상권업종중분류명", "상권업종대분류명", "업종명", "업종코드"]:
        if c in mall_df.columns:
            by_col = c
            break
    if by_col is None:
        return out

    df = mall_df
    if "행정동명" in df.columns:
        df = df[df["행정동명"] == region_name]
    if df.empty:
        return out

    g = df.groupby(by_col).size().sort_values(ascending=False)
    total = int(g.sum()) if len(g) else 0
    out["총_점포수"] = total

    # 상위 10개만 노출
    for k, v in g.head(10).items():
        out[f"점포수_{by_col}_{k}"] = int(v)
    return out


# -----------------------------
# 핵심: 지역 리포트용 행(row) 생성
# (표준 컬럼명: …유동인구수 / …상주인구수 / …직장인구수)
# latest: {"flow": DataFrame(index=행정동명, 최신행), ...}
# dfs:    원본 DF dict
# store_cache: precompute_store_counts 결과 (있으면 활용)
# -----------------------------
def build_row_for_region(region_name: str,
                         dfs: Dict[str, Optional[pd.DataFrame]],
                         latest: Dict[str, Optional[pd.DataFrame]],
                         store_cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    flow_df     = dfs.get("flow")
    resident_df = dfs.get("resident")
    worker_df   = dfs.get("worker")
    one_df      = dfs.get("one_person")  # 자치구 단위일 수 있음
    rent_df     = dfs.get("rent")
    mall_df     = dfs.get("mall")

    row: Dict[str, Any] = {"행정동명": region_name}

    # 최신행 인덱스 (main.py에서 precompute_latest_by_region로 생성한 것)
    flow_latest_idx     = latest.get("flow")
    resident_latest_idx = latest.get("resident")
    worker_latest_idx   = latest.get("worker")

    # ----- 유동인구 (최신행) -----
    if isinstance(flow_latest_idx, pd.DataFrame) and region_name in flow_latest_idx.index:
        flow_latest = flow_latest_idx.loc[region_name]
        for k in [
            "총_유동인구수","남성_유동인구수","여성_유동인구수",
            "10대_유동인구수","20대_유동인구수","30대_유동인구수",
            "40대_유동인구수","50대_유동인구수","60대이상_유동인구수",
        ]:
            if k in flow_latest:
                v = flow_latest.get(k)
                row[k] = float(v) if pd.notna(v) else None

    # ----- 상주인구 (최신행) -----
    if isinstance(resident_latest_idx, pd.DataFrame) and region_name in resident_latest_idx.index:
        res_latest = resident_latest_idx.loc[region_name]
        for k in [
            "총_상주인구수","남성_상주인구수","여성_상주인구수",
            "10대_상주인구수","20대_상주인구수","30대_상주인구수",
            "40대_상주인구수","50대_상주인구수","60대이상_상주인구수",
        ]:
            if k in res_latest:
                v = res_latest.get(k)
                row[k] = float(v) if pd.notna(v) else None

    # ----- 직장인구 (최신행) -----
    if isinstance(worker_latest_idx, pd.DataFrame) and region_name in worker_latest_idx.index:
        work_latest = worker_latest_idx.loc[region_name]
        for k in [
            "총_직장인구수","남성_직장인구수","여성_직장인구수",
            "10대_직장인구수","20대_직장인구수","30대_직장인구수",
            "40대_직장인구수","50대_직장인구수","60대이상_직장인구수",
        ]:
            if k in work_latest:
                v = work_latest.get(k)
                row[k] = float(v) if pd.notna(v) else None

    # ----- 1인가구 (자치구 단위: 합계 + 서울평균 총합) -----
    gu_name = None
    for key in ["resident", "flow", "worker", "mall", "rent"]:
        df = dfs.get(key)
        if isinstance(df, pd.DataFrame) and "행정동명" in df.columns and region_name in df["행정동명"].values:
            gc = _find_first(df, ["자치구명", "자치구", "구명"])
            if gc:
                val = df.loc[df["행정동명"] == region_name, gc].iloc[0]
                if isinstance(val, str) and val.strip():
                    gu_name = val.strip().replace(" ", "")
                    break

    if isinstance(one_df, pd.DataFrame):
        # 자치구명 컬럼 식별
        one_gc = _find_first(one_df, ["자치구명","자치구","구명"])
        one_df_use = one_df.copy()
        # 연령대 컬럼 정규화: '1인가구_<라벨>' 형식으로 row에 투입
        age_cols = [c for c in one_df_use.columns if _friendly_age_label(str(c))]
        # 자치구 합계(해당 구)
        if one_gc and gu_name:
            sub = one_df_use[one_df_use[one_gc] == gu_name]
        else:
            sub = one_df_use

        if not sub.empty and age_cols:
            # 해당 구의 각 연령 카테고리 합 → row["1인가구_<라벨>"]
            for c in age_cols:
                lab = _friendly_age_label(str(c))
                if not lab: continue
                row[f"1인가구_{lab}"] = _safe_sum(sub[c])

            # 해당 구의 1인가구 총합
            gu_total = _safe_sum(sub[age_cols].sum(axis=1))
            if gu_total is not None:
                row["총_1인가구"] = gu_total

        # 서울평균 총 1인가구(자치구별 총합의 평균)
        if age_cols:
            # 자치구별 합 → 평균
            if one_gc:
                tmp = (pd.to_numeric(one_df_use[age_cols], errors="coerce")
                         .fillna(0).sum(axis=1))
                seoul_avg_one = _safe_mean(tmp)
                row["서울평균_총_1인가구"] = seoul_avg_one
            else:
                # 자치구 구분이 없으면 전체 평균으로라도 채움
                tmp = pd.to_numeric(one_df_use[age_cols], errors="coerce").fillna(0).sum(axis=1)
                row["서울평균_총_1인가구"] = _safe_mean(tmp)

    # 추가로, 구명은 report_generator에 쓰일 수 있으므로 남겨둠
    if gu_name:
        row["one_person_자치구명"] = gu_name


    if isinstance(one_df, pd.DataFrame):
        # one_df가 자치구 기준이라면 자치구로 필터
        if gu_name:
            one_sub = one_df[one_df[[c for c in ["자치구명","자치구","구명"] if c in one_df.columns][0]] == gu_name]
        else:
            one_sub = one_df
        if not one_sub.empty:
            one_cols = [c for c in one_sub.columns if str(c).startswith("1인가구")]
            for c in one_cols:
                row[c] = _safe_sum(one_sub[c])
            if gu_name:
                row["one_person_출처_단위"] = "자치구"
                row["one_person_자치구명"] = gu_name

    # ----- 임대료 (행정동 우선, 없으면 자치구 평균) -----
    if isinstance(rent_df, pd.DataFrame) and not rent_df.empty:
        r = rent_df.copy()

        # 동/구 기준 부분집합
        rsub_dong = r[r["행정동명"] == region_name] if "행정동명" in r.columns else pd.DataFrame()
        rsub_gu   = r[(r["자치구명"] == gu_name)] if (gu_name and "자치구명" in r.columns) else pd.DataFrame()

        rent_cols = [c for c in r.columns if str(c).startswith("임대료_")]
        # 동 최신행 값 집어넣기(있으면)
        if not rsub_dong.empty and rent_cols:
            # 기준년분기 정렬 가능 시 최신, 아니면 마지막
            if "기준년분기" in rsub_dong.columns:
                latest_row = rsub_dong.sort_values("기준년분기").tail(1).iloc[0]
            else:
                latest_row = rsub_dong.iloc[-1]
            for c in rent_cols:
                row[c] = _safe_float(latest_row.get(c))

        # 서울평균_임대료: r 전체 rent_cols의 평균
        if rent_cols:
            row["서울평균_임대료"] = _safe_mean(r[rent_cols].stack())

        # 자치구평균_임대료: 구 단위 행이 있으면 그 평균, 없으면 해당 구의 동들 평균
        gu_avg = None
        if not rsub_gu.empty and rent_cols:
            gu_avg = _safe_mean(rsub_gu[rent_cols].stack())
        elif rent_cols and gu_name and "자치구명" in r.columns:
            # 자치구명 매칭되는 모든 행 평균
            gu_rows = r[r["자치구명"] == gu_name]
            if not gu_rows.empty:
                gu_avg = _safe_mean(gu_rows[rent_cols].stack())
        row["자치구평균_임대료"] = gu_avg



    # ----- 업종 집계 (사전계산 store_cache 사용) -----
    if store_cache:
        by_col = store_cache.get("by_col")
        region_top = (store_cache.get("region_top") or {})
        global_top = store_cache.get("global_top")
        top_map = region_top.get(region_name) or global_top
        if top_map:
            row["총_점포수"] = int(sum(top_map.values()))
            if by_col:
                for k, v in top_map.items():
                    row[f"점포수_{by_col}_{k}"] = int(v)
    else:
        # 사전계산이 없다면 mall_df에서 즉석 집계
        row.update(_aggregate_stores(mall_df, region_name))

    return row


# -----------------------------
# 출력 & 요약 도우미 (터미널 표시용)
# -----------------------------
def pretty_print_report(row: Dict[str, Any]) -> None:
    print("\n====================")
    print(f"[지역 리포트] {row.get('행정동명', '')}")
    print("====================")

    def _p(title, keys):
        vals = [(k, row.get(k)) for k in keys if row.get(k) is not None]
        if not vals:
            return
        print(f"\n■ {title}")
        for k, v in vals:
            if isinstance(v, (int, float)):
                try:
                    print(f"  - {k}: {int(v):,}")
                except Exception:
                    print(f"  - {k}: {v}")
            else:
                print(f"  - {k}: {v}")

    _p("유동인구(최신)", [
        "총_유동인구수","남성_유동인구수","여성_유동인구수",
        "10대_유동인구수","20대_유동인구수","30대_유동인구수",
        "40대_유동인구수","50대_유동인구수","60대이상_유동인구수",
    ])

    _p("상주인구(최신)", [
        "총_상주인구수","남성_상주인구수","여성_상주인구수",
        "10대_상주인구수","20대_상주인구수","30대_상주인구수",
        "40대_상주인구수","50대_상주인구수","60대이상_상주인구수",
    ])

    _p("직장인구(최신)", [
        "총_직장인구수","남성_직장인구수","여성_직장인구수",
        "10대_직장인구수","20대_직장인구수","30대_직장인구수",
        "40대_직장인구수","50대_직장인구수","60대이상_직장인구수",
    ])

    one_cols = [k for k in row if str(k).startswith("1인가구") and row.get(k) is not None]
    _p("1인가구(자치구 기반)", one_cols)

    rent_cols = [k for k in row if str(k).startswith("임대료_") and row.get(k) is not None]
    _p("임대료", rent_cols + (["서울평균_임대료"] if row.get("서울평균_임대료") is not None else []))

    top_keys = [(k, row[k]) for k in row if str(k).startswith("점포수_") and isinstance(row[k], (int, float))]
    if top_keys:
        print("\n■ 업종별 점포 수 (상위)")
        for k, v in sorted(top_keys, key=lambda x: x[1], reverse=True):
            print(f"  - {k}: {int(v)}")


def summarize_with_gpt_if_needed(row: Dict[str, Any], use_gpt: bool) -> Optional[str]:
    """
    외부 GPT 호출 없이 규칙 기반 요약 문장을 만들어 반환.
    (use_gpt 플래그는 인터페이스 호환을 위해 둠)
    """
    if not use_gpt:
        return None
    dong = row.get("행정동명", "해당 지역")
    parts = [f"{dong} 상권 스냅샷입니다."]

    tf = row.get("총_유동인구수")
    if isinstance(tf, (int, float)):
        parts.append(f"유동인구는 최신 분기 기준 약 {int(tf):,}명으로 추정됩니다.")
        m = row.get("남성_유동인구수"); f = row.get("여성_유동인구수")
        if isinstance(m, (int, float)) and isinstance(f, (int, float)) and (m + f) > 0:
            parts.append(f"성비는 남 {m/(m+f)*100:.1f}%, 여 {f/(m+f)*100:.1f}% 수준입니다.")
        ages = [(k, row.get(k)) for k in [
            "10대_유동인구수","20대_유동인구수","30대_유동인구수",
            "40대_유동인구수","50대_유동인구수","60대이상_유동인구수"
        ] if isinstance(row.get(k), (int, float))]
        if ages:
            top_age = max(ages, key=lambda x: x[1])[0].replace("_유동인구수","")
            parts.append(f"연령대는 {top_age} 비중이 상대적으로 큽니다.")

    res_vals = [(k, row.get(k)) for k in [
        "10대_상주인구수","20대_상주인구수","30대_상주인구수",
        "40대_상주인구수","50대_상주인구수","60대이상_상주인구수"
    ] if isinstance(row.get(k), (int, float))]
    if res_vals:
        parts.append(f"상주인구는 {max(res_vals, key=lambda x: x[1])[0].replace('_상주인구수','')}가 두드러집니다.")

    workers = row.get("총_직장인구수")
    if isinstance(workers, (int, float)):
        parts.append(f"직장인구는 약 {int(workers):,}명으로 점심·퇴근 시간 수요가 기대됩니다.")

    gu = row.get("one_person_자치구명")
    if gu:
        parts.append(f"{dong}이(가) 속한 {gu}의 1인가구 지표를 참고했습니다.")

    if len(parts) == 1:
        parts.append("데이터 매칭이 제한적입니다. 데이터 소스 컬럼명/행정동명이 일치하는지 확인하세요.")
    return " ".join(parts)
