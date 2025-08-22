# report_generator.py
import json
import math
from typing import Dict, List, Optional, Tuple

def _safe_float(x) -> Optional[float]:
    try:
        if x is None: return None
        f = float(x)
        if math.isnan(f): return None
        return f
    except Exception:
        return None

def _pct(a: Optional[float], b: Optional[float]) -> Optional[int]:
    """a/b*100 을 정수(반올림)로. 못 구하면 None."""
    if a is None or b is None or b == 0:
        return None
    try:
        return int(round((a / b) * 100))
    except Exception:
        return None

def _pct_text(pct: Optional[int]) -> str:
    return "해당 데이터가 부족합니다." if pct is None else f"{pct}%"

def _more_or_less(pct: Optional[int]) -> str:
    if pct is None:
        return "데이터 부족"
    if pct >= 105:
        return "많은편"
    if pct <= 95:
        return "적은편"
    return "유사한편"

def _top2_share_from_row(row: Dict, cols: List[str], labels: List[str]) -> Optional[List[Tuple[str, int]]]:
    """
    row에서 cols의 값을 합하여 비중(%)을 구하고 상위 2개 반환 [(라벨, 퍼센트), ...].
    값이 하나도 없으면 None.
    """
    vals: List[Tuple[str, float]] = []
    for c, lab in zip(cols, labels):
        v = _safe_float(row.get(c))
        if v is not None:
            vals.append((lab, v))
    if not vals:
        return None
    total = sum(v for _, v in vals)
    if total <= 0:
        return None
    shares = [(lab, int(round(v / total * 100))) for lab, v in vals]
    shares.sort(key=lambda x: x[1], reverse=True)
    return shares[:2]

def _store_top5_line(row: Dict, region_name: str, gu_name: Optional[str]) -> str:
    """
    '점포수_' 접두의 항목들을 찾아 상위 5개를 '1위: A(n), 2위: B(n) ...' 형태로 문장화.
    없으면 '해당 데이터가 부족합니다.' 반환.
    """
    pairs: List[Tuple[str, int]] = []
    for k, v in row.items():
        if str(k).startswith("점포수_"):
            label = str(k).split("점포수_", 1)[1]
            cnt = _safe_float(v)
            if cnt is not None:
                pairs.append((label, int(round(cnt))))
    if not pairs:
        return "해당 데이터가 부족합니다."

    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:5]
    parts = [f"{i+1}위: {lab}({cnt})" for i, (lab, cnt) in enumerate(top)]
    head = f"{gu_name} {region_name}" if gu_name else region_name
    return f"{head}의 점포 수는 " + ", ".join(parts) + "의 분포를 보이고 있다."

def _rent_sentence(row: Dict, region_name: str, gu_name: Optional[str]) -> str:
    """
    임대료_ 접두 컬럼들을 평균내고, 서울/자치구 평균과 비교 문장 생성.
    각 값이 없으면 '해당 데이터가 부족합니다.' 반환.
    """
    # 동 평균
    rent_vals = []
    for k, v in row.items():
        if str(k).startswith("임대료_"):
            fv = _safe_float(v)
            if fv is not None:
                rent_vals.append(fv)
    local_avg = _safe_float(sum(rent_vals) / len(rent_vals)) if rent_vals else None

    # 서울 평균, 자치구 평균(있을 때만)
    seoul_avg = _safe_float(row.get("서울평균_임대료"))
    gu_avg    = _safe_float(row.get("자치구평균_임대료"))

    if local_avg is None and (gu_avg is None):
        return "해당 데이터가 부족합니다."

    # 우선 동 기준으로 비교, 없으면 자치구 기준으로 비교
    if local_avg is not None:
        pct_vs_seoul = _pct(local_avg, seoul_avg)
        pct_vs_gu    = _pct(local_avg, gu_avg) if gu_avg is not None else None
        if pct_vs_seoul is None and pct_vs_gu is None:
            return "해당 데이터가 부족합니다."
        if pct_vs_gu is not None:
            return f"{region_name}의 평균 임대료는 서울시 평균에 비해 {_pct_text(pct_vs_seoul)}이고, 소속 자치구({gu_name or '해당 구'}) 평균에 비해 {_pct_text(pct_vs_gu)}이다."
        else:
            return f"{region_name}의 평균 임대료는 서울시 평균 대비 {_pct_text(pct_vs_seoul)} 수준이다."
    else:
        # 동 데이터가 부족 → 자치구 평균으로 서울과 비교
        pct_vs_seoul = _pct(gu_avg, seoul_avg)
        if pct_vs_seoul is None:
            return "해당 데이터가 부족합니다."
        return f"행정동 단위 데이터가 부족하여 자치구({gu_name or '해당 구'}) 평균으로 비교하면, 서울시 평균 대비 {_pct_text(pct_vs_seoul)} 수준이다."

def generate_region_report(
    region_name: str,
    row_for_region: Dict,
    *,
    gu_name: Optional[str] = None,
    use_gpt: bool = False,
    gpt_fn=None
) -> Dict:
    """
    row_for_region: 한 지역(행정동) 기준으로 이미 병합/집계된 dict.
    - 비교를 위해 다음 키들이 있으면 활용합니다(없으면 '해당 데이터가 부족합니다.'):
      '서울평균_총_상주인구수', '서울평균_총_유동인구수', '서울평균_총_직장인구수',
      '서울평균_임대료', '자치구평균_임대료'
    - 1인가구는 연령대별 합이 있으면 상위 1·2위 비중 문장만 생성(서울평균 비교값이 있으면 함께 사용).
    """
    # ---------- 1) 1인가구 ----------
    one_cols = []
    for k in row_for_region.keys():
        if str(k).startswith("1인가구_"):
            one_cols.append(k)
    # 1인가구 총합, 서울평균(있으면)
    one_vals = [_safe_float(row_for_region.get(c)) for c in one_cols]
    one_vals = [v for v in one_vals if v is not None]
    one_total = sum(one_vals) if one_vals else None
    seoul_one_total = _safe_float(row_for_region.get("서울평균_총_1인가구"))

    # 상위 2위
    one_pairs = []
    for c in one_cols:
        v = _safe_float(row_for_region.get(c))
        if v is not None:
            label = str(c).replace("1인가구_", "").replace("_", "~")
            one_pairs.append((label, v))
    one_line: str
    if one_pairs:
        total = sum(v for _, v in one_pairs)
        one_pairs.sort(key=lambda x: x[1], reverse=True)
        top1_lab, top1_val = one_pairs[0]
        top2 = one_pairs[1] if len(one_pairs) > 1 else None
        top1_share = int(round((top1_val / total) * 100)) if total > 0 else None
        if top2:
            top2_lab, top2_val = top2
            top2_share = int(round((top2_val / total) * 100)) if total > 0 else None
            if seoul_one_total is not None and one_total is not None:
                pct_vs_seoul = _pct(one_total, seoul_one_total)
                one_line = (
                    f"\"{region_name}\"동이 속한 {gu_name or '해당 구'}의 1인가구 규모는 서울시 평균 {_pct_text(pct_vs_seoul)}이고, "
                    f"특히 {top1_lab} 1인가구가 {_pct_text(top1_share)}로 가장 많으며, "
                    f"{top2_lab} 1인가구가 {_pct_text(top2_share)}로 그 뒤를 이운다."
                )
            else:
                one_line = (
                    f"{region_name}이 속한 {gu_name or '해당 구'}는 {top1_lab} 1인가구가 {_pct_text(top1_share)}로 가장 많고, "
                    f"{top2_lab} 1인가구가 {_pct_text(top2_share)}로 그 뒤를 이운다."
                )
        else:
            # 상위 1개만
            if seoul_one_total is not None and one_total is not None and total > 0:
                pct_vs_seoul = _pct(one_total, seoul_one_total)
                one_line = (
                    f"\"{region_name}\"동이 속한 {gu_name or '해당 구'}의 1인가구 규모는 서울시 평균 {_pct_text(pct_vs_seoul)}이며, "
                    f"{top1_lab} 비중이 가장 높다."
                )
            else:
                one_line = f"{region_name}이 속한 {gu_name or '해당 구'}는 {top1_lab} 1인가구 비중이 가장 높다."
    else:
        one_line = "해당 데이터가 부족합니다."

    # ---------- 2) 상주인구 ----------
    res_cols = ["10대_상주인구수","20대_상주인구수","30대_상주인구수",
                "40대_상주인구수","50대_상주인구수","60대이상_상주인구수"]
    res_labels = ["10대","20대","30대","40대","50대","60대 이상"]
    res_total = _safe_float(row_for_region.get("총_상주인구수"))
    res_seoul_avg = _safe_float(row_for_region.get("서울평균_총_상주인구수"))
    res_pct = _pct(res_total, res_seoul_avg)
    res_top2 = _top2_share_from_row(row_for_region, res_cols, res_labels)
    if res_total is None or res_seoul_avg is None or not res_top2:
        resident_line = "해당 데이터가 부족합니다."
    else:
        r1 = f"1위: {res_top2[0][0]} 상주인구수가 {res_top2[0][1]}%"
        r2 = f", 2위: {res_top2[1][0]} 상주인구수가 {res_top2[1][1]}%" if len(res_top2) > 1 else ""
        resident_line = (
            f"\"{region_name}\"동의 상주인구는 서울시 타 행정동 평균에 비해 {res_pct}% 으로 {_more_or_less(res_pct)}이고, "
            f"연령대별로는 {r1}{r2}이다."
        )

    # ---------- 3) 유동인구 ----------
    flow_cols = ["10대_유동인구수","20대_유동인구수","30대_유동인구수",
                 "40대_유동인구수","50대_유동인구수","60대이상_유동인구수"]
    flow_labels = ["10대","20대","30대","40대","50대","60대 이상"]
    flow_total = _safe_float(row_for_region.get("총_유동인구수"))
    flow_seoul_avg = _safe_float(row_for_region.get("서울평균_총_유동인구수"))
    flow_pct = _pct(flow_total, flow_seoul_avg)
    flow_top2 = _top2_share_from_row(row_for_region, flow_cols, flow_labels)
    if flow_total is None or flow_seoul_avg is None or not flow_top2:
        flow_line = "해당 데이터가 부족합니다."
    else:
        f1 = f"1위: {flow_top2[0][0]} 유동인구수가 {flow_top2[0][1]}%"
        f2 = f", 2위: {flow_top2[1][0]} 유동인구수가 {flow_top2[1][1]}%" if len(flow_top2) > 1 else ""
        flow_line = (
            f"\"{region_name}\"동의 유동인구는 서울시 타 행정동 평균에 비해 {flow_pct}% 으로 {_more_or_less(flow_pct)}이고, "
            f"연령대별로는 {f1}{f2}이다."
        )

    # ---------- 4) 직장인구 ----------
    work_cols = ["10대_직장인구수","20대_직장인구수","30대_직장인구수",
                 "40대_직장인구수","50대_직장인구수","60대이상_직장인구수"]
    work_labels = ["10대","20대","30대","40대","50대","60대 이상"]
    work_total = _safe_float(row_for_region.get("총_직장인구수"))
    work_seoul_avg = _safe_float(row_for_region.get("서울평균_총_직장인구수"))
    work_pct = _pct(work_total, work_seoul_avg)
    work_top2 = _top2_share_from_row(row_for_region, work_cols, work_labels)
    if work_total is None or work_seoul_avg is None or not work_top2:
        worker_line = "해당 데이터가 부족합니다."
    else:
        w1 = f"1위: {work_top2[0][0]} 직장인구수가 {work_top2[0][1]}%"
        w2 = f", 2위: {work_top2[1][0]} 직장인구수가 {work_top2[1][1]}%" if len(work_top2) > 1 else ""
        tendency = "적은편(100% 이하)" if work_pct is not None and work_pct <= 100 else _more_or_less(work_pct)
        worker_line = (
            f"\"{region_name}\"동의 직장인구는 서울시 타 행정동 평균에 비해 {work_pct}% 으로 {tendency}이고, "
            f"연령대별로는 {w1}{w2}이다."
        )

    # ---------- 5) 점포수 ----------
    store_line = _store_top5_line(row_for_region, region_name, gu_name)

    # ---------- 6) 임대료 ----------
    rent_line = _rent_sentence(row_for_region, region_name, gu_name)

    # ---------- 7) 입지 특성 ----------
    place_line = row_for_region.get("입지_힌트") or "교통·학교·상업시설 등 입지적 요인이 복합적으로 작용하는 지역이다."

    # ---------- 8) 최종 JSON ----------
    report = {
        "region": region_name,
        "report": {
            "인구": {
                "1인가구": one_line,
                "상주인구": resident_line,
                "유동인구": flow_line,
                "직장인구": worker_line
            },
            "업종": {
                "점포수": store_line,
                "임대료_특징": rent_line
            },
            "입지_특성": place_line
        }
    }

    # (선택) GPT 요약 붙이기
    if use_gpt and gpt_fn:
        try:
            summary = gpt_fn(
                f"다음 JSON을 자연스러운 한국어로 4~6문장 요약해줘. 숫자/사실 왜곡 금지.\n{json.dumps(report, ensure_ascii=False)}"
            )
            if isinstance(summary, str) and summary.strip():
                report["report"]["요약"] = summary.strip()
        except Exception:
            pass

    return report
