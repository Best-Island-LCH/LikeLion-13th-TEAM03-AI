# main.py
import os
import sys
import re
import json
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

# -----------------------------
# (선택) OpenAI: 설정되어 있으면 사용, 아니면 자동 Fallback
# -----------------------------
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    _OPENAI_AVAILABLE = False

_client = None
if _OPENAI_AVAILABLE:
    _api_key = os.getenv("OPENAI_API_KEY")
    if _api_key:
        try:
            _client = OpenAI(api_key=_api_key)
        except Exception:
            _client = None

# === LLM 재랭킹 토글 (환경변수 USE_LLM_RERANK=1 이면 켜짐) ===
USE_LLM_RERANK = os.getenv("USE_LLM_RERANK", "0").strip() == "1"
RERANK_MODEL = "gpt-4o-mini"

# -----------------------------
# AI로 biz_feature 생성 (실패 시 규칙기반 Fallback)
# -----------------------------
def _gen_biz_feature(type_small: str) -> str:
    """
    업종별 특성을 AI로 생성 (실패 시 기본 문구 반환, 실행 중단 없음)
    """
    # 1) OpenAI 사용 가능 & 키 있음 → 호출
    if _client is not None:
        try:
            resp = _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "너는 대한민국 상권 분석 전문가다. 숫자/사실 왜곡 없이 간결히."},
                    {"role": "user", "content": f"'{type_small}' 업종의 상권 적합 입지 특성을 2~3문장으로 요약해줘. "
                                                f"유동/직장/상주/임대료 관점과 '어떤 연령/시간대'에 강점인지도 언급."}
                ],
                max_tokens=200
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"[경고] biz_feature OpenAI 호출 실패: {e}")

    # 2) Fallback (룰 기반 요약)
    t = type_small
    # 간단 룰: 음식/서비스/전문업으로 크게 나누어 베이스 문구
    if any(k in t for k in ["카페", "커피", "디저트", "분식", "치킨", "피자", "한식", "일식", "중식", "양식", "면", "고기", "샐러드", "포장마차"]):
        base = "식음료 업종은 점심·퇴근 시간대 및 주말 유동에 민감하며, 20~40대 비중이 높은 지역에서 매출 탄력이 큽니다."
    elif any(k in t for k in ["미용", "헤어", "네일", "피부", "마사지", "세탁", "편의점", "편의"]):
        base = "생활밀착 서비스 업종은 상주인구 밀도·재방문 수요가 핵심이며, 거주 30~50대 비중과 주거밀집·통학동선 가시성이 중요합니다."
    elif any(k in t for k in ["학원", "교육", "교습", "입시"]):
        base = "교육·학원 업종은 10~20대 유동, 학군·통학동선·주거밀집도가 중요하며, 주중 오후·저녁 시간대 수요가 큽니다."
    elif any(k in t for k in ["병원", "의원", "의료", "약국"]):
        base = "의료·헬스케어는 상주인구·근무인구의 안정적 수요가 중요하며, 가시성·접근성·주차 편의가 영향을 줍니다."
    else:
        base = "전반적으로 유동·직장·상주·임대료를 종합해 입지를 판단해야 하며, 주고객 연령과 시간대 일치가 성과에 직결됩니다."
    return f"{t}은(는) {base}"

def _llm_rerank_candidates(type_small: str, prelim: list) -> Optional[list]:
    """
    prelim: [(region, base_score, reason_dict), ...]
    반환: [(region, base_score, reason_dict, llm_explanation:str|None, confidence:float|None), ...]
    """
    if not USE_LLM_RERANK or _client is None:
        return None
    try:
        # LLM 입력용 간단 JSON
        payload = [
            {"region": r, "base_score": round(float(s), 4), "reasons": rsn}
            for (r, s, rsn) in prelim
        ]
        sys_prompt = (
            "너는 상권 추천 재랭킹 모델이다. 업종과 후보들을 보고 더 설득력 있는 순서로 재정렬한다. "
            "데이터 일관성(서술 간 모순 여부)과 업종-특성 적합성을 함께 고려해라. "
            "반드시 JSON만 출력한다."
        )
        user_prompt = (
            f"업종: {type_small}\n"
            f"후보목록(JSON): {json.dumps(payload, ensure_ascii=False)}\n\n"
            "출력 형식(JSON 예): {\n"
            '  "ranked": [\n'
            '    {"region":"역삼1동","explanation":"유동/직장 강점, 임대료 수용 가능","confidence":0.78},\n'
            '    {"region":"서교동","explanation":"야간 유동/젊은층 강점","confidence":0.73}\n'
            "  ]\n"
            "}\n"
            "반드시 위 스키마를 지켜서 JSON만 출력해."
        )
        resp = _client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        txt = (resp.choices[0].message.content or "").strip()
        data = json.loads(txt)
        ranked = data.get("ranked", [])
        # region 기준으로 prelim 매칭
        out = []
        for item in ranked:
            region = item.get("region")
            if not region:
                continue
            match = next(((r, s, rsn) for (r, s, rsn) in prelim if r == region), None)
            if match:
                out.append((match[0], match[1], match[2], item.get("explanation"), item.get("confidence")))
        return out or None
    except Exception as e:
        print(f"[경고] LLM 재랭킹 실패: {e}")
        return None



# ==============================
# 설정 (필요 시 수정)
# ==============================
AGE_ONE_REGEXES = [
    (re.compile(r"^1인가구[_\- ]?(\d{2})[~∼\-](\d{2})세$"), lambda a,b: f"{a}~{b}세"),
    (re.compile(r"^1인가구[_\- ]?(\d{2})대$"),             lambda a,b=None: f"{a}대"),
    (re.compile(r"^1인가구[_\- ]?60대이상$"),              lambda *_: "60대이상"),
]
AGE_ALIAS = {
    "20~24세": "20~24세",
    "25~29세": "25~29세",
    "30~34세": "30~34세",
    "35~39세": "35~39세",
    "40~44세": "40~44세",
    "45~49세": "45~49세",
}

# -----------------------------
# LLM 재랭킹 유틸
# -----------------------------
def _extract_features_for_llm(
    dong: str,
    flow_row: Optional[pd.Series],
    worker_row: Optional[pd.Series],
    resident_row: Optional[pd.Series],
    rent_row: Optional[pd.Series],
    seoul_avg: Dict[str, Any],
) -> Dict[str, Any]:
    """LLM이 이해하기 쉽게 정리된 정량 피처 묶음."""
    def _get(row, col):
        try:
            if isinstance(row, pd.Series):
                v = row.get(col)
                return float(v) if pd.notna(v) else None
        except Exception:
            pass
        return None

    # 유동
    flow_total = _get(flow_row, "총_유동인구수")
    flow_ratio = _safe_ratio(flow_total or 0.0, seoul_avg.get("flow_avg", 1.0)) if flow_total else None

    # 직장: 20/30/40대 합, 전체 합
    worker_cols_20_40 = [c for c in ["20대_직장인구수","30대_직장인구수","40대_직장인구수"]
                         if isinstance(worker_row, pd.Series) and c in worker_row.index]
    worker_cols_all = [c for c in [
        "10대_직장인구수","20대_직장인구수","30대_직장인구수",
        "40대_직장인구수","50대_직장인구수","60대이상_직장인구수"
    ] if isinstance(worker_row, pd.Series) and c in worker_row.index]
    worker_20_40 = _sum_cols(worker_row, worker_cols_20_40) if worker_cols_20_40 else 0.0
    worker_total = _sum_cols(worker_row, worker_cols_all) if worker_cols_all else 0.0
    worker_20_40_share = _safe_ratio(worker_20_40, worker_total) if worker_total>0 else None

    # 상주: 30/40대 합, 전체 합
    resident_cols_30_40 = [c for c in ["30대_상주인구수","40대_상주인구수"]
                           if isinstance(resident_row, pd.Series) and c in resident_row.index]
    resident_cols_all = [c for c in [
        "10대_상주인구수","20대_상주인구수","30대_상주인구수",
        "40대_상주인구수","50대_상주인구수","60대이상_상주인구수"
    ] if isinstance(resident_row, pd.Series) and c in resident_row.index]
    resident_30_40 = _sum_cols(resident_row, resident_cols_30_40) if resident_cols_30_40 else 0.0
    resident_total = _sum_cols(resident_row, resident_cols_all) if resident_cols_all else 0.0
    resident_30_40_share = _safe_ratio(resident_30_40, resident_total) if resident_total>0 else None

    # 임대료: 서울 평균 대비
    rent_ratio_to_seoul = None
    if isinstance(rent_row, pd.Series):
        rent_cols = [c for c in rent_row.index if str(c).startswith("임대료_")]
        if rent_cols:
            try:
                vals = [float(rent_row[c]) for c in rent_cols if pd.notna(rent_row[c])]
                local_avg = sum(vals)/len(vals) if vals else None
            except Exception:
                local_avg = None
            if local_avg is not None and seoul_avg.get("rent_avg"):
                rent_ratio_to_seoul = _safe_ratio(local_avg, seoul_avg["rent_avg"])

    return {
        "region": dong,
        "flow_total": flow_total,
        "flow_ratio_vs_seoul": flow_ratio,           # 높을수록 좋음
        "worker_20_40_share": worker_20_40_share,    # 0~1 (높을수록 유리)
        "resident_30_40_share": resident_30_40_share,# 0~1
        "rent_ratio_vs_seoul": rent_ratio_to_seoul,  # 낮을수록 유리
    }


from typing import Optional

LLM_RERANK_ENABLED = True  # 끄고 싶으면 False------------------>>>>

def _llm_rerank_industry(type_small: str, prelim: list, topk: int = 5) -> Optional[list]:
    """
    prelim: [
      {
        "region": str,
        "raw_score": float,
        "features": [{"name": str, "text": str, "weight": float}, ...],  # <= 반드시 존재
        "biz_feature": str (optional)
      }, ...
    ]
    return: [{"region": str, "score": float, "rationale": str}, ...] or None
    """
    if not LLM_RERANK_ENABLED or _client is None:
        return None
    try:
        # 방어적으로 features를 보정 (혹시라도 None/누락이면 빈 리스트로)
        safe_prelim = []
        for c in prelim:
            feats = c.get("features") or []
            # feats가 제대로 된 구조인지 한 번 더 보정
            fixed_feats = []
            for f in feats:
                if not isinstance(f, dict):
                    continue
                fixed_feats.append({
                    "name": f.get("name", ""),
                    "text": str(f.get("text", "")),
                    "weight": float(f.get("weight", 0.0))
                })
            safe_prelim.append({
                "region": c.get("region", ""),
                "raw_score": float(c.get("raw_score", 0.0)),
                "features": fixed_feats,
                "biz_feature": c.get("biz_feature", "")
            })

        # 프롬프트 구성(간결 JSON)
        user_payload = {
            "task": "RERANK_REGIONS_FOR_CATEGORY",
            "category": type_small,
            "candidates": safe_prelim,
            "topk": int(topk),
            "instructions": [
                "각 후보의 features.weight를 가중치로 반영해 종합 점수를 매겨주세요.",
                "텍스트는 정합성(모순/과장/의미불명 표현 여부)만 체크하여 감점 요인으로 활용.",
                "최종 출력은 score 내림차순 정렬된 JSON 리스트",
                "형식: [{\"region\": str, \"score\": float, \"rationale\": str}, ...]"
            ]
        }

        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a ranking assistant for retail site selection. Be concise and consistent."},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=0.2,
            max_tokens=400
        )
        text = (resp.choices[0].message.content or "").strip()
        # 파싱
        try:
            parsed = json.loads(text)
        except Exception:
            # 모델이 코드블록으로 감싸거나 앞뒤로 말 붙일 수 있으니 숫자/키만 추출 시도
            m = re.search(r'\[.*\]', text, re.S)
            parsed = json.loads(m.group(0)) if m else []

        # 최소 필드 검증
        out = []
        for item in parsed:
            region = item.get("region")
            score  = item.get("score")
            rationale = item.get("rationale", "")
            if isinstance(region, str) and isinstance(score, (int, float)):
                out.append({"region": region, "score": float(score), "rationale": str(rationale)})
        if not out:
            return None

        # 상위 topk만
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:topk]

    except Exception as e:
        print(f"[경고] LLM 재랭킹 실패: {e}")
        return None


# ==== biz_feature 키워드 추출기 =========================================
import re
from typing import Optional, Dict, Any, Set

_POS_WORDS = r"(높|강|많|집중|핵심|우세|유리|풍부|활발|밀집|활성|강세)"
_NEG_WORDS = r"(낮|약|부족|취약|희소|쇠퇴|약세)"
_RENT_POS  = r"(저렴|합리|낮\s*은?\s*임대료)"
_RENT_NEG  = r"(비싸|높\s*은?\s*임대료|고임대료)"

def _extract_topics_from_biz_feature(text: Optional[str]) -> Dict[str, Any]:
    """
    biz_feature 문장에서 4개 축(유동인구/직장인구/연령층/임대료) 신호를 추출.
    반환 예:
    {
      "유동인구": "high" | "mentioned" | None,
      "직장인구": "high" | "mentioned" | None,
      "연령층": {"젊은층","노년층","1인가구"},  # set (없으면 빈 set)
      "임대료": "low" | "high" | "mentioned" | None
    }
    """
    result = {"유동인구": None, "직장인구": None, "연령층": set(), "임대료": None}
    if not text or not isinstance(text, str):
        return result

    t = re.sub(r"\s+", " ", text.lower())

    def has(pats):
        if isinstance(pats, (list, tuple, set)):
            return any(re.search(p, t) for p in pats)
        return re.search(pats, t) is not None

    # 유동인구
    if has([r"유동\s*인구", r"보행", r"유입", r"트래픽", r"체류"]):
        result["유동인구"] = "high" if has(_POS_WORDS) and not has(_NEG_WORDS) else "mentioned"

    # 직장인구
    if has([r"직장", r"오피스", r"직장인", r"점심\s*수요", r"근로"]):
        result["직장인구"] = "high" if has(_POS_WORDS) and not has(_NEG_WORDS) else "mentioned"

    # 연령층
    if has([r"20대", r"30대", r"청년", r"젊"]):   result["연령층"].add("젊은층")
    if has([r"60대", r"시니어", r"노인", r"고령", r"실버"]): result["연령층"].add("노년층")
    if has([r"1\s*인\s*가구", r"1인가구", r"원룸", r"오피스텔", r"싱글"]): result["연령층"].add("1인가구")

    # 임대료 (낮음=호재)
    if has([r"임대료", r"월세", r"렌트", r"보증금"]):
        if has(_RENT_POS):   result["임대료"] = "low"
        elif has(_RENT_NEG): result["임대료"] = "high"
        else:                result["임대료"] = "mentioned"

    return result
# =======================================================================



# 동별 입지 힌트 JSON 파일 merge
def merge_json_files(paths: list[str]) -> dict:
    merged = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                merged.update(d)  # 동일 키는 뒤에 오는 파일이 덮어씀
            else:
                print(f"[경고] {p} 최상위가 dict 아님. 건너뜀.")
        except Exception as e:
            print(f"[경고] {p} 로딩 실패: {e}")
    return merged

LOCAL_HINTS = merge_json_files([
    os.path.join("local_hints", "ddm.json"),
    os.path.join("local_hints", "dobong.json"),
    os.path.join("local_hints", "dongjak.json"),
    os.path.join("local_hints", "eunpyeong.json"),
    os.path.join("local_hints", "gangbuk.json"),
    os.path.join("local_hints", "gangdong.json"),
    os.path.join("local_hints", "gangnam.json"),
    os.path.join("local_hints", "gangseo.json"),
    os.path.join("local_hints", "geumchun.json"),
    os.path.join("local_hints", "guro.json"),
    os.path.join("local_hints", "gwanak.json"),
    os.path.join("local_hints", "gwangjin.json"),
    os.path.join("local_hints", "jongno.json"),
    os.path.join("local_hints", "jung.json"),
    os.path.join("local_hints", "jungnang.json"),
    os.path.join("local_hints", "mapo.json"),
    os.path.join("local_hints", "nowon.json"),
    os.path.join("local_hints", "sdm.json"),
    os.path.join("local_hints", "seocho.json"),
    os.path.join("local_hints", "songpa.json"),
    os.path.join("local_hints", "sungbuk.json"),
    os.path.join("local_hints", "sungdong.json"),
    os.path.join("local_hints", "yangcheon.json"),
    os.path.join("local_hints", "ydp.json"),
    os.path.join("local_hints", "yongsan.json")
])


# ==============================
# 공통 유틸
# ==============================
def _norm_region(s: Any) -> Any:
    if not isinstance(s, str): return s
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\(.+?\)", "", s)
    return s

def _ensure_dong_norm(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame): return df
    if "행정동명" in df.columns:
        df["행정동명"] = df["행정동명"].astype(str).map(_norm_region)
    return df

def _ensure_gu_norm(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame): return df
    for c in ["자치구명", "자치구", "구명"]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(_norm_region)
    return df

def _safe_mean(x) -> Optional[float]:
    try:
        s = pd.to_numeric(x, errors="coerce")
        v = float(s.mean())
        return v if pd.notna(v) else None
    except Exception:
        return None

def _percent(a, b) -> Optional[float]:
    try:
        if a is None or b in (None, 0) or pd.isna(b):
            return None
        return round((float(a) / float(b)) * 100, 0)
    except Exception:
        return None

def _pct_txt(p: Optional[float]) -> str:
    return "해당 데이터가 부족합니다." if (p is None or pd.isna(p)) else f"{int(round(float(p)))}%"

def _mf_txt(p: Optional[float]) -> str:
    if p is None or pd.isna(p): return "비교 불가"
    try:
        return "많은편" if float(p) >= 100 else "적은편"
    except Exception:
        return "비교 불가"

def _latest_by_dong(df: Optional[pd.DataFrame], date_col="기준년분기") -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame) or df.empty or "행정동명" not in df.columns:
        return None
    if date_col not in df.columns:
        return df.groupby("행정동명", as_index=False).tail(1).set_index("행정동명")
    try:
        return (df.sort_values(date_col)
                  .groupby("행정동명", as_index=False)
                  .tail(1)
                  .set_index("행정동명"))
    except Exception:
        return (df.groupby("행정동명", as_index=False)
                  .tail(1)
                  .set_index("행정동명"))

def _safe_ratio(n: float, d: float) -> float:
    try:
        n = float(n); d = float(d)
        if d == 0: return 0.0
        return n / d
    except Exception:
        return 0.0

def _sum_cols(row, cols):
    v = 0.0
    for c in cols:
        x = row.get(c)
        if pd.notna(x):
            try:
                v += float(x)
            except:
                pass
    return v

def _resolve_gu_from_dfs(region_name: str, dfs: Dict[str, Optional[pd.DataFrame]]) -> Optional[str]:
    cand_keys = ["resident", "flow", "worker", "mall", "rent"]
    gu_cols = ["자치구명", "자치구", "구명"]
    for key in cand_keys:
        df = dfs.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "행정동명" not in df.columns:
            continue
        sub = df[df["행정동명"] == region_name]
        if sub.empty:
            continue
        for gc in gu_cols:
            if gc in sub.columns:
                val = sub.iloc[0][gc]
                if isinstance(val, str) and val.strip():
                    return _norm_region(val)
    return None


# ==============================
# 1인가구 라벨 유틸
# ==============================
def _friendly_age_label(colname: str) -> Optional[str]:
    colname = str(colname).strip()
    mapping = {
        "계": None,
        "20세미만": "20세미만",
        "20~24세": "20~24세",
        "25~29세": "25~29세",
        "30~34세": "30~34세",
        "35~39세": "35~39세",
        "40~44세": "40~44세",
        "45~49세": "45~49세",
        "50~54세": "50~54세",
        "55~59세": "55~59세",
        "60~64세": "60~64세",
        "65~69세": "65~69세",
        "70~74세": "70~74세",
        "75~79세": "75~79세",
        "80~84세": "80~84세",
        "85세이상": "85세이상",
    }
    if colname in mapping:
        return mapping[colname]

    m = re.match(r"^(\d{2})\s*[~∼-]\s*(\d{2})세$", colname)
    if m:
        return f"{m.group(1)}~{m.group(2)}세"
    m = re.match(r"^(\d{2})세이상$", colname)
    if m:
        return f"{m.group(1)}세이상"
    m = re.match(r"^(\d{2})대$", colname)
    if m:
        return f"{m.group(1)}대"
    return None

def _normalize_one_person_df(one_df: pd.DataFrame) -> pd.DataFrame:
    df = one_df.copy()
    gu_col = next((c for c in ["자치구명", "자치구", "구명"] if c in df.columns), None)
    if gu_col is None:
        if df.index.name in [None, ""] and df.index.dtype == "object":
            df = df.reset_index().rename(columns={"index": "자치구명"})
            gu_col = "자치구명"
        elif df.index.name and df.index.dtype == "object":
            df = df.reset_index().rename(columns={df.index.name: "자치구명"})
            gu_col = "자치구명"
        else:
            first_col = df.columns[0]
            if df[first_col].astype(str).str.contains("구").any():
                df = df.rename(columns={first_col: "자치구명"})
                gu_col = "자치구명"

    if "성별" in df.columns and (df["성별"] == "계").any():
        df = df[df["성별"] == "계"]

    age_cols = []
    for c in df.columns:
        lab = _friendly_age_label(str(c))
        if lab:
            age_cols.append(c)

    keep_cols = [gu_col] + age_cols if gu_col else age_cols
    df = df[keep_cols]

    for c in age_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if gu_col:
        df[gu_col] = df[gu_col].astype(str).str.strip().str.replace(" ", "", regex=False)
        df = df.rename(columns={gu_col: "자치구명"})
    return df


# ==============================
# 임대료 데이터 정규화
# ==============================
def _normalize_rent_df(rent_df: pd.DataFrame) -> pd.DataFrame:
    """
    다양한 포맷을 다음 스키마로 표준화:
    - 자치구명, 행정동명, 집계수준(행정동/자치구), 임대료_YYYYQn...
    - A열 자치구명, B열(전체/행정동명) 구조 + 중간 공란도 처리
    """
    df = rent_df.copy()

    # 우선 컬럼명 정리
    df = df.rename(columns=lambda c: str(c).strip())

    # 이미 표준 스키마면 숫자화만 해서 반환
    rent_like = [c for c in df.columns if str(c).startswith("임대료_")]
    if "자치구명" in df.columns and ("행정동명" in df.columns or "집계수준" in df.columns) and rent_like:
        for c in rent_like:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "집계수준" not in df.columns:
            df["집계수준"] = df["행정동명"].apply(lambda x: "자치구" if str(x) in ["", "전체", "nan", "None"] else "행정동")
        return _ensure_dong_norm(_ensure_gu_norm(df))

    # 그렇지 않으면 "자치구-행정동-분기" 구조로 가정
    cols = list(df.columns)

    # A열=자치구, B열=전체 또는 행정동, 그 외=분기 값으로 가정
    A = cols[0]  # 자치구
    B = cols[1] if len(cols) > 1 else None  # 전체/행정동명
    quarter_cols_raw = cols[2:]

    # 자치구 Forward fill (중간 공란을 위 값으로 채움)
    df[A] = df[A].astype(str)
    df[A] = df[A].replace({"nan": None, "None": None})
    df[A] = df[A].ffill()

    # 행정동명/전체
    if B:
        df[B] = df[B].astype(str)

    # 분기 컬럼 표준명으로 변환
    def _qname(c):
        # 예) '2024년 3분기' -> '임대료_2024Q3'
        s = str(c)
        m = re.search(r"(\d{4}).*([1-4])분기", s)
        if m:
            return f"임대료_{m.group(1)}Q{m.group(2)}"
        return f"임대료_{s}"

    new_cols = {}
    for c in quarter_cols_raw:
        new_cols[c] = _qname(c)
    df = df.rename(columns=new_cols)

    # 스키마 구성
    out = pd.DataFrame()
    out["자치구명"] = df[A].astype(str).str.strip().str.replace(" ", "", regex=False)
    if B:
        bvals = df[B].fillna("").astype(str).str.strip()
        out["집계수준"] = bvals.apply(lambda x: "자치구" if (x == "" or x == "전체") else "행정동")
        out["행정동명"] = bvals.apply(lambda x: None if (x == "" or x == "전체") else _norm_region(x))
    else:
        out["집계수준"] = "자치구"
        out["행정동명"] = None

    # 임대료 수치
    rent_cols = [c for c in out.columns if c.startswith("임대료_")]  # 초기에는 없음
    for c in df.columns:
        if str(c).startswith("임대료_"):
            out[c] = pd.to_numeric(df[c], errors="coerce")
            rent_cols.append(c)
    rent_cols = sorted(set(rent_cols))

    out = _ensure_gu_norm(_ensure_dong_norm(out))
    return out

def _row_total_from_cols(row: pd.Series, cols: List[str]) -> Optional[float]:
    try:
        vals = pd.to_numeric(row[cols], errors="coerce")
        if not vals.notna().any():          # 전부 NaN이면 None
            return None
        s = float(vals.fillna(0).sum())
        return s                             # 0도 유효한 값으로 인정
    except Exception:
        return None




# ==============================
# 리포트 빌드 (JSON용)
# ==============================
def build_report_json(region_name: str,
                      dfs: Dict[str, Optional[pd.DataFrame]],
                      latest: Dict[str, Optional[pd.DataFrame]],
                      store_cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    flow_idx = latest.get("flow")
    res_idx  = latest.get("resident")
    work_idx = latest.get("worker")
    one_df   = dfs.get("one_person")
    rent_df  = dfs.get("rent")
    mall_df  = dfs.get("mall")

    # print("[체크] latest-keys:",
    # "flow", isinstance(latest.get("flow"), pd.DataFrame),
    # "resident", isinstance(latest.get("resident"), pd.DataFrame),
    # "worker", isinstance(latest.get("worker"), pd.DataFrame))
    # for name in ["flow","resident","worker"]:
    #     idx = latest.get(name)
    #     if isinstance(idx, pd.DataFrame):
    #         print(f"  - {name} rows:", len(idx))
    #         print(f"  - {name} has {region_name}? ->", region_name in idx.index)

    # ========== 0) 자치구명 추정 ==========
    gu_name = _resolve_gu_from_dfs(region_name, dfs)

    # ========== 1) 1인가구 ==========
    one_sentence = "해당 데이터가 부족합니다."
    if isinstance(one_df, pd.DataFrame) and gu_name:
        one_norm = _normalize_one_person_df(one_df)
        if "자치구명" in one_norm.columns:
            age_cols = [c for c in one_norm.columns if _friendly_age_label(str(c))]
            if age_cols:
                # 서울 평균(자치구별 총합의 평균)
                age_numeric = one_norm[age_cols].apply(pd.to_numeric, errors="coerce")
                seoul_totals = age_numeric.sum(axis=1)
                seoul_avg_total = _safe_mean(seoul_totals)

                sub = one_norm[one_norm["자치구명"] == gu_name]
                if not sub.empty:
                    gu_numeric = sub[age_cols].apply(pd.to_numeric, errors="coerce")
                    gu_total = float(gu_numeric.sum(axis=1).iloc[0] or 0.0)

                    pct_vs_seoul = _percent(gu_total, seoul_avg_total)   # 서울 평균 대비 %
                    shares = []
                    if gu_total > 0:
                        for c in age_cols:
                            v = float(pd.to_numeric(sub[c], errors="coerce").iloc[0] or 0.0)
                            shares.append((_friendly_age_label(c), round(v/gu_total*100, 0)))
                        shares.sort(key=lambda x: x[1], reverse=True)

                    if shares:
                        top1 = shares[0]
                        top2 = shares[1] if len(shares) > 1 else None
                        if top2:
                            one_sentence = (
                                f"{region_name}이 속한 {gu_name}의 1인가구 규모는 서울시 평균 {_pct_txt(pct_vs_seoul)}이고, "
                                f"특히 {top1[0]} 1인가구가 {int(top1[1])}%로 가장 많으며, "
                                f"{top2[0]} 1인가구가 {int(top2[1])}%로 그 뒤를 따른다."
                            )
                        else:
                            one_sentence = (
                                f"{region_name}이 속한 {gu_name}의 1인가구 규모는 서울시 평균 {_pct_txt(pct_vs_seoul)}이며, "
                                f"{top1[0]} 비중이 가장 높다."
                            )

    # ========== 2) 상주인구 ==========
    resident_sentence = "해당 데이터가 부족합니다."
    if isinstance(res_idx, pd.DataFrame) and region_name in res_idx.index:
        r = res_idx.loc[region_name].copy()
        # 행 전체를 숫자로 강제 (문자열/콤마 → NaN)
        r = pd.to_numeric(r, errors="coerce")

        # 서울 평균(타 동 평균) 대비 %
        pct_vs_seoul = None
        if "총_상주인구수" in res_idx.columns:
            others = pd.to_numeric(
                res_idx.drop(index=[region_name], errors="ignore")["총_상주인구수"],
                errors="coerce"
            )
            base = float(others.mean()) if others.notna().any() else None
            val  = float(r.get("총_상주인구수")) if pd.notna(r.get("총_상주인구수")) else None
            if base and val is not None:
                pct_vs_seoul = round((val / base) * 100)

        # 연령 점유율
        age_cols = [c for c in [
            "10대_상주인구수","20대_상주인구수","30대_상주인구수",
            "40대_상주인구수","50대_상주인구수","60대이상_상주인구수"
        ] if c in res_idx.columns]

        pairs = []
        if age_cols:
            age_vals = pd.to_numeric(r[age_cols], errors="coerce")
            total = float(age_vals.sum()) if age_vals.notna().any() else 0.0
            if total > 0:
                shares = (age_vals / total * 100).round(0)
                for c in age_cols:
                    if pd.notna(shares.get(c)):
                        pairs.append((c.replace("_상주인구수",""), int(shares[c])))
                pairs.sort(key=lambda x: x[1], reverse=True)

        if pairs:
            top1 = pairs[0]
            top2 = pairs[1] if len(pairs) > 1 else None
            sent = (f"{region_name}의 상주인구는 서울시 타 행정동 평균에 비해 "
                    f"{_pct_txt(pct_vs_seoul)} 으로 {_mf_txt(pct_vs_seoul)}이고, ")
            sent += f"연령대별로는 1위: {top1[0]} 상주인구수가 {top1[1]}%"
            if top2:
                sent += f", 2위: {top2[0]} 상주인구수가 {top2[1]}% 이다."
            else:
                sent += " 이다."
            resident_sentence = sent


    # ========== 3) 유동인구 ==========
    flow_sentence = "해당 데이터가 부족합니다."
    if isinstance(flow_idx, pd.DataFrame) and region_name in flow_idx.index:
        f = flow_idx.loc[region_name].copy()
        # 행 전체를 숫자로 강제
        f = pd.to_numeric(f, errors="coerce")

        # 서울 평균(타 동 평균) 대비 %
        pct_vs_seoul = None
        if "총_유동인구수" in flow_idx.columns:
            others = pd.to_numeric(
                flow_idx.drop(index=[region_name], errors="ignore")["총_유동인구수"],
                errors="coerce"
            )
            base = float(others.mean()) if others.notna().any() else None
            val  = float(f.get("총_유동인구수")) if pd.notna(f.get("총_유동인구수")) else None
            if base and val is not None:
                pct_vs_seoul = round((val / base) * 100)

        # 연령 점유율
        age_cols = [c for c in [
            "10대_유동인구수","20대_유동인구수","30대_유동인구수",
            "40대_유동인구수","50대_유동인구수","60대이상_유동인구수"
        ] if c in flow_idx.columns]

        pairs = []
        if age_cols:
            age_vals = pd.to_numeric(f[age_cols], errors="coerce")
            total = float(age_vals.sum()) if age_vals.notna().any() else 0.0
            if total > 0:
                shares = (age_vals / total * 100).round(0)
                for c in age_cols:
                    if pd.notna(shares.get(c)):
                        pairs.append((c.replace("_유동인구수",""), int(shares[c])))
                pairs.sort(key=lambda x: x[1], reverse=True)

        if pairs:
            top1 = pairs[0]
            top2 = pairs[1] if len(pairs) > 1 else None
            sent = (f"{region_name}의 유동인구는 서울시 타 행정동 평균에 비해 "
                    f"{_pct_txt(pct_vs_seoul)} 으로 {_mf_txt(pct_vs_seoul)}이고, ")
            sent += f"연령대별로는 1위: {top1[0]} 유동인구수가 {top1[1]}%"
            if top2:
                sent += f", 2위: {top2[0]} 유동인구수가 {top2[1]}% 이다."
            else:
                sent += " 이다."
            flow_sentence = sent


    # ========== 4) 직장인구 ==========
    worker_sentence = "해당 데이터가 부족합니다."
    if isinstance(work_idx, pd.DataFrame) and region_name in work_idx.index:
        # --- 임시 디버그: 실제 컬럼 확인 ---
        # dbg_cols = [c for c in work_idx.columns if "직장" in str(c)]
        # print("[DBG worker cols]", dbg_cols[:20])

        w_raw = work_idx.loc[region_name]
        # 1) 컬럼 정규화(혹시 남아있는 변형들 흡수)
        def _normalize_worker_row(row: pd.Series) -> pd.Series:
            import re
            out = {}
            for k, v in row.items():
                s = str(k).strip().replace(" ", "")
                s = s.replace("직장_인구", "직장인구").replace("_수", "수")
                # 연령대_10_직장인구수 -> 10대_직장인구수
                m = re.match(r"^연령대_(\d{2})_직장인구수$", s)
                if m:
                    s = f"{m.group(1)}대_직장인구수"
                s = s.replace("연령대_60_이상_직장인구수", "60대이상_직장인구수")
                out[s] = v
            return pd.Series(out)

        w = _normalize_worker_row(w_raw)
        w = pd.to_numeric(w, errors="coerce")

        # 사용할 연령 컬럼 집합(존재하는 것만)
        age_cols_worker = [c for c in [
            "10대_직장인구수","20대_직장인구수","30대_직장인구수",
            "40대_직장인구수","50대_직장인구수","60대이상_직장인구수"
        ] if c in w.index]

        # ── (A) 대상 동 총계: 1) 총_직장인구수 > 0  2) 연령합 > 0  3) 남+여 합 > 0 ──
        def _row_total(row: pd.Series) -> Optional[float]:
            val = row.get("총_직장인구수")
            if pd.notna(val) and float(val) > 0:
                return float(val)
            if age_cols_worker:
                s = float(pd.to_numeric(row[age_cols_worker], errors="coerce").sum())
                if s > 0: return s
            # 마지막 보정: 남+여
            male, female = row.get("남성_직장인구수"), row.get("여성_직장인구수")
            if pd.notna(male) and pd.notna(female) and (float(male) + float(female)) > 0:
                return float(male) + float(female)
            return None

        val_total = _row_total(w)

        # ── (B) 서울(타 동) 평균 분모: 위와 동일 로직을 모든 동에 적용 ──
        def _row_total_any(row: pd.Series) -> Optional[float]:
            r = pd.to_numeric(row, errors="coerce")
            # 1) 총계
            if "총_직장인구수" in r.index and pd.notna(r["총_직장인구수"]) and float(r["총_직장인구수"]) > 0:
                return float(r["총_직장인구수"])
            # 2) 연령합
            ac = [c for c in age_cols_worker if c in r.index]
            if ac:
                s = float(pd.to_numeric(r[ac], errors="coerce").sum())
                if s > 0: return s
            # 3) 남+여
            if "남성_직장인구수" in r.index and "여성_직장인구수" in r.index:
                m, f = r["남성_직장인구수"], r["여성_직장인구수"]
                if pd.notna(m) and pd.notna(f) and (float(m) + float(f)) > 0:
                    return float(m) + float(f)
            return None

        base = None
        if isinstance(work_idx, pd.DataFrame):
            # 현재 동 제외 후 각 행의 총계를 계산
            others_idx = work_idx.drop(index=[region_name], errors="ignore")
            totals = []
            for _, row in others_idx.iterrows():
                nrow = _normalize_worker_row(row)
                t = _row_total_any(pd.to_numeric(nrow, errors="coerce"))
                if t is not None and t > 0:
                    totals.append(t)
            if totals:
                base = float(pd.Series(totals).mean())

        # ── (C) 서울 평균 대비 % ──
        pct_vs_seoul = None
        if base and val_total is not None:
            pct_vs_seoul = int(round((val_total / base) * 100, 0))

        # ── (D) 연령 비중 ──
        pairs = []
        if age_cols_worker:
            s = pd.to_numeric(w[age_cols_worker], errors="coerce")
            total_age = float(s.sum()) if s.notna().any() else 0.0
            if total_age > 0:
                shares = (s / total_age * 100).round(0)
                for c in age_cols_worker:
                    if pd.notna(shares.get(c)):
                        pairs.append((c.replace("_직장인구수",""), int(shares[c])))
                pairs.sort(key=lambda x: x[1], reverse=True)

        # ── (E) 문장 조립 ──
        if val_total is not None and pairs:
            top1 = pairs[0]; top2 = pairs[1] if len(pairs) > 1 else None
            if pct_vs_seoul is not None:
                sent = (f"{region_name}의 직장인구는 서울시 타 행정동 평균에 비해 "
                        f"{_pct_txt(pct_vs_seoul)} 으로 "
                        f"{'적은편' if pct_vs_seoul < 100 else _mf_txt(pct_vs_seoul)}이고, ")
            else:
                sent = f"\"{region_name}\"동의 직장인구는 서울시 타 행정동 평균과의 직접 비교는 어려우나, "
            sent += f"연령대별로는 1위: {top1[0]} 직장인구수가 {top1[1]}%"
            if top2:
                sent += f", 2위: {top2[0]} 직장인구수가 {top2[1]}% 이다."
            else:
                sent += " 이다."
            worker_sentence = sent
        elif val_total is not None:
            # 총계만 확보된 경우
            worker_sentence = (f"\"{region_name}\"동의 직장인구 총계는 확인되며"
                               f"{(' 서울 평균 대비 ' + _pct_txt(pct_vs_seoul)) if pct_vs_seoul is not None else ''} 수준이다.")
        else:
            worker_sentence = "해당 데이터가 부족합니다."


    # ========== 5) 점포수 ==========
    store_sentence = "해당 데이터가 부족합니다."
    # store_cache를 우선 사용
    top_map = None
    if store_cache:
        region_top = store_cache.get("region_top", {})
        global_top = store_cache.get("global_top", {})
        top_map = region_top.get(region_name) or global_top
    # 없으면 mall_df 즉석 집계
    if top_map is None and isinstance(mall_df, pd.DataFrame) and not mall_df.empty:
        by_col = None
        for c in ["상권업종소분류명", "상권업종중분류명", "상권업종대분류명", "업종명", "업종코드"]:
            if c in mall_df.columns:
                by_col = c; break
        if by_col:
            sub = mall_df[mall_df["행정동명"] == region_name] if "행정동명" in mall_df.columns else mall_df
            if not sub.empty:
                vc = sub[by_col].value_counts().head(5)
                top_map = {str(k): int(v) for k, v in vc.items()}

    if top_map:
        items = sorted(top_map.items(), key=lambda x: x[1], reverse=True)[:5]
        rank_txt = ", ".join([f"{i+1}위: {name}({cnt})" for i, (name, cnt) in enumerate(items)])
        if gu_name:
            store_sentence = f"{gu_name} {region_name}의 점포 수는 {rank_txt}의 분포를 보이고 있다."
        else:
            store_sentence = f"{region_name}의 점포 수는 {rank_txt}의 분포를 보이고 있다."

    # ========== 6) 임대료 특징 ==========
    rent_sentence = "해당 데이터가 부족합니다."
    if isinstance(rent_df, pd.DataFrame) and not rent_df.empty:
        r = _normalize_rent_df(rent_df)

        rent_cols = [c for c in r.columns if str(c).startswith("임대료_")]
        if rent_cols:
            # 서울 평균 (모든 행의 분기값 평균)
            seoul_avg = _safe_mean(pd.to_numeric(r[rent_cols].stack(), errors="coerce"))
            # 자치구 평균
            gu_avg = None
            if gu_name:
                g1 = r[(r["자치구명"] == gu_name) & (r["집계수준"] == "자치구")]
                if not g1.empty:
                    gu_avg = _safe_mean(pd.to_numeric(g1[rent_cols].stack(), errors="coerce"))
                else:
                    g2 = r[(r["자치구명"] == gu_name) & (r["집계수준"] == "행정동")]
                    if not g2.empty:
                        gu_avg = _safe_mean(pd.to_numeric(g2[rent_cols].stack(), errors="coerce"))
            # 대상 동 평균
            local_avg = None
            d1 = r[(r["행정동명"] == region_name) & (r["집계수준"] == "행정동")]
            if not d1.empty:
                local_avg = _safe_mean(pd.to_numeric(d1[rent_cols].stack(), errors="coerce"))

            def _pct_safe(a, b):
                try:
                    if a is None or b is None or b == 0 or pd.isna(b):
                        return None
                    return round((float(a) / float(b)) * 100)
                except Exception:
                    return None

            if local_avg is not None:
                pct_vs_seoul = _pct_safe(local_avg, seoul_avg)
                if gu_avg is not None:
                    pct_vs_gu = _pct_safe(local_avg, gu_avg)
                    rent_sentence = (
                        f"{region_name}의 평균 임대료는 서울시 평균에 비해 "
                        f"({ '해당 데이터가 부족합니다.' if pct_vs_seoul is None else str(pct_vs_seoul)+'%'})이고, "
                        f" {gu_name} 평균에 비해 "
                        f"({ '해당 데이터가 부족합니다.' if pct_vs_gu is None else str(pct_vs_gu)+'%'})입니다."
                    )
                else:
                    rent_sentence = (
                        f"{region_name}의 평균 임대료는 서울시 평균에 비해 "
                        f"({ '해당 데이터가 부족합니다.' if pct_vs_seoul is None else str(pct_vs_seoul)+'%'}) 수준이다."
                    )
            elif gu_avg is not None:
                pct_vs_seoul = _pct_safe(gu_avg, seoul_avg)
                rent_sentence = (
                    f"행정동 단위 데이터가 부족하여 자치구({gu_name}) 평균으로 비교하면, "
                    f"서울시 평균 대비 "
                    f"({ '해당 데이터가 부족합니다.' if pct_vs_seoul is None else str(pct_vs_seoul)+'%'}) 수준이다."
                )

    # --------- 입지 특성 ---------
    location_sentence = LOCAL_HINTS.get(region_name, "해당 행정동의 입지 특성 데이터가 없습니다.")

    report_json = {
        "region": region_name,
        "report": {
            "인구": {
                "1인가구": one_sentence,
                "상주인구": resident_sentence,
                "유동인구": flow_sentence,
                "직장인구": worker_sentence
            },
            "업종": {
                "점포수": store_sentence,
                "임대료_특징": rent_sentence
            },
            "입지_특성": location_sentence
        }
    }
    return report_json

# ==============================
# 업종 리포트 (스코어 + 이유)
# ==============================
def _industry_score_and_reason(
    dong: str,
    flow_row: Optional[pd.Series],
    worker_row: Optional[pd.Series],
    resident_row: Optional[pd.Series],
    rent_row: Optional[pd.Series],
    seoul_avg: dict,
    type_small: str,
    biz_feature_text: Optional[str] = None  # ★ 추가
):

    total_flow = float(flow_row.get("총_유동인구수")) if isinstance(flow_row, pd.Series) and pd.notna(flow_row.get("총_유동인구수")) else 0.0
    worker_20_40 = 0.0
    if isinstance(worker_row, pd.Series):
        worker_20_40 = _sum_cols(worker_row, ["20대_직장인구수","30대_직장인구수","40대_직장인구수"])
    resident_30_40 = 0.0
    if isinstance(resident_row, pd.Series):
        resident_30_40 = _sum_cols(resident_row, ["30대_상주인구수","40대_상주인구수"])

    rent_val = None
    if isinstance(rent_row, pd.Series):
        rent_cols = [c for c in rent_row.index if str(c).startswith("임대료_")]
        if rent_cols:
            try:
                rent_vals = [float(rent_row[c]) for c in rent_cols if pd.notna(rent_row[c])]
                rent_val = sum(rent_vals) / len(rent_vals) if rent_vals else None
            except:
                rent_val = None

    flow_ratio = _safe_ratio(total_flow, seoul_avg.get("flow_avg", 1.0))
    worker_ratio = _safe_ratio(worker_20_40, seoul_avg.get("worker_20_40_avg", 1.0))
    resident_ratio = _safe_ratio(resident_30_40, seoul_avg.get("resident_30_40_avg", 1.0))

    rent_ratio_to_seoul = None
    if rent_val is not None and seoul_avg.get("rent_avg"):
        rent_ratio_to_seoul = _safe_ratio(rent_val, seoul_avg["rent_avg"])

    score = 0.0
    score += 0.40 * flow_ratio
    score += 0.30 * worker_ratio
    score += 0.20 * resident_ratio
    if rent_ratio_to_seoul is not None:
        score += 0.10 * (1.2 - min(1.2, rent_ratio_to_seoul))  # 싸면 가점

    reason = {}
    reason["유동인구"] = f"서울 평균 대비 약 {flow_ratio*100:.0f}% 수준으로 {'높음' if flow_ratio>=1.1 else '안정적 수준' if flow_ratio>=0.95 else '비교적 낮음'}"
    if isinstance(worker_row, pd.Series):
        total_worker = _sum_cols(worker_row, [
            "10대_직장인구수","20대_직장인구수","30대_직장인구수",
            "40대_직장인구수","50대_직장인구수","60대이상_직장인구수"
        ])
        pct_20_40 = _safe_ratio(worker_20_40, total_worker) * 100 if total_worker>0 else 0.0
        reason["직장인구"] = f"20~40대 직장인 비중이 {pct_20_40:.0f}%로 점심·저녁 수요를 뒷받침"
    if isinstance(resident_row, pd.Series):
        total_resident = _sum_cols(resident_row, [
            "10대_상주인구수","20대_상주인구수","30대_상주인구수",
            "40대_상주인구수","50대_상주인구수","60대이상_상주인구수"
        ])
        pct_30_40 = _safe_ratio(resident_30_40, total_resident) * 100 if total_resident>0 else 0.0
        reason["연령층"] = f"상주인구 중 30~40대 비중이 {pct_30_40:.0f}%로 주 고객층과 일치"
    if rent_ratio_to_seoul is not None:
        reason["임대료"] = f"서울 평균 대비 {rent_ratio_to_seoul*100:.0f}% 수준으로 {'높음' if rent_ratio_to_seoul>=1.1 else '안정적 수준' if rent_ratio_to_seoul>=0.95 else '비교적 낮음'}"
    reason["상권특징"] = LOCAL_HINTS.get(dong, "해당 행정동의 입지 특성 데이터가 없습니다.")

    # ---- biz_feature에서 키워드 추출하여 reason 보강 ----
    topics = _extract_topics_from_biz_feature(biz_feature_text)

    # 유동인구
    sig = topics.get("유동인구")
    if sig:
        if "유동인구" in reason and isinstance(reason["유동인구"], str):
            if sig == "high" and "강세" not in reason["유동인구"]:
                reason["유동인구"] += "; LLM: 유동인구 강세"
            elif sig == "mentioned" and "LLM" not in reason["유동인구"]:
                reason["유동인구"] += "; LLM 언급"
        else:
            reason["유동인구"] = "LLM 언급: 유동인구 " + ("강세" if sig == "high" else "중요")

    # 직장인구
    sig = topics.get("직장인구")
    if sig:
        if "직장인구" in reason and isinstance(reason["직장인구"], str):
            if sig == "high" and "강세" not in reason["직장인구"]:
                reason["직장인구"] += "; LLM: 직장인 수요 강세"
            elif sig == "mentioned" and "LLM" not in reason["직장인구"]:
                reason["직장인구"] += "; LLM 언급"
        else:
            reason["직장인구"] = "LLM 언급: 직장인 수요 " + ("강세" if sig == "high" else "중요")

    # 연령층 (set → 콤마 문자열)
    ages = topics.get("연령층") or set()
    if ages:
        add = "LLM 언급: " + ", ".join(sorted(ages))
        if "연령층" in reason and isinstance(reason["연령층"], str):
            if add not in reason["연령층"]:
                reason["연령층"] += ("; " if reason["연령층"] else "") + add
        else:
            reason["연령층"] = add

    # 임대료
    sig = topics.get("임대료")
    if sig:
        add = "LLM 언급: 임대료 " + ("저렴" if sig == "low" else "높음" if sig == "high" else "중요")
        if "임대료" in reason and isinstance(reason["임대료"], str):
            if add not in reason["임대료"]:
                reason["임대료"] += ("; " if reason["임대료"] else "") + add
        else:
            reason["임대료"] = add


    return score, reason


def build_industry_report(type_small: str, dfs: dict, topk: int = 5) -> dict:
    flow_df     = _ensure_dong_norm(dfs.get("flow"))
    worker_df   = _ensure_dong_norm(dfs.get("worker"))
    resident_df = _ensure_dong_norm(dfs.get("resident"))
    rent_df     = dfs.get("rent")
    rent_df     = _normalize_rent_df(rent_df) if isinstance(rent_df, pd.DataFrame) else None

    flow_latest     = _latest_by_dong(flow_df)
    worker_latest   = _latest_by_dong(worker_df)
    resident_latest = _latest_by_dong(resident_df)
    rent_latest     = _latest_by_dong(rent_df) if isinstance(rent_df, pd.DataFrame) else None

    seoul_flow_avg = 1.0
    if isinstance(flow_latest, pd.DataFrame) and "총_유동인구수" in flow_latest.columns:
        seoul_flow_avg = float(pd.to_numeric(flow_latest["총_유동인구수"], errors="coerce").mean())

    def _avg_worker_20_40(dfidx):
        if not isinstance(dfidx, pd.DataFrame): return 1.0
        cols = ["20대_직장인구수","30대_직장인구수","40대_직장인구수"]
        exist = [c for c in cols if c in dfidx.columns]
        if not exist: return 1.0
        vals = []
        for _, row in dfidx.iterrows():
            vals.append(_sum_cols(row, exist))
        return float(pd.Series(vals).mean()) if vals else 1.0

    def _avg_resident_30_40(dfidx):
        if not isinstance(dfidx, pd.DataFrame): return 1.0
        cols = ["30대_상주인구수","40대_상주인구수"]
        exist = [c for c in cols if c in dfidx.columns]
        if not exist: return 1.0
        vals = []
        for _, row in dfidx.iterrows():
            vals.append(_sum_cols(row, exist))
        return float(pd.Series(vals).mean()) if vals else 1.0

    def _avg_rent(dfidx):
        if not isinstance(dfidx, pd.DataFrame): return None
        rent_cols = [c for c in dfidx.columns if str(c).startswith("임대료_")]
        if not rent_cols: return None
        vals = []
        for _, row in dfidx.iterrows():
            try:
                v = sum(float(row[c]) for c in rent_cols if pd.notna(row[c])) / len(rent_cols)
                vals.append(v)
            except: pass
        return float(pd.Series(vals).mean()) if vals else None

    seoul_avg = {
        "flow_avg": seoul_flow_avg if seoul_flow_avg>0 else 1.0,
        "worker_20_40_avg": _avg_worker_20_40(worker_latest),
        "resident_30_40_avg": _avg_resident_30_40(resident_latest),
        "rent_avg": _avg_rent(rent_latest)
    }

    biz_text = _gen_biz_feature(type_small)


    dongs = set()
    for dfidx in [flow_latest, worker_latest, resident_latest]:
        if isinstance(dfidx, pd.DataFrame):
            dongs.update(dfidx.index.tolist())

    scored = []
    for dong in dongs:
        flow_row     = flow_latest.loc[dong] if isinstance(flow_latest, pd.DataFrame) and dong in flow_latest.index else None
        worker_row   = worker_latest.loc[dong] if isinstance(worker_latest, pd.DataFrame) and dong in worker_latest.index else None
        resident_row = resident_latest.loc[dong] if isinstance(resident_latest, pd.DataFrame) and dong in resident_latest.index else None
        rent_row     = rent_latest.loc[dong] if isinstance(rent_latest, pd.DataFrame) and dong in rent_latest.index else None

        score, reason = _industry_score_and_reason(
            dong, flow_row, worker_row, resident_row, rent_row, seoul_avg, type_small,
            biz_feature_text=biz_text
        )
        scored.append((dong, score, reason))



    

    # 1) 규칙기반 1차 후보(여유 폭: topk*2, 최소 6개)
    prelim = scored[:max(topk * 2, 6)]

    # 2) (옵션) LLM 재랭킹
    rerank_used = False
    llm_ranked = _llm_rerank_candidates(type_small, prelim)  # [(region, score, reason, exp, conf), ...] or None

    if llm_ranked:
        rerank_used = True
        # 상위 topk만 채택
        top = llm_ranked[:topk]
    else:
        # LLM 미사용/실패 시 규칙기반 그대로
        top = [(r, s, rsn, None, None) for (r, s, rsn) in scored[:topk]]

    scored.sort(key=lambda x: x[1], reverse=True)

    # === LLM 후보 생성: 항상 features 키를 채워 넣기 ===
    WEIGHTS = {"상주인구": 0.1,"유동인구": 0.30, "직장인구": 0.20, "연령층": 0.20, "임대료": 0.20}
    max_for_llm = 8  # LLM에 넘길 최대 후보 수 (너무 많으면 프롬프트가 길어짐)

    candidates_for_llm = []
    for dong, raw_score, reason in scored[:max_for_llm]:
        # reason(dict)에 있는 4개 지표를 features로 정규화
        feats = []
        for k, w in WEIGHTS.items():
            txt = reason.get(k)
            # 문자열이 아닐 수도 있으니 문자열화
            if txt is None:
                txt = ""
            feats.append({
                "name": k,
                "text": str(txt),
                "weight": w
            })
        candidates_for_llm.append({
            "region": dong,
            "raw_score": float(raw_score),
            "features": feats,                 # ✅ 항상 채워서 넘김
            "reason": reason,                  # 참고용(LLM 프롬프트/파싱엔 안 써도 됨)
        })


    # === (신규) LLM 재랭킹 시도 (실패 시 규칙기반 폴백) ===
    # 1) LLM에 보낼 후보(여유폭: topk*2, 최소 6개)
    prelim = scored[:max(topk * 2, 6)]
    candidates_for_llm = [
        {
            "region": d,
            "base_score": float(s),
            "reasons": r  # dict
        }
        for (d, s, r) in prelim
    ]

    reranked = _llm_rerank_industry(type_small, candidates_for_llm, topk=topk)

    out = {
        "type_small": type_small,
        "biz_feature": biz_text,
        "meta": {
            "rerank_used": bool(reranked),
            "rerank_model": RERANK_MODEL if reranked else None
        },
        "recommendations": []
    }

    if reranked:
        print("[LLM] 재랭킹 결과 사용")
        # reranked: [{"region": "...", "score": <float|int|None>, "rationale": "..."}, ...]
        for item in reranked:
            dong = item.get("region")
            if not dong:
                continue

            # 기존 reason 찾기 (prelim에서 찾아야 LLM 후보만 매칭됨)
            base_reason = next((r for (d, _, r) in prelim if d == dong), {})
            reason = dict(base_reason) if isinstance(base_reason, dict) else {}

            llm_score = item.get("score")
            rationale = item.get("rationale")

            # 안전한 문자열 조립
            if isinstance(llm_score, (int, float)) and rationale:
                reason["LLM평가"] = f"{int(round(llm_score))}점 - {rationale}"
            elif isinstance(llm_score, (int, float)):
                reason["LLM평가"] = f"{int(round(llm_score))}점"
            elif rationale:
                reason["LLM평가"] = rationale

            out["recommendations"].append({
                "region": dong,
                "reason": reason
            })
    else:
        print("[LLM] 실패 또는 비활성 → 규칙기반 결과 사용")
        for dong, _, reason in scored[:topk]:
            out["recommendations"].append({
                "region": dong,
                "reason": reason
            })

    return out




# ==============================
# 점포 사전계산
# ==============================
def precompute_latest_by_region(
    df: Optional[pd.DataFrame],
    key_col: str = "행정동명",
    date_col: str = "기준년분기"
) -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if key_col not in df.columns:
        return None

    # 1) 날짜가 있으면 날짜 기준 최신행
    if date_col in df.columns:
        try:
            return (df.sort_values(date_col)
                      .groupby(key_col, as_index=False)
                      .tail(1)
                      .set_index(key_col))
        except Exception:
            pass  # 폴백으로 이동

    # 2) 날짜가 없거나 정렬 실패 → 그냥 동별 마지막 행
    try:
        return (df.groupby(key_col, as_index=False)
                  .tail(1)
                  .set_index(key_col))
    except Exception:
        return None


def precompute_store_counts(mall_df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    if not isinstance(mall_df, pd.DataFrame) or mall_df.empty:
        return None
    df = mall_df.copy()
    by_col = None
    for c in ["상권업종소분류명", "상권업종중분류명", "상권업종대분류명", "업종명", "업종코드"]:
        if c in df.columns:
            by_col = c
            break
    if by_col is None:
        return None
    if "행정동명" in df.columns:
        g = df.groupby(["행정동명", by_col]).size().rename("cnt").reset_index()
        top = (
            g.sort_values(["행정동명", "cnt"], ascending=[True, False])
             .groupby("행정동명", as_index=False)
             .head(10)
        )
        region_map: Dict[str, Dict[str, int]] = {}
        for dong, sub in top.groupby("행정동명"):
            region_map[dong] = {str(k): int(v) for k, v in zip(sub[by_col], sub["cnt"])}
        return {"by_col": by_col, "region_top": region_map}
    else:
        vc = df[by_col].value_counts().head(10)
        return {"by_col": by_col, "global_top": {str(k): int(v) for k, v in vc.items()}}


# ==============================
# 디버그
# ==============================
"""
def _debug_check_region_availability(region_name: str, dfs: Dict[str, Optional[pd.DataFrame]]):
    def _count(df, name):
        if isinstance(df, pd.DataFrame):
            if "행정동명" in df.columns:
                cnt = int((df["행정동명"] == region_name).sum())
                return f"- {name}: '행정동명' O, 해당 동 행 수 = {cnt}"
            return f"- {name}: '행정동명' 컬럼 없음 (샘플: {list(df.columns)[:8]})"
        return f"- {name}: DataFrame 아님/비어있음"
    print("\n[디버그] 행정동 매칭 현황")
    for key in ["flow", "resident", "worker", "one_person", "rent", "mall"]:
        print(_count(dfs.get(key), key))
"""

# ==============================
# 엔트리
# ==============================
def main():
    print("데이터 로드 중...")
    try:
        from data_loader import load_all_data
    except Exception:
        print("[오류] data_loader.load_all_data를 가져올 수 없습니다.")
        sys.exit(1)

    try:
        resident, flow, mall, worker, rent, one_person = load_all_data()
    except Exception as e:
        print(f"[오류] load_all_data 실행 실패: {e}")
        sys.exit(1)

    # 정규화
    resident = _ensure_dong_norm(_ensure_gu_norm(resident))
    flow     = _ensure_dong_norm(_ensure_gu_norm(flow))
    worker   = _ensure_dong_norm(_ensure_gu_norm(worker))
    mall     = _ensure_dong_norm(_ensure_gu_norm(mall))
    rent     = _normalize_rent_df(_ensure_dong_norm(_ensure_gu_norm(rent))) if isinstance(rent, pd.DataFrame) else rent
    one_person = _ensure_gu_norm(one_person)

    dfs: Dict[str, Optional[pd.DataFrame]] = {
        "resident": resident, "flow": flow, "worker": worker,
        "rent": rent, "mall": mall, "one_person": one_person,
    }

    # 사전계산
    latest = {
        "flow": precompute_latest_by_region(flow),
        "resident": precompute_latest_by_region(resident),
        "worker": precompute_latest_by_region(worker),
    }
    store_cache = precompute_store_counts(mall)

    # ===== 인터랙티브 루프 =====
    while True:
        try:
            mode = input("모드 선택 (1: 지역 리포트, 2: 업종 리포트, q: 종료) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if mode in ("q", "quit", "exit"):
            print("종료합니다.")
            break

        if mode == "1":
            raw_region = input("행정동명을 입력하세요 (예: 전농1동) > ").strip()
            region = _norm_region(raw_region)
            # _debug_check_region_availability(region, dfs)
            result = build_report_json(region, dfs, latest, store_cache)
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif mode == "2":
            type_small = input("업종(소분류)을 입력하세요 (예: 일식 면 요리) > ").strip()
            result = build_industry_report(type_small, dfs, topk=2)
            print(json.dumps(result, ensure_ascii=False, indent=2))

        else:
            print("지원하지 않는 모드입니다. 1 또는 2 또는 q를 입력하세요.")
            continue

        print("\n==== RUN END ====\n")


if __name__ == "__main__":
    main()
