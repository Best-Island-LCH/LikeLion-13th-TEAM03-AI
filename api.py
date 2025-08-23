# api.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd

# main.py의 로직 재사용 (import만 하면 __main__이 아니라 인터랙티브 루프는 실행 안 됨)
from main import (
    build_industry_report, build_report_json,
    precompute_latest_by_region,
    _ensure_dong_norm, _ensure_gu_norm, _normalize_rent_df
)

app = FastAPI(title="TEAM03 AI Report Service", version="0.1.0")

# CORS (해커톤 편의상 전체 허용; 운영 시 도메인 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- 프로세스 시작 시 1회 데이터 로드 ----
from data_loader import load_all_data
resident, flow, mall, worker, rent, one_person = load_all_data()

resident = _ensure_dong_norm(_ensure_gu_norm(resident))
flow     = _ensure_dong_norm(_ensure_gu_norm(flow))
worker   = _ensure_dong_norm(_ensure_gu_norm(worker))
mall     = _ensure_dong_norm(_ensure_gu_norm(mall))
rent     = _normalize_rent_df(_ensure_dong_norm(_ensure_gu_norm(rent))) if isinstance(rent, pd.DataFrame) else rent
one_person = _ensure_gu_norm(one_person)

dfs = {
    "resident": resident, "flow": flow, "worker": worker,
    "rent": rent, "mall": mall, "one_person": one_person,
}

latest = {
    "flow": precompute_latest_by_region(flow),
    "resident": precompute_latest_by_region(resident),
    "worker": precompute_latest_by_region(worker),
}
store_cache = None  # 필요하면 main.precompute_store_counts(mall) 사용

# ---- 요청 스키마 ----
class RegionReq(BaseModel):
    region: str

class IndustryReq(BaseModel):
    sex: Optional[str] = None              # "남성" / "여성" 등 (옵션)
    type_large: Optional[str] = None       # "음식"
    type_medium: Optional[str] = None      # "일식"
    type_small: str                        # "일식 면 요리"  ← 기존 필드 그대로 사용
    budget: Optional[str] = None           # "4,000만원 이하"
    topk: int = 2

    class Config:
        extra = "ignore"   # 추가로 들어오는 필드가 있어도 422 안 나게 무시

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/report/region")
def report_region(req: RegionReq):
    return build_report_json(req.region.strip(), dfs, latest, store_cache)

@app.post("/report/industry")
def report_industry(req: IndustryReq):
    return build_industry_report(req.type_small.strip(), dfs, topk=2)
