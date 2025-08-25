# api.py
import os
import traceback
from threading import Thread
from typing import Optional, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# main.py의 로직 재사용
from main import (
    build_industry_report, build_report_json,
    precompute_latest_by_region,
    _ensure_dong_norm, _ensure_gu_norm, _normalize_rent_df,
    # 필요하면: precompute_store_counts
)

app = FastAPI(title="TEAM03 AI Report Service", version="0.1.0")

# CORS (데모 편의상 전체 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------------------
# 요청 바디 스키마
# -------------------------
class RegionReq(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Pydantic v2
    region: str

class IndustryReq(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sex: Optional[Literal["남성","여성"]] = None
    type_large: Optional[str] = None
    type_medium: Optional[str] = None
    type_small: str
    budget: Optional[str] = None
    # topk: int = 2

# -------------------------
# 헬스체크
# -------------------------
@app.get("/healthz")
def healthz():
    # 데이터 준비 상태를 함께 노출 (서버는 살아있음을 우선 보장)
    return {
        "ok": True,
        "data_ready": bool(getattr(app.state, "data_ready", False)),
        "error": getattr(app.state, "data_error", None),
    }

# -------------------------
# 데이터 컨텍스트 로더 (지연 로드)
# -------------------------
def _load_context():
    """
    무거운 CSV/엑셀 로딩과 전처리를 백그라운드에서 수행.
    실패해도 서버는 떠 있고 /healthz 는 200 응답.
    """
    try:
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

        # 필요하면 매장 상위 집계도 미리 계산
        # from main import precompute_store_counts
        # store_cache = precompute_store_counts(mall)
        store_cache = None

        # 앱 상태에 저장
        app.state.dfs = dfs
        app.state.latest = latest
        app.state.store_cache = store_cache
        app.state.data_ready = True
        app.state.data_error = None
        print("[startup] data context loaded")

    except Exception as e:
        app.state.data_ready = False
        app.state.data_error = str(e)
        print("[startup] data load failed:", e)
        traceback.print_exc()

@app.on_event("startup")
def _startup():
    # 서버는 즉시 뜨고, 데이터는 백그라운드에서 로드
    app.state.data_ready = False
    app.state.data_error = None
    Thread(target=_load_context, daemon=True).start()

def _ctx():
    """엔드포인트에서 쓸 컨텍스트 획득. 데이터 미준비 시 503."""
    if not getattr(app.state, "data_ready", False):
        raise HTTPException(status_code=503, detail="Data is still loading. Please retry in a moment.")
    return app.state.dfs, app.state.latest, app.state.store_cache

# -------------------------
# API 엔드포인트
# -------------------------
@app.post("/report/region")
def report_region(req: RegionReq):
    dfs, latest, store_cache = _ctx()
    return build_report_json(req.region.strip(), dfs, latest, store_cache)

@app.post("/report/industry")
def report_industry(req: IndustryReq):
    dfs, latest, _ = _ctx()
    return build_industry_report(req.type_small.strip(), dfs, topk=2)
