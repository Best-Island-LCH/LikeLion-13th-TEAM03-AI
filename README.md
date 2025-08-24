# LikeLion-13th-Assignment-Template
🦁 SKHU 멋쟁이사자처럼 13기 과제 PR 템플릿 레포지토리입니다.

```mermaid
flowchart LR
  subgraph Clients
    FE[Frontend] -->|HTTP| API
    BE[Backend] -->|HTTP| API
  end

  subgraph AI_Service["AI API (FastAPI)"]
    API["api.py<br/>/healthz<br/>/report/industry<br/>/report/region"]
    CORE["main.py<br/>build_industry_report()<br/>_industry_score_and_reason()"]
    LOADER["data_loader.py<br/>load_all_data()"]
    HINTS[(local_hints/*.json)]
    DATA[(raw_data/*.csv, *.xlsx)]
    LLM["OpenAI API<br/>(옵션, 실패 시 폴백)"]
  end

  API --> CORE
  CORE --> LOADER
  LOADER --> DATA
  CORE --> HINTS
  CORE -->|옵션| LLM

  subgraph Platform["Railway (Container)"]
    UVICORN["uvicorn<br/>PORT=$PORT"]
  end

  API <-->|stdout logs| UVICORN
```

```mermaid
sequenceDiagram
  participant C as Client (FE/BE)
  participant A as FastAPI (api.py)
  participant M as Scoring (main.py)
  participant D as Data Loader (data_loader.py)
  participant H as local_hints/*.json
  participant O as OpenAI(옵션)

  C->>A: POST /report/industry {type_small}
  A->>D: (최초 1회 로드 후 캐시된 DF 사용)
  A->>M: build_industry_report(type_small, dfs, topk=2)
  M->>M: _industry_score_and_reason(...) 반복, 상위 2개 선별
  alt LLM 사용 가능
    M->>O: biz_feature 요약 / 재랭킹 시도
    O-->>M: 결과(실패 시 폴백)
  end
  M->>H: 지역 힌트 문장 병합
  M-->>A: 리포트 JSON
  A-->>C: 200 OK + JSON
```

```mermaid
flowchart TB
  Dev[로컬 개발] --> GH[GitHub Repo]
  GH --> CI[Railway Build]
  CI --> IMG[Container Image]
  IMG --> RUNTIME[Railway Runtime (asia-southeast1 ...)]
  RUNTIME -->|Public URL| User
```

# 헬스체크
curl -sS https://<your-public-url>/healthz

# 지역 리포트
curl -X POST https://<your-public-url>/report/region \
  -H "Content-Type: application/json" \
  --data '{"region":"전농1동"}'

# 업종 리포트 (topk는 서버에서 2로 고정)
curl -X POST https://<your-public-url>/report/industry \
  -H "Content-Type: application/json" \
  --data '{"sex":"남성","type_large":"음식","type_medium":"일식","type_small":"일식 면 요리","budget":"5000만원 이상"}'
