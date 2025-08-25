# gpt_client.py
import os
import re
from typing import Optional, Any

# .env 지원(없어도 무방)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenAI SDK가 없어도 더미 모드로 동작하도록 방어
try:
    from openai import OpenAI, OpenAIError
except Exception:  # SDK 미설치 등
    OpenAI = None
    class OpenAIError(Exception):
        pass

# 환경변수
API_KEY = os.getenv("OPENAI_API_KEY")
# 실제 호출 여부 플래그(없으면 기본 False)
USE_REAL = os.getenv("GPT_USE_REAL", "0").strip().lower() in ("1","true","yes","on")
DEFAULT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# 클라이언트 준비(조건 만족 시에만)
client: Optional[Any]= None
if USE_REAL and API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=API_KEY)
    except Exception as e:
        # 초기화 실패해도 더미로 폴백
        client = None
        print(f"[경고] OpenAI 클라이언트 초기화 실패: {e}")

#==============================================================
# gpt_client.py ─ 클라이언트 초기화 블록 바로 아래에 추가
print("[LLM-BOOT] OPENAI_API_KEY set?:", bool(API_KEY))
print("[LLM-BOOT] GPT_USE_REAL:", USE_REAL)
print("[LLM-BOOT] OpenAI SDK available?:", OpenAI is not None)
print("[LLM-BOOT] client:", "ON" if client else "OFF")
print("[LLM-BOOT] model:", DEFAULT_MODEL)
#================================================================

def _dummy_response(prompt: str) -> str:
    """
    개발/로컬에서 API 없이도 그럴듯한 문장을 돌려주기 위한 더미 생성기.
    프롬프트 패턴을 가볍게 파싱해서 업종 특화 문구도 지원.
    """
    # 업종 특성 프롬프트 탐지
    m = re.search(r"(.+?)\s*업종", prompt)
    kind = (m.group(1).strip() if m else "").replace('"','').replace("'", "")
    if kind:
        return (
            f"{kind} 업종은(는) 유동인구 흐름, 인근 직장인·상주 인구의 연령 구조, "
            f"경쟁 점포 밀집도와 임대료 수준 등을 종합 고려해 입지를 선정하는 것이 좋습니다. "
            f"핵심 수요 시간대(점심/저녁·주말)와 주 고객층(연령·성별)에 맞춘 상품·가격·동선을 설계하면 효과적입니다."
        )
    return "요청 내용을 간단히 요약합니다. (개발 모드: 더미 응답)"

def ask_gpt(
    prompt: str,
    model: Optional[str] = None,
    *,
    system: str = "너는 상권 분석 전문가야.",
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> str:
    """
    단일 프롬프트를 받아 응답 텍스트를 반환.
    - 실제 호출 조건: USE_REAL=1(또는 true/yes/on) & OPENAI_API_KEY 존재 & openai SDK 설치
    - 실패/부재 시 더미 응답 반환(실행 중단 없음)
    """
    model = model or DEFAULT_MODEL
    if client is None:
        return _dummy_response(prompt)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or _dummy_response(prompt)
    except OpenAIError as e:
        # 요금/쿼터/레이트리밋 등 모든 API 오류 → 폴백
        print(f"[경고] GPT 호출 실패: {e}")
        return _dummy_response(prompt)
    except Exception as e:
        print(f"[경고] GPT 호출 예외: {e}")
        return _dummy_response(prompt)

def gen_biz_feature(type_small: str) -> str:
    """
    업종 특성 문단(2~3문장) 생성. 실패 시 더미 문구로 폴백.
    """
    prompt = (
        f"{type_small} 업종의 지역적 상권 특성을 한국어로 2~3문장으로 간단 요약해줘. "
        f"가능하면 유동인구/직장인/상주 연령/임대료/경쟁 점포 관점으로 설명해."
    )
    return ask_gpt(prompt)

if __name__ == "__main__":
    print(gen_biz_feature("미용실"))
