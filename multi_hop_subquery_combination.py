from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from dotenv import load_dotenv

load_dotenv(override=True)


# ===== 1. Pydantic 모델 정의 =====
class DecomposedAspects(BaseModel):
    """질문 분해 결과를 구조화하는 Pydantic 모델"""

    technical: str = Field(description="기술적 사양 관점의 검색 쿼리")
    safety: str = Field(description="안전 및 규정 관점의 검색 쿼리")
    contract: str = Field(description="계약 및 법적 관점의 검색 쿼리")
    schedule: str = Field(description="일정 및 비용 관점의 검색 쿼리")

    def to_list(self) -> List[str]:
        """모든 aspect를 리스트로 반환"""
        return [self.technical, self.safety, self.contract, self.schedule]


class SubQueries(BaseModel):
    """Sub-query 생성 결과를 구조화하는 Pydantic 모델"""

    queries: List[str] = Field(
        description="생성된 검색 쿼리 리스트 (2-4개)", min_items=2, max_items=4
    )


class CombinedQueries(BaseModel):
    """최종 조합 쿼리를 구조화하는 Pydantic 모델"""

    original_query: str = Field(description="원본 질문")
    expanded_queries: List[str] = Field(description="확장된 검색 쿼리들")
    combined_queries: List[str] = Field(description="조합된 검색 쿼리들")

    def get_all_queries(self) -> List[str]:
        """모든 쿼리를 통합하여 반환"""
        all_queries = (
            [self.original_query] + self.expanded_queries + self.combined_queries
        )
        return list(dict.fromkeys(all_queries))[:10]  # 중복 제거 후 최대 10개


# ===== 2. State 정의 =====
class MultiQueryState(TypedDict):
    """Multi-Query 생성을 위한 전체 State"""

    original_question: str
    sub_queries: Annotated[List[str], add]  # 리스트를 누적
    combined_queries: List[str]
    decomposed_aspects: List[str]


class SubQueryState(TypedDict):
    """개별 Sub-Query 생성을 위한 State"""

    aspect: str
    query_number: int


# ===== 3. LLM 모델 설정 =====
llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)


# ===== 4. 노드 함수 정의 (PydanticOutputParser 사용) =====
def decompose_question(state: MultiQueryState) -> Dict:
    """원본 질문을 여러 관점으로 분해합니다."""

    # PydanticOutputParser 생성
    parser = PydanticOutputParser(pydantic_object=DecomposedAspects)

    system_prompt = """당신은 건설 프로젝트 문서 검색 전문가입니다.
    주어진 질문을 다음 4개 관점에서 분석하여 각각의 구체적인 검색 쿼리로 분해하세요:
    1. 기술적 사양 관점
    2. 안전 및 규정 관점
    3. 계약 및 법적 관점
    4. 일정 및 비용 관점
    
    각 관점별로 구체적이고 검색 가능한 쿼리를 작성하세요.
    
    {format_instructions}"""

    user_prompt = f"다음 질문을 분해하세요: {state['original_question']}"

    messages = [
        SystemMessage(
            content=system_prompt.format(
                format_instructions=parser.get_format_instructions()
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    # 구조화된 출력 파싱
    parsed_output = parser.parse(response.content)
    aspects = parsed_output.to_list()

    print(f"\n🔍 질문 분해 완료: {len(aspects)}개 관점")
    print(f"  1. 기술적: {parsed_output.technical}")
    print(f"  2. 안전: {parsed_output.safety}")
    print(f"  3. 계약: {parsed_output.contract}")
    print(f"  4. 일정: {parsed_output.schedule}")

    return {"decomposed_aspects": aspects}


def generate_subquery(state: SubQueryState) -> Dict:
    """특정 관점에 대한 sub-query를 생성합니다."""

    # PydanticOutputParser 생성
    parser = PydanticOutputParser(pydantic_object=SubQueries)

    system_prompt = """당신은 RAG 시스템을 위한 검색 쿼리 생성 전문가입니다.
    주어진 관점에 대해 2-4개의 구체적이고 다양한 검색 쿼리를 생성하세요.
    각 쿼리는 다른 각도에서 정보를 찾을 수 있도록 작성하세요.
    
    {format_instructions}"""

    user_prompt = f"다음 관점에 대한 검색 쿼리를 생성하세요: {state['aspect']}"

    messages = [
        SystemMessage(
            content=system_prompt.format(
                format_instructions=parser.get_format_instructions()
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    # 구조화된 출력 파싱
    parsed_output = parser.parse(response.content)
    queries = parsed_output.queries

    print(f"  ✅ Query #{state['query_number']}: {queries[0]}")

    return {"sub_queries": queries}


def combine_queries(state: MultiQueryState) -> Dict:
    """Sub-query들을 조합하여 최종 검색 쿼리를 생성합니다."""

    sub_queries = state.get("sub_queries", [])
    original_question = state.get("original_question", "")

    # 확장 쿼리 생성
    expanded = sub_queries.copy()

    # 조합 쿼리 생성
    combined = []
    if len(sub_queries) >= 2:
        for i in range(min(3, len(sub_queries) - 1)):
            combined.append(f"{sub_queries[i]} AND {sub_queries[i+1]}")

    # 구조화된 출력 생성
    result = CombinedQueries(
        original_query=original_question,
        expanded_queries=expanded,
        combined_queries=combined,
    )

    # 모든 쿼리 통합
    all_queries = result.get_all_queries()

    # 5개 미만이면 변형 쿼리 추가
    while len(all_queries) < 5 and sub_queries:
        all_queries.append(
            f"관련 문서: {sub_queries[len(all_queries) % len(sub_queries)]}"
        )

    print(f"\n🎯 최종 쿼리 생성 완료: {len(all_queries)}개")

    return {"combined_queries": all_queries[:10]}


def route_to_subquery_generation(state: MultiQueryState) -> List[Send]:
    """각 aspect에 대해 Send로 sub-query 생성을 라우팅합니다."""

    aspects = state.get("decomposed_aspects", [])
    sends = []

    print(f"\n📤 {len(aspects)}개의 sub-query 생성 작업을 시작합니다...")

    for i, aspect in enumerate(aspects, 1):
        send_obj = Send("generate_subquery", {"aspect": aspect, "query_number": i})
        sends.append(send_obj)

    return sends


# ===== 5. 그래프 구성 =====
def create_multi_query_graph():
    """Multi-Query Generator 그래프를 생성합니다."""

    builder = StateGraph(MultiQueryState)

    # 노드 추가
    builder.add_node("decompose_question", decompose_question)
    builder.add_node("generate_subquery", generate_subquery)
    builder.add_node("combine_queries", combine_queries)

    # 엣지 추가
    builder.add_edge(START, "decompose_question")
    builder.add_conditional_edges(
        "decompose_question", route_to_subquery_generation, ["generate_subquery"]
    )
    builder.add_edge("generate_subquery", "combine_queries")
    builder.add_edge("combine_queries", END)

    # 그래프 컴파일
    return builder.compile()


# ===== 6. 메인 실행 함수 =====
def run_multi_query_generation(question: str):
    """주어진 질문에 대해 Multi-Query를 생성합니다."""

    # 그래프 생성
    app = create_multi_query_graph()

    print(f"\n{'='*60}")
    print(f"🔍 입력 질문: {question}")
    print(f"{'='*60}")

    # 그래프 실행
    result = app.invoke(
        {
            "original_question": question,
            "sub_queries": [],
            "combined_queries": [],
            "decomposed_aspects": [],
        }
    )

    # 결과 출력
    print(f"\n\n{'='*60}")
    print("📊 생성된 Multi-Query 결과 (구조화된 출력)")
    print(f"{'='*60}")

    queries = result.get("combined_queries", [])
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}]")
        print(f"  {query}")

    print(f"\n✅ 총 {len(queries)}개의 검색 쿼리가 생성되었습니다.")

    return result


# ===== 7. 테스트 실행 =====
if __name__ == "__main__":
    # 건설 프로젝트 관련 Multi-hop 질문 예제
    test_questions = [
        "서울 강남 A타워 프로젝트에서 지하 3층 콘크리트 타설 시 안전 규정과 품질 기준을 모두 만족하면서 우기 대비 공정 일정을 어떻게 조정해야 하나요?",
        "B건설 현장의 크레인 작업 시 풍속 제한 규정과 야간 작업 소음 규제를 고려할 때, 계약서상 공기 준수를 위한 대안은 무엇인가요?",
        "철근 콘크리트 구조물의 균열 발생 시 보수 방법과 관련 계약 조항, 그리고 추가 비용 청구 절차는 어떻게 되나요?",
    ]

    # 첫 번째 질문으로 전체 테스트
    for question in test_questions:
        result = run_multi_query_generation(question)

    print("\n" + "=" * 60)
    print("🎉 PydanticOutputParser를 활용한 구조화된 Multi-Query Generator 완료!")
    print("=" * 60)
