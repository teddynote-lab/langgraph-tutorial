# 📚 LangGraph Tutorial

> **LangGraph의 모든 것을 마스터하세요**  
> 이 종합 가이드는 초급자부터 고급 개발자까지 LangGraph의 핵심 개념부터 실무 활용까지 체계적으로 학습할 수 있도록 설계된 한국어 튜토리얼입니다. 실전 프로젝트와 심화 실습을 통해 복잡한 AI 에이전트 시스템을 구축하는 전문 역량을 키워보세요.

<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="https://langchain-ai.github.io/langgraph/static/wordmark_dark.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://langchain-ai.github.io/langgraph/static/wordmark_light.svg">
  <img alt="LangGraph Logo" src="https://langchain-ai.github.io/langgraph/static/wordmark_dark.svg" width="80%">
</picture>

<div>
<br>
</div>

[![Version](https://img.shields.io/pypi/v/langgraph.svg)](https://pypi.org/project/langgraph/)
[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://langchain-ai.github.io/langgraph/)

**LangGraph**는 장기 실행, 상태 유지 에이전트를 구축, 관리 및 배포하기 위한 저수준 오케스트레이션 프레임워크입니다. Klarna, Replit, Elastic 등 에이전트 미래를 선도하는 기업들에서 신뢰받고 있으며, 복잡한 AI 워크플로우를 위한 강력한 도구를 제공합니다.

## 🎯 핵심 기능

- **내구성 있는 실행**: 장애를 견디고 장기간 실행 가능한 에이전트 구축, 중단된 지점에서 자동 재개
- **휴먼-인-더-루프**: 실행 중 언제든지 에이전트 상태를 검사하고 수정하여 인간의 감독을 원활히 통합
- **포괄적인 메모리**: 진행 중인 추론을 위한 단기 작업 메모리와 세션 간 장기 지속 메모리를 갖춘 진정한 상태 유지 에이전트
- **LangSmith 디버깅**: 실행 경로 추적, 상태 전환 캡처, 상세한 런타임 메트릭을 제공하는 시각화 도구
- **프로덕션 배포**: 상태 유지 장기 실행 워크플로우의 고유한 문제를 처리하도록 설계된 확장 가능한 인프라

## ⬇️ 프로젝트 다운로드

다음 명령어를 사용하여 프로젝트를 다운로드하십시오:

```bash
git clone https://github.com/teddynote-lab/LangGraph-Tutorial.git
cd LangGraph-Tutorial
```

## 🔧 설치 방법

### UV 패키지 매니저를 사용한 설치

본 프로젝트는 `uv` 패키지 매니저를 사용하여 의존성을 관리합니다. 다음 단계를 따라 설치하십시오.

#### UV 설치 (사전 요구사항)

**macOS:**
```bash
# Homebrew 사용
brew install uv

# 또는 curl 사용
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
# PowerShell 사용
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 또는 pip 사용
pip install uv
```

#### 프로젝트 의존성 설치

UV가 설치되었다면, 다음 명령어로 프로젝트 의존성을 설치하십시오:

```bash
uv sync
```

이 명령어는 가상 환경을 자동으로 생성하고 모든 필요한 의존성을 설치합니다.

#### 가상 환경 활성화

```bash
# 가상 환경 활성화
source .venv/bin/activate  # macOS/Linux

# 또는
.venv\Scripts\activate     # Windows
```

## 📁 프로젝트 폴더 구성

```
langgraph-tutorial/
├── 01-QuickStart/          # LangGraph 빠른 시작 가이드
│   ├── 01-QuickStart-LangGraph-Tutorial.ipynb
│   ├── 02-QuickStart-LangGraph-Graph-API.ipynb
│   └── 03-QuickStart-LangGraph-Subgraph.ipynb
├── 02-Practice/            # 실습 문제 모음
│   ├── 01-Practice-Chatbot.ipynb
│   ├── 02-Practice-Tool-Integration.ipynb
│   ├── 03-Practice-Memory-Chatbot.ipynb
│   ├── 04-Practice-Human-in-the-Loop.ipynb
│   ├── 05-Practice-State-Customization.ipynb
│   ├── 06-Practice-Business-Email-Bot.ipynb
│   ├── 07-Practice-Send-Multi-Query-RAG.ipynb
│   ├── 08-Practice-Command-Workflow.ipynb
│   └── 09-Practice-Financial-Multi-Query-RAG.ipynb
├── 03-Modules/             # 핵심 모듈별 심화 학습
│   ├── 01-Core-Features/   # LangGraph 핵심 기능
│   ├── 02-RAG/            # Retrieval-Augmented Generation
│   ├── 03-Use-Cases/      # 실제 활용 사례들
│   ├── 04-MCP/            # Model Context Protocol
│   ├── 05-Supervisor/     # 멀티 에이전트 관리
│   └── 06-Memory/         # 메모리 관리 시스템
└── 99-Templates/          # 개발용 템플릿
    └── 00-Practice-Template.ipynb
```

**참고**
- 02-Practice/ 폴더에는 실습 문제가 있습니다. 이에 대한 정답지는 별도로 제공될 예정입니다.

### 폴더별 상세 설명

- **01-QuickStart/**: LangGraph의 기본 개념과 사용법을 빠르게 익힐 수 있는 입문 자료
- **02-Practice/**: 단계별 실습 문제를 통해 실무 능력을 향상시킬 수 있는 연습 자료
- **03-Modules/**: 각 기능별로 세분화된 심화 학습 자료
  - **01-Core-Features/**: 기본 기능, 챗봇, 에이전트, 메모리, 스트리밍 등
  - **02-RAG/**: 문서 검색 및 생성 통합 시스템 구현
  - **03-Use-Cases/**: 실제 비즈니스 시나리오 기반 활용 사례
  - **04-MCP/**: Model Context Protocol을 활용한 고급 통합
  - **05-Supervisor/**: 멀티 에이전트 협업 및 관리 시스템
  - **06-Memory/**: 장기 메모리 및 상태 관리 시스템
- **99-Templates/**: 새로운 실습이나 프로젝트 개발을 위한 기본 템플릿

## 🔗 참고 링크

### 📚 공식 문서 및 리포지토리
- [LangGraph 공식 GitHub](https://github.com/langchain-ai/langgraph) - LangGraph 소스 코드 및 최신 업데이트
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/) - 상세한 API 문서 및 가이드

### 🎓 학습 자료
- [테디노트 유튜브 채널](https://www.youtube.com/c/teddynote) - AI/ML 관련 한국어 강의 및 튜토리얼
- [RAG 고급 온라인 강의](https://fastcampus.co.kr/data_online_teddy) - 체계적인 RAG 시스템 구축 강의

## 📄 라이센스

본 프로젝트의 라이센스 정보는 [LICENSE](./LICENSE) 파일을 참조하십시오.

## 🏢 제작자

**Made by TeddyNote LAB**