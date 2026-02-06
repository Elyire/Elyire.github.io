---
title: 랭체인 LangChain "RAG from Scratch 시리즈" 요약 정리 (II)
date: 2024-04-04 20:10:00 +09:00
categories: [LLM, LangChain]
tags: [RAG, 랭체인, LangChain]
pin: false
slug: rag-from-scratch-langchain-02
redirect_from:
  - /posts/RAG-from-scratch-LangChain-02/
---
## **Part 10 Routing**

이번 영상에서는 쿼리 라우팅(Query Routing)에 초점을 맞춰 보겠습니다.

**🔧 문제(Problem):**  데이터 검색 시스템을 구축할 때, 우리는 종종 벡터 데이터베이스, SQL 데이터베이스 등 다양한 **데이터 저장소**와 검색에 사용할 **프롬프트**를 사용자 질의에 따라 선택해야 하는 문제에 직면합니다. 이는 모든 유형의 사용자 질의에 대해 단일 데이터 저장소나 고정된 프롬프트를 사용하는 것이 항상 최선의 결과를 보장하지는 않기 때문입니다.

**💡 아이디어(Idea)**: 이러한 상황에서 우리는 두 가지 전략을 고려해 볼 수 있습니다. 첫 번째는 논리적 라우팅(logical routing)으로, 대형 언어 모델을 활용하여 사용자 질의의 의도를 파악하고, 그에 가장 적합한 데이터 저장소를 선택하도록 하는 것입니다. 두 번째는 의미론적 라우팅(sementic routing)으로, 사용자 질문과 사전에 정의된 여러 프롬프트를 임베딩한 후, 유사도를 기준으로 사용자 질문에 가장 적합한 프롬프트를 선택하는 방식입니다. 

**📽️ 동영상**: [https://www.youtube.com/watch?v=pfpIndq7Fi8](https://www.youtube.com/watch?v=pfpIndq7Fi8){:target="_blank"}

 **💻 코드:** [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb){:target="_blank"}

![](https://pbs.twimg.com/media/GI9xPiKaIAIULpA?format=jpg&name=large)

## **Part 11. Query Structuring**

이번 영상에서는 쿼리 구조화(Query Structuring)에 초점을 맞춰 보겠습니다.

**🔧 문제(Problem):**  데이터베이스와 상호 작용할 때, 우리는 주로 SQL이나 Cypher와 같은 도메인 특화 언어(DSL)를 사용합니다. 이는 관계형 데이터베이스나 그래프 데이터베이스 등 각 데이터베이스 유형에 특화된 언어입니다. 또한, 많은 벡터 저장소는 메타데이터를 활용하여 데이터 청크를 필터링할 수 있는 구조화된 질의 기능을 제공합니다. 그러나 RAG 시스템은 자연어로 표현된 질문을 입력으로 받기 때문에, 이를 해당 데이터베이스에 맞는 구조화된 질의로 변환하는 과정이 필요합니다.

**💡 아이디어(Idea)**: 이러한 문제를 해결하기 위해, 최근 많은 연구가 쿼리 구조화(Query Structuring)에 집중하고 있습니다. 쿼리 구조화란 사용자가 자연어로 입력한 질문을 해당 데이터베이스에 특화된 도메인 특화 언어(DSL)로 변환하는 과정을 말합니다. 이를 통해 사용자는 자연어로 질문을 입력하고, 시스템은 이를 데이터베이스가 이해할 수 있는 구조화된 쿼리로 자동 변환하여 처리할 수 있게 됩니다. 영상에서는 함수 호출을 활용하여 벡터 저장소에 특화된 질의 구조화 방법을 소개하고 있으며, 참고자료 링크에서는 텍스트를 SQL이나 Cypher로 변환하는 방법을 자세히 다루고 있습니다.

**📽️ 동영상**: [https://www.youtube.com/watch?v=kl6NwWYxvbM](https://www.youtube.com/watch?v=kl6NwWYxvbM){:target="_blank"}

 **💻 코드:** [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb){:target="_blank"}

 **🧠 참고자료 (References):** 
 1. **Blog with links to various tutorials and templates:** [https://blog.langchain.dev/query-construction/](https://blog.langchain.dev/query-construction/){:target="_blank"}
 2. **Deep dive on graphDBs:** (courtesy of [@neo4j](https://twitter.com/neo4j){:target="_blank"}): [https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/](https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/){:target="_blank"}
 3. **Query structuring docs:** [https://python.langchain.com/docs/use_cases/query_analysis/techniques/structuring](https://python.langchain.com/docs/use_cases/query_analysis/techniques/structuring){:target="_blank"}
 4. **Self-query retriever docs:** [https://python.langchain.com/docs/modules/data_connection/retrievers/self_query](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query){:target="_blank"}

![](https://pbs.twimg.com/media/GJsHyb3aMAAQvhe?format=jpg&name=medium)

## **Part 12. Multi-Representation Indexing**

이번 영상에서는 원본 문서 전체(full documents)의 인덱싱을 위한 몇 가지 유용한 팁을 다룹니다.

**🔧 문제(Problem):**  RAG 시스템을 설계할 때, 많은 경우 문서를 일정한 크기의 청크로 분할하고, 사용자 질문과의 유사도를 기준으로 일부 청크를 선택하여 언어 모델(LLM)에 전달하는 방식을 사용합니다. 그러나 이 과정에서 청크의 크기와 개수를 적절히 설정하는 것은 쉽지 않은 문제입니다. 만약 선택된 청크가 질문에 답하기에 충분한 맥락을 포함하고 있지 않다면, 언어 모델이 정확하고 포괄적인 답변을 생성하기 어려울 수 있습니다. 즉, 부적절한 청크 분할로 인해 검색 및 답변 생성 성능이 저하될 위험이 있는 것입니다.

**💡 아이디어(Idea)**: 이러한 문제를 해결하기 위해, Proposition indexing(@tomchen0 외)이라는 흥미로운 연구가 제안되었습니다. 이 방법은 대형 언어 모델(LLM)을 활용하여 문서의 요약("propositions")을 생성하되, 이를 검색에 최적화된 형태로 만드는 것이 핵심입니다. 우리는 이 아이디어를 기반으로 두 가지 검색 방식을 개발했습니다.

첫 번째는 "다중 벡터 검색기(multi-vector retriever)"입니다. 이 방식은 **문서 요약을 임베딩**하여 검색에 활용하지만, 실제 언어 모델에는 **원본 문서 전체**를 전달합니다.

두 번째는 "상위 문서 검색기(parent-doc retriever)"입니다. 이 방식은 문서를 **청크 단위로 임베딩**하지만, 역시 언어 모델에는 **원본 문서 전체**를 제공합니다.

이 두 가지 검색 방식은 서로 다른 장점을 취하기 위해 고안되었습니다. 요약이나 청크와 같이 간결한 표현을 사용하면 검색 효율을 높일 수 있습니다. 하지만 동시에 이를 원본 문서와 연결함으로써, 언어 모델이 답변 생성에 필요한 모든 맥락을 활용할 수 있도록 합니다.

이러한 접근 방식은 매우 범용적이어서, 텍스트 문서뿐만 아니라 테이블이나 이미지 등 다양한 유형의 데이터에도 적용할 수 있습니다. 먼저 데이터의 요약을 생성하고 이를 인덱싱에 활용하되, 실제 언어 모델에는 원본 데이터를 전달하는 것입니다. 이를 통해 텍스트 기반의 유사도 검색을 위한 표현으로서의 **요약**을 활용함으로써, 우리는 테이블이나 이미지를 직접 임베딩할 때 직면할 수 있는 여러 어려움, 예를 들어 부적절한 청크 분할이나 멀티 모달 임베딩의 필요성 등을 우회할 수 있습니다.

**📽️ 동영상**: [https://www.youtube.com/watch?v=gTCU9I6QqCE&feature=youtu.be](https://www.youtube.com/watch?v=gTCU9I6QqCE&feature=youtu.be){:target="_blank"}

 **💻 코드:** [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_12_to_14.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_12_to_14.ipynb){:target="_blank"}

 **🧠 참고자료 (References):** 
 1. **Proposition indexing:** [https://arxiv.org/pdf/2312.06648.pdf](https://arxiv.org/pdf/2312.06648.pdf){:target="_blank"}
 2. **Multi-vector:** [https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector){:target="_blank"}
 3. **Parent-document:** [https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever){:target="_blank"}
 4. **Blog applying this to tables:** [https://blog.langchain.dev/semi-structured-multi-modal-rag/](https://blog.langchain.dev/semi-structured-multi-modal-rag/){:target="_blank"}
 5. **Blog applying this to images with eval:** [https://blog.langchain.dev/multi-modal-rag-template/](https://blog.langchain.dev/multi-modal-rag-template/){:target="_blank"}


## **Part 13. Indexing with RAPTOR**

이번 영상에서는 RAPTOR를 활용한 계층적 인덱싱(hierarchical indexing)에 초점을 맞춰 보겠습니다.

**🔧 문제(Problem):**  RAG 시스템은 사용자의 질문에 효과적으로 답하기 위해, 질문의 유형에 따라 서로 다른 수준의 정보를 처리할 수 있어야 합니다. 예를 들어, 어떤 질문은 특정 문서 내에서 찾을 수 있는 구체적인 사실을 요구할 수 있습니다. 이를 "하위 수준"의 질문이라고 할 수 있죠. 반면에, 여러 문서에 걸쳐 나타나는 개념이나 아이디어를 종합해야 하는 "상위 수준"의 질문도 있습니다.

기존의 RAG 시스템은 주로 문서를 일정한 크기의 청크로 나누고, 이를 기반으로 kNN 검색을 수행하는 방식을 사용해 왔습니다. 그러나 이 방법은 주로 하위 수준의 질문에 적합하며, 상위 수준의 질문을 처리하는 데에는 한계가 있습니다. 상위 수준의 개념이나 아이디어는 여러 문서에 분산되어 있을 가능성이 높기 때문입니다. 따라서 RAG 시스템이 다양한 수준의 질문에 효과적으로 대응하기 위해서는, 문서 청크 검색만으로는 부족하며 새로운 접근 방식이 필요합니다.

**💡 아이디어(Idea)**: 이러한 문제를 해결하기 위해, RAPTOR(Recursive Abstractive Processing for Tree-Organized Retrieval, @parthsarthi03 *et al.*)라는 새로운 접근 방식이 제안되었습니다. RAPTOR의 핵심 아이디어는 문서를 계층적으로 요약하여, 상위 수준의 개념을 효과적으로 포착하는 것입니다.

RAPTOR는 먼저 문서를 임베딩 공간에 투영하고, 유사한 문서들을 클러스터링합니다. 그 다음, 각 클러스터를 대표하는 요약을 생성합니다. 이 요약은 해당 클러스터에 속한 문서들의 공통적인 주제나 아이디어를 담고 있게 됩니다.

흥미로운 점은, RAPTOR가 이 과정을 재귀적으로 반복한다는 것입니다. 즉, 생성된 요약들을 다시 클러스터링하고, 이를 요약하는 식으로 계층을 구성해 나가는 거죠. 이렇게 하면 점점 더 추상적이고 고차원적인 개념을 포착하는 요약들의 트리가 만들어집니다.

최종적으로, RAPTOR는 이렇게 생성된 다양한 수준의 요약들과 원본 문서를 함께 인덱싱합니다. 따라서 사용자의 질문이 하위 수준의 구체적인 정보를 요구하는 경우에는 원본 문서나 하위 수준의 요약을, 상위 수준의 개념이나 아이디어를 묻는 경우에는 상위 수준의 요약을 검색 결과로 제공할 수 있게 됩니다.

이러한 계층적 인덱싱 방법은 RAG 시스템이 다양한 유형의 질문에 유연하게 대응할 수 있도록 도와줍니다. RAPTOR는 기존의 청크 기반 검색의 한계를 극복하고, 질문의 추상화 수준에 맞는 정보를 효과적으로 검색할 수 있는 새로운 가능성을 제시했습니다.

**📽️ 동영상**: [https://www.youtube.com/watch?v=z_6EeA2LDSw](https://www.youtube.com/watch?v=z_6EeA2LDSw){:target="_blank"}

 **💻 코드:** 
 - [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_12_to_14.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_12_to_14.ipynb){:target="_blank"}
 - [https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb](https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb){:target="_blank"}

 **🧠 참고자료 (References):** 
 1. **Paper:** [https://arxiv.org/pdf/2401.18059.pdf](https://arxiv.org/pdf/2401.18059.pdf){:target="_blank"}
 2. **Deep dive:** [https://www.youtube.com/watch?v=jbGchdTL7d0](https://www.youtube.com/watch?v=jbGchdTL7d0){:target="_blank"}
 3. **👉(한글자료🇰🇷)테디노트:** [https://www.youtube.com/watch?v=gcdkISrpMCA&t=152s](https://www.youtube.com/watch?v=gcdkISrpMCA&t=152s){:target="_blank"}


## **Part 14. Indexing with ColBERT**

이번 영상에서는 세밀한 유사도 검색을 위한 ColBERT를 사용한 인덱싱에 초점을 맞춰 보겠습니다.

**🔧 문제(Problem):**  임베딩 모델은 텍스트를 고정 길이의 벡터 표현으로 압축하여 문서의 의미론적 내용을 포착합니다. 이러한 압축은 효율적인 search와 retrieval에 매우 유용하지만, 문서의 모든 의미론적 뉘앙스와 문서의 세부 정보를 고정 길이의 단일 벡터 표현에 담아내야 하므로 이 벡터에 상당한 부담을 지웁니다. 경우에 따라 쿼리와 관련이 없거나 중복된 내용들이 벡터 표현에 포함되기도 하는데, 이는 검색을 위한 임베딩 벡터의 의미론적 유용성을 희석시킬 수 있습니다. 이는 궁극적으로 RAG의 검색 성능에 부정적인 영향을 미칠 수 있습니다.

**💡 아이디어(Idea)**: ColBERT(@lateinteraction & @matei_zaharia)는 더 세분화된 임베딩 접근 방식으로 이 문제를 해결하는 멋진 방법입니다:

(1) 문서와 쿼리의 각 토큰에 대해, BERT와 같은 사전 학습된 언어 모델을 이용해 해당 토큰의 문맥적 뉘앙스가 반영된 임베딩 벡터를 생성합니다(feat. 셀프 어텐션 메커니즘).

(2) 쿼리를 구성하는 첫번째 토큰과 문서 내의 모든 토큰 사이의 유사도를 계산합니다. 

(3) 이 중 가장 높은 유사도 점수를 선택합니다. 

(4) 쿼리를 구성하는 모든 토큰(두번째 토큰, 세번째 토큰...n번째 토큰)에 대해 (2)-(3) 과정을 반복하여 각 쿼리 토큰별로 문서 내 최고 매칭 토큰을 찾습니다(각 쿼리 토큰에 대해 문서 내에서 가장 유사도가 높은 토큰을 찾는 과정).

(5) 각 쿼리 토큰과 문서 토큰 간의 최고 유사도 점수를 모두 합산하여 최종적인 쿼리-문서 유사도 점수를 산출합니다. 

즉, 쿼리의 각 토큰이 문서와 얼마나 관련이 있는지를 개별적으로 평가하고, 그 결과를 종합하여 최종적인 쿼리-문서 유사성 점수를 도출합니다. 이렇게 문서와 쿼리 사이의 토큰 단위 유사도를 세밀하게 평가하는 방식은 기존 방법 대비 우수한 성능을 보여주었습니다. 

**📽️ 동영상**: [https://www.youtube.com/watch?v=cN6S0Ehm7_8](https://www.youtube.com/watch?v=cN6S0Ehm7_8){:target="_blank"}

 **💻 코드:** [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_12_to_14.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_12_to_14.ipynb){:target="_blank"}
 
 **🧠 참고자료 (References):** 
 1. **Paper:** [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832){:target="_blank"}
 2. **Nice review from [@DataStax](https://twitter.com/DataStax){:target="_blank"}**: [https://hackernoon.com/how-colbert-helps-developers-overcome-the-limits-of-rag](https://hackernoon.com/how-colbert-helps-developers-overcome-the-limits-of-rag){:target="_blank"}
 3. **Nice post from [@simonw](https://twitter.com/simonw){:target="_blank"}:** [https://til.simonwillison.net/llms/colbert-ragatouille](https://til.simonwillison.net/llms/colbert-ragatouille){:target="_blank"}
 4. **ColBERT repo:** [https://github.com/stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT){:target="_blank"}
 5. **RAGatouille to support RAG with ColBERT:** [https://github.com/bclavie/RAGatouille](https://github.com/bclavie/RAGatouille){:target="_blank"}

![](https://pbs.twimg.com/media/GJ7tExPakAAmwOK?format=jpg&name=large)

## **Part 15. Building Corrective RAG from scratch with open-source, local LLMs (Feedback + Self-reflection)**

RAG from Scratch 시리즈의 마지막 영상입니다. 이번 영상에서는 RAG 시스템 성능을 향상시키기 위한 self-reflection & feedback에 초점을 맞춰 보겠습니다.

**🔧 문제(Problem):**  RAG 시스템은 낮은 품질의 검색 결과(예: 사용자 질문이 인덱스의 도메인을 벗어나는 경우)나 생성 과정에서의 환각(hallucination, 모델이 사실이 아닌 내용을 마치 사실인 것처럼 생성하는 현상)으로 인해 어려움을 겪을 수 있습니다. 단순한 검색-생성 파이프라인은 이러한 종류의 오류를 감지하거나 스스로 교정할 수 있는 능력이 없습니다.

**💡 아이디어(Idea)**: 최근 @itamar_mar는 코드 생성 분야에서 "플로우 엔지니어링(Flow engineering)"이라는 개념을 소개했습니다. 이는 코드 질문에 대한 답을 반복적으로 구축하면서, 단위 테스트를 통해 오류를 확인하고 자체적으로 수정해 나가는 과정을 말합니다.

@HamelHusain의 블로그 포스트에서도 추론 루프에서 단위 테스트(unit test)의 유용성을 언급한 바 있습니다. 이러한 아이디어는 Self-RAG(@AkariAsai *et al.*)와 Corrective-RAG(@Jiachen_Gu and coworkers)와 같은 여러 RAG 관련 연구에서 적용되었습니다.

이 연구들에서는 RAG 답변 흐름(answer flow) 내에서 문서의 관련성, 환각 현상, 답변의 품질 등을 확인하는 과정을 수행합니다. 저희 연구진은 LangGraph를 활용하여 이러한 확인 및 피드백 과정을 조율하는 방식으로 위의 아이디어들을 구현했습니다.

또한 LangGraph를 사용하면 보다 작은 규모의 오픈소스 모델로도 안정적인 RAG 시스템을 구축할 수 있음을 보여주었습니다.

**📽️ 동영상**: [https://www.youtube.com/watch?v=E2shqsYwxck](https://www.youtube.com/watch?v=E2shqsYwxck){:target="_blank"}

 **💻 코드:** 
 - **C-RAG:** [https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb){:target="_blank"}
 - **Self-RAG:** [https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb){:target="_blank"}
 - **Both with MistralAI-7b + ollama:** 
  [https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_local.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_local.ipynb){:target="_blank"}
  [https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_local.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_local.ipynb){:target="_blank"}
  
 **🧠 참고자료 (References):** 
 1. **C-RAG:** [https://arxiv.org/pdf/2401.15884.pdf](https://arxiv.org/pdf/2401.15884.pdf)
 2. **Self-RAG:** [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)
 3. **Flow Engineering:** [https://twitter.com/karpathy/status/1748043513156272416?s=20](https://twitter.com/karpathy/status/1748043513156272416?s=20)
 4. **Blog on evals, covering unit tests:** [https://hamel.dev/blog/posts/evals/](https://hamel.dev/blog/posts/evals/)

![](https://pbs.twimg.com/media/GKQSf04aoAIQeSU?format=jpg&name=large)
