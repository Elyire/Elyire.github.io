---
title: 랭체인 LangChain "RAG from Scratch 시리즈" 요약 정리 (I)
date: 2024-04-04 20:00:00 +09:00
categories: [LLM, LangChain]
tags: [RAG, 랭체인, LangChain]
pin: false
slug: rag-from-scratch-langchain-01
permalink: /posts/rag-from-scratch-langchain-01/
redirect_from:
  - /posts/RAG-from-scratch-LangChain-01/
---
YouTube의 LangChain 채널에서는 [RAG from Scratch 시리즈](https://www.youtube.com/watch?v=wd7TZ4w1mSw&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x){:target="_blank"}가 연재되고 있습니다. 본 포스팅은 [LangChain의 X 계정](https://twitter.com/LangChainAI){:target="_blank"}의 소개글을 바탕으로 RAG from Scratch 각 파트가 어떤 내용을 다루고 있는지에 대한 정보를 제공합니다. X 계정의 소개글을 그대로 직역하지 않고, 독자가 이해하기 쉽도록 원문을 의역하고 필요하다면 간단한 설명을 추가하여 작성하였습니다. 

Part 1-4에서는 이 시리즈의 overview 및 RAG 파이프라인 구성 요소에 대한 간단한 설명을 제공합니다. 이 부분은 본 포스팅에서 다루지 않습니다. Part 5-9에서는 **Query Translation** 기법들을 다룹니다. Part 10과 11에서는 각각 **Routing**과 **Query Structuring**을, Part 12-14에서는 **Advanced Indexing** 기법들, 마지막으로 Part 15에서는 **단위 테스트를 통한 오류 확인 및 자체 수정 과정**에 대해 다룹니다. 앞으로 두 차례의 포스팅을 통해 이들을 정리해 보고자 합니다.

다음 그림은 Retrieval Augmented Generation (RAG) 시스템의 파이프라인(Query-Indexing-Retrieval-Generation)을 묘사하고 있으며, RAG 성능 향상을 위해 각 단계에서 사용될 수 있는 기법들이 정리되어 있습니다.   

![](https://lh7-us.googleusercontent.com/rNCNTufneIqDRJoSDSy7Dbs4abkvK34ju5AqtJIz2277KFn6ix9nD0RqPXzzD-jkmVCtkNQhEuMl6fSxO6xzNiJBV_k8BuZ56JBOqX0TeTYLotM3n5tPAvq7DXU2DnDou4_BKZ58ObwkScY1xn0nyicsew=s2048)


## **Part 5. Query Translation: Multi Query**

RAG From Scratch 동영상 시리즈는 짧고 집중적인 동영상과 코드를 통해 중요한 RAG 개념들을 살펴봅니다.

앞으로 몇 번에 걸쳐 다중 쿼리(Multi Query)를 시작으로 쿼리 번역(Query translation)에 초점을 맞춘 영상을 공개할 예정입니다.

 **🔧 문제(Problem)**: RAG 시스템에서 사용자의 질문을 처리하는 것은 종종 어려운 과제가 됩니다. 사용자가 모호한 질문을 하면, 단순히 질문과 문서 사이의 거리(유사도)만을 기준으로 검색하는 방식으로는 적절한 문서를 찾기 어려울 수 있기 때문입니다. 이런 경우 검색 결과로 나온 문서들도 질문의 의도를 명확히 파악하기 어려운, 즉 모호한 내용을 담고 있을 가능성이 높습니다.
 
**💡 아이디어(Idea)**: 사용자의 질문을 다양한 관점에서 재작성하는 것이 해결책이 될 수 있습니다. 먼저 원래의 질문을 여러 가지 방식으로 바꿔 표현해 보는 것입니다. 그 다음 재작성된 각각의 질문에 대해 문서를 검색합니다. 이렇게 하면 원래 질문의 모호함을 해소하면서도, 관련된 다양한 문서를 얻을 수 있습니다. 마지막으로 검색된 모든 문서 중 중복을 제거하여, 사용자의 질문에 대한 최종 검색 결과를 제시하는 것입니다.

 **📽️ 동영상**: [https://youtube.com/watch?v=JChPi0CRnDY](https://youtube.com/watch?v=JChPi0CRnDY){:target="_blank"}

 **💻 코드:** [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb){:target="_blank"}

 **🧠 관련 작업 / 아이디어들(Related works / Ideas):** 
1. **Ma et al with RRR:** [https://arxiv.org/pdf/2305.14283.pdf](https://arxiv.org/pdf/2305.14283.pdf){:target="_blank"}
2. **@Raudaschl with RAG-Fusion:** [https://github.com/Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion){:target="_blank"}
3. **@RickLamers:** [https://twitter.com/RickLamers/status/1673624398874419201?s=20](https://twitter.com/RickLamers/status/1673624398874419201?s=20){:target="_blank"} 


## **Part 6. Query Translation: RAG Fusion**

이번 영상은 @Raudaschl의 RAG-Fusion에 초점을 맞춘 Query Translation에 관한 두 번째 영상입니다.

**🔧 문제(Problem):** RAG 시스템에서 사용자가 입력한 단일 검색 쿼리는 몇 가지 한계점을 가지고 있습니다. 우선 단일 쿼리만으로는 사용자가 실제로 관심 있어 하는 주제의 범위를 모두 포괄하기 어려울 수 있습니다. 또한 단일 쿼리에 대한 검색 결과는 종종 부분적이거나 불완전할 수 있어, 사용자가 원하는 종합적인 정보를 제공하지 못할 수도 있습니다. 

**💡 아이디어(Idea)**: 이 문제를 해결하기 위해 사용자 쿼리를 여러 관점에서 재작성하고, 재작성된 각 쿼리에 대한 문서를 검색한 후, **Reciprocal Rank Fusion(RRF)을 사용**하여 여러 검색 결과 리스트들의 순위를 결합해 **최종적인 통합 순위(unified ranking)를 생성**합니다. 그리고 이 통합된 순위에 따라 정렬된 문서를 토대로 최종 답변을 생성합니다. 

**📽️ 동영상**: [https://www.youtube.com/watch?v=77qELPbNgxA](https://www.youtube.com/watch?v=77qELPbNgxA){:target="_blank"}

 **💻 코드:** [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb){:target="_blank"}

 **🧠 참고자료 (References):** 
1. **Repo:** [https://github.com/Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion){:target="_blank"}
2. **Blog:** [https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1){:target="_blank"}
3. **RRF:** [https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf){:target="_blank"}


## **Part 7. Query Translation: Decomposition**

이번 영상은 Least-to-Most prompting (@denny_zhou 외)와 IR-CoT (Trivedi 외)의 아이디어를 활용한 decomposition 접근법에 초점을 맞춘 Query Translation에 관한 세 번째 영상입니다.

**🔧 문제(Problem):**  복잡한 질문이 들어오게 되면, 이와 관련된 충분한 배경지식이나 전제 조건이 모두 파악되지 않은 상태에서 검색이 이루어질 수 있기 때문에, 한 번의 retrieval 단계로는 이 복잡한 질문이 제대로 해결되지 않을 가능성이 커집니다.

**💡 아이디어(Idea)**: 복잡한 질문을 더 작고 다루기 쉬운 하위 문제나 하위 질문으로 분해하는 것이 해결책이 될 수 있습니다. 질문을 일련의 하위 문제(sub-problems)나 하위 질문(sub-questions)로 분해하여 순차적으로 해결하거나(첫번째 하위 질문과 이에 대한 검색 결과로부터 나온 답변을 얻고, 이를 활용하여 두번째 하위 질문을 해결하는 식) 혹은 병렬적(각 하위 질문에 대한 개별적 답변들을 종합하여 최종 답변을 얻는 식)으로 해결할 수 있습니다. 이러한 접근 방식은 'Least-to-Most 프롬프팅'(@denny_zhou 등), 'IR-CoT'(Trivedi 등)와 같은 연구들에서 제안되었으며, 복잡한 질문 처리에 활용될 수 있습니다.

**📽️ 동영상**: [https://www.youtube.com/watch?v=h0OPWlEOank](https://www.youtube.com/watch?v=h0OPWlEOank){:target="_blank"}

 **💻 코드:** [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb){:target="_blank"}

 **🧠 참고자료 (References):** 
1. **Least-to-most:** [https://arxiv.org/pdf/2205.10625.pdf](https://arxiv.org/pdf/2205.10625.pdf){:target="_blank"}
2. **IR-CoT:** [https://arxiv.org/pdf/2212.10509.pdf](https://arxiv.org/pdf/2212.10509.pdf){:target="_blank"}
3. **@cohere for query expansion courtesy of @virattt:** [https://gist.github.com/virattt/9099bf1a32ff2b99383b2fabba0ae763](https://gist.github.com/virattt/9099bf1a32ff2b99383b2fabba0ae763){:target="_blank"}


## **Part 8. Query Translation: Step-Back Prompting**

이번 영상은 DeepMind의 @denny_zhou 그룹의 step-back prompting에 초점을 맞춘 Query Translation에 관한 네 번째 영상입니다.

**🔧 문제(Problem):**  복잡한 사용자 질문 중에는, 단순한 사실 관계를 넘어 좀 더 추상적이고 개념적인 이해를 필요로 하는 경우가 있습니다. 예를 들어 "인공지능의 발전이 가져올 사회적 변화는 무엇일까요?"라는 질문에 잘 답하려면, 인공지능의 정의, 응용 분야, 발전 방향 등 관련 개념에 대한 포괄적인 이해가 필요합니다. 또한 기술 발전이 사회에 미치는 영향을 분석하기 위한 일반적인 원리나 방법론에 대한 지식도 요구됩니다. 이처럼 어려운 질문에 효과적으로 답하기 위해서는, 해당 주제와 관련된 상위 개념과 근본 원리에 대한 이해가 뒷받침되어야 합니다.

**💡 아이디어(Idea)**: 먼저 LLM에게 주어진 질문 내용과 관련된 상위 개념이나 원리에 대한 일반적인 step-back 질문을 하도록 프롬프트하고, 이에 관련된 사실들(facts)을 검색합니다. 이렇게 수집한 기반 지식을 활용하여 사용자의 질문에 답변할 수 있습니다. 여기서 step-back이란 주어진 질문에서 한 발 떨어져서, 그 질문과 관련된 더 넓은 맥락이나 상위 개념을 고려하는 것을 의미합니다. 이렇게 "step-back"하여 상위 개념을 이해하면, 원래의 질문에 대해 더 포괄적이고 통찰력 있는 답변을 제공할 수 있다는 것이 "step-back prompting"의 기본 아이디어입니다. 이는 문제를 더 넓은 시각에서 바라보고 접근하는 방식이라고 할 수 있겠습니다. 

**📽️ 동영상**: [https://www.youtube.com/watch?v=xn1jEjRyJ2U](https://www.youtube.com/watch?v=xn1jEjRyJ2U){:target="_blank"}

 **💻 코드:** [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb){:target="_blank"}

 **🧠 참고자료 (References):**  
**Step-back:** [https://arxiv.org/pdf/2310.06117.pdf](https://arxiv.org/pdf/2310.06117.pdf){:target="_blank"}


## **Part 9. Query Translation: HyDE**

Query Translation에 관한 마지막 영상으로, Gao 등(Gao _et al._)이 제안한 가상의 문서 임베딩(Hypothetical Document Embeddings, HyDE) 기법에 초점을 맞춰 봅니다.

**🔧 문제(Problem):**  일반적으로 문서와 사용자 질문은 그 형식과 내용 면에서 큰 차이를 보입니다. 그럼에도 불구하고 많은 검색 기법들은 유사도 검색을 위해 문서와 질문을 동일한 임베딩 공간에 투영합니다. 이는 문서와 질문 간의 직접적인 유사도 비교가 항상 최선의 결과를 보장하지는 않을 수 있음을 시사합니다.

**💡 아이디어(Idea)**: 이러한 문제를 해결하기 위해, HyDE는 대형 언어 모델(LLM)을 활용하여 사용자 질문을 가상의 문서로 변환하는 아이디어를 제안합니다. 이 가상 문서는 마치 질문에 대한 답변을 포함하고 있는 것처럼 생성됩니다. 그 다음, 이렇게 생성된 가상 문서를 임베딩하여 실제 문서와의 유사도를 비교합니다. 이는 문서-문서 간 유사도 검색(doc-doc similarity search)이 문서-질문 간 유사도 검색(query-doc similarity search)보다 더 관련성 높은 결과를 얻을 수 있다는 전제에 기반한 것입니다.

**📽️ 동영상**: [https://www.youtube.com/watch?v=SaDzIVkYqyY](https://www.youtube.com/watch?v=SaDzIVkYqyY){:target="_blank"}

 **💻 코드:** [https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb){:target="_blank"}

**🧠 참고자료 (References):**   
**Hyde:** [https://arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496){:target="_blank"}
