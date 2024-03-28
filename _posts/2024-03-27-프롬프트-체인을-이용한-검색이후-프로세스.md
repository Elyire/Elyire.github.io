---
title: 프롬프트 체인(Prompt Chain)을 이용한 검색이후(Post-Retrieval) 프로세스
date: 2024-03-28 17:20:00 +09:00
categories: [프롬프트 엔지니어링, 프롬프트 작성기법]
tags: [RAG, 프롬프트 작성기법, 랭체인]
pin: false
---

## 프롬프트 체인(Prompt Chain)

프롬프트 체인(prompt chain 혹은 prompt chaining)은 task 수행을 위해 2개 이상의 프롬프트를 사용하는 접근법으로서, 이전 프롬프트 지시(instruction)에 의한 출력 결과를 다음 프롬프트 지시의 내용의 일부로 포함시켜 사용하는 것이 핵심이다. 

예를 들어, LLM이 주어진 아티클을 검사하여 문법 오류 목록을 작성하고, 이 목록이 빠짐없이 작성되었는지는 확인하는 프로세스에 프롬프트 체인을 활용할 수 있다.[1]

첫번째 프롬프트는 이용해 문법 오류 목록을 생성하도록 지시한다.

```
Here is an article:
<article>
{ARTICLE}
</article>

Please identify any grammatical errors in the article. 
Please only respond with the list of errors, and nothing else. 
If there are no grammatical errors, say "There are no errors."
```

첫번째 프롬프트로부터 생성된 문법 오류 목록을 {ERRORS}라고 한다면, 두 번째 프롬프트에 이를 추가하여 문법 오류 목록에 빠진 내용이 없는지 확인하도록 지시한다.  

```
Here is an article:
<article>
{ARTICLE}
</article>

Please identify any grammatical errors in the article that are missing from the following list:
<list>
{ERRORS}
</list>

If there are no errors in the article that are missing from the list, say "There are no additional errors."
```


## 프롬프트 체인(Prompt Chain) 기반 검색이후(Post-Retrieval) 프로세스

효과적인 최종 답변 생성을 위해, 구축된 벡터 데이터베이스 기반 검색기(혹은 리트리버, retriever)를 이용해 쿼리 관련 문서를 가져온 후, 이 문서들에 추가적인 처리를 해주는 다양한 접근법들이 시도되어 왔다 (RankGPT[2], RAG-Fusion[3], CRAG[4] 등).  

문득 복잡한 프로그래밍 없이, 순수하게 프롬프트 엔지니어링만으로 검색이후 프로세스를 적용해보면 어떨까 하는 생각이 들었다.

1) 쿼리와 유사한 문서들을 기존의 임베딩 벡터를 이용한 유사도 검색을 통해 가져온다.  
2) LLM을 이용해(즉, 프롬프트를 이용해) retrieved docs 중, 쿼리와 깊은 관련이 있는 문장이나 단락을 리스트업하고 순위를 매긴다.  
3) 순위가 매겨진 관련 문장 혹은 단락 리스트를 이용해 최종답변을 LLM이 생성하게 한다.  

즉, 쿼리에 대한 최종답변에 필요한 내용을 가져오기 위해 시맨틱 검색과 LLM의 능력을 모두 사용하는 것이다. 만약 내가 어떤 retrieved documents를 누군가로부터 받아서,  주어진 질문에 대한 최종 답변을 만들어야 한다면 어떤 프로세스를 거칠까를 생각해 봤다. 나라면 먼저 내게 주어진 retrieved documents(아마도 서류 뭉탱이일수도 있겠다)로부터 질문 답변에 필요한 문장이나 문단을 골라 따로 적어놓고, 이렇게 선별된 문장이나 문단을 활용해서 질문에 대한 최종 답변을 작성할 것 같다는 생각이 들었다. 

위의 아이디어는 이 과정을 그대로 RAG에 적용한 것이다. Basic RAG에서는 LLM이 retrieved documents로부터 바로 최종답변을 생성하게 되는데, 이 방법은 **프롬프트 체인 개념을 활용해 LLM으로 하여금 한 번 더 정리하는 시간을 갖게하는 것**이다. 정말 단순한 접근법이라 이미 제안되었을 것이 분명한데, 관련 내용을 아직 찾지는 못했다(너무 단순한 접근이라 아예 언급 자체가 안 되는 것인가..제보해 주시면 감사하겠습니다🙂).

LLM이 retrieved documents로부터 곧바로 최종 답변을 생성하는 Basic RAG의 결과와 비교했을 때, 동일한 답변 길이에서 좀 더 자연스러운 답변을 출력함을 확인할 수 있었다. 필자가 구현 중인 종교서/철학서와 관련한 RAG 시스템에서의 효과는 이런데, 다른 분야의 RAG에서는 어떤 효과가 있는지도 궁금하다.  

```python
from operator import itemgetter
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic

# Embedding model
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {"normalize_embeddings": True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Loading vectorstore with Semantic-chunked data
vectorstore = FAISS.load_local(
    folder_path="C:\...",
    embeddings=embedding,
    index_name='index name'
    )

retriever = vectorstore.as_retriever(search_kwargs={"k":8})

question = "스토아 학파는 인간이 고통에 어떻게 대처해야 한다고 했나요?"
retrieved_docs = retriever.get_relevant_documents(question)

# Generation
llm = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")
```

```python
template_1 = """

# Instruction

Given the following retrieved documents and the query, 
extract sentences or passages from the documents that are most relevant to answering the query. 
Then, provide **only ranked list of the top relevant sentences or passages**. 
Note that retrieved documents are delimited by XML tag of <retrieved_documents></retrieved_documents>.


# Query: {question}

<retrieved_documents>
# Retrieved Documents:
{context}
</retrieved_documents>


# Constraint:
1. Carefully read through each retrieved document and identify sentences or passages that are most relevant to answering the given query.
2. Extract the relevant sentences or passages and provide them as a numbered list.
3. Rank the extracted sentences or passages based on their relevance and potential to directly answer the query.
4. If there are no relevant sentences or passages found in the retrieved documents, indicate that no satisfactory answer was found.

# Output
Ranked list of the top relevant sentences or passages:
"""


prompt_1 = ChatPromptTemplate.from_template(template_1)
```

```python
 chain1 = (
    {"question": itemgetter("question"), "context": itemgetter("context")}
    | prompt_1
    | llm
    | StrOutputParser()
    )
```

```python
list_of_relevant_contents = chain1.invoke({"question": question, "context": retrieved_docs})

# list_of_relevant_contents
"""
1. 스토아학파는 우주에는 '로고스'라는 신성한 이성 구조가 존재한다고 믿었으며, 이 로고스에 순응하는 것이 행복한 삶을 살 수 있는 방법이라고 가르쳤다.
2. 스토아학파는 감성보다 이성에 절대 우위를 두며, 고난을 견딜 수 없게 만드는 압도적인 고통이 집착에서 오므로 삶에서 마주치는 그 무엇에도 지나치게 애착을 갖지 않는 법을 배워야 한다고 주장했다.
3. 스토아학파는 죽음이 한 상태에서 다른 상태로 탈바꿈하는 일일 뿐이라고 가르쳤으며, 죽음에 대한 두려움에서 벗어나 현재를 사랑하며 살아가는 것이 중요하다고 강조했다.
"""
```

```python
template_2 = """
   
# Instruction
Using the provided ranked list of the top relevant sentences or passages extracted from the retrieved documents, 
generate a final answer to the given query. 
Synthesize the information from the ranked list of the top relevant sentences or passages to create a coherent, logical, comprehensive, and concise response.

# Query: {question}

Ranked list of the top relevant sentences or passages:
{list_of_relevant_contents}

# Constraint:
1. Carefully review the provided list of the top relevant sentences or passages and identify the key information relevant to the query.
2. Synthesize the information from the ranked list of the top relevant sentences or passages to generate a final answer that directly addresses the query.
3. You DO NOT need to use all the information from the ranked list of the top relevant sentences or passages when generating a final answer.
4. Ensure that the final answer is coherent, logical, comprehensive, and concise. Rule out sentences or passages that make your final answer less than coherent and logical. 
5. If the ranked list of the top relevant sentences or passages do not provide sufficient information to answer the query satisfactorily, indicate that more information may be needed.
6. Use your own words to create a natural and fluent response, while maintaining the accuracy of the information provided in the ranked list of the top relevant sentences or passages.

# Output
Final Answer:
"""


prompt_2 = ChatPromptTemplate.from_template(template_2)
```

```python
chain2 = (
    {"question": itemgetter("question"), "list_of_relevant_contents": itemgetter("list_of_relevant_contents")}
    | prompt_2
    | llm
    | StrOutputParser()
    )
```

```python
final_answer = chain2.invoke({"question":question, "list_of_relevant_contents":list_of_relevant_contents})

# final_answer
"""
"스토아 학파는 인간이 고통에 대처하는 방법으로 다음과 같은 견해를 제시했습니다.

첫째, 스토아 학파는 우주에는 '로고스'라는 신성한 이성 구조가 존재한다고 믿었으며, 이 로고스에 순응하는 것이 행복한 삶을 살 수 있는 방법이라고 가르쳤습니다. 
즉, 감성보다 이성에 절대 우위를 두며, 고난을 견딜 수 없게 만드는 압도적인 고통이 집착에서 오므로 삶에서 마주치는 그 무엇에도 지나치게 애착을 갖지 않는 법을 배워야 한다고 주장했습니다.

둘째, 스토아 학파는 죽음이 한 상태에서 다른 상태로 탈바꿈하는 일일 뿐이라고 가르쳤으며, 죽음에 대한 두려움에서 벗어나 현재를 사랑하며 살아가는 것이 중요하다고 강조했습니다.

따라서 스토아 학파는 인간이 고통에 대처하기 위해서는 이성에 의해 지배되어야 하며, 삶에 대한 집착을 버리고 현재에 충실하게 살아가는 것이 중요하다고 주장했습니다."
"""
```


### cf) Basic RAG 결과

```
# Your role
You are a brilliant assistant for question-answering tasks.

# Instruction
Your task is to answer the question using the following pieces of retrieved context.
When you generate an answer, follow the steps in order.
1. Think deeply and multiple times about the user's question\nUser's question:\n{question}\nYou must understand the intent of their question and provide the most appropriate answer.
2. Choose the most relevant content from the retrieved context that addresses the user's question and use it to generate an answer.

Retrieved Context:
{context}


# Constraint
- Each sentence that is generated should be well-connected and logical.
- If you don't know the answer, just say that you don't know.
- Use five sentences maximum. Keep the answer concise but logical/natural/in-depth.
-  **Answer in Korean.**

Question:
{question}
```

```
'스토아 학파는 인간이 고통에 대처하는 방법으로 세 가지를 제시했습니다. 

첫째, 예상치 못한 고난과 역경을 신의 섭리이자 결국 유익이 되는 역사로 받아들이는 자세를 가져야 한다고 했습니다. 
스토아 학파는 우주가 신성하고 이성적이며 완벽한 질서를 유지하고 있다고 믿었기 때문에, 세상이 보내주는 것을 전폭적으로 받아들이는 삶을 살아야 한다고 주장했습니다.

둘째, 감성보다 이성에 절대 우위를 두며, 고난을 견딜 수 없게 만드는 압도적인 고통이 집착에서 오므로 삶에서 마주치는 그 무엇에도 지나치게 애착을 갖지 않는 법을 배워야 한다고 했습니다.

셋째, 죽음은 한 상태에서 다른 상태로 탈바꿈하는 일일 뿐이라는 스토아 사상을 통해, 죽음에 대한 두려움을 극복하고 구원과 비슷한 경지에 이르게 된다고 주장했습니다.'
```


## References

[1] Anthropic. n.d. ["Chain prompts." In Prompt Engineering.](https://docs.anthropic.com/claude/docs/chain-prompts) Accessed March 28, 2024. https://docs.anthropic.com/claude/docs/chain-prompts{:target="_blank"}  
[2] Arjun, [Improving RAG: using LLMs as reranking agents](https://medium.com/@arjunkmrm/improving-rag-using-llms-as-re-ranking-agents-a6c66839dee5){:target="_blank"} Medium, 2024, https://medium.com/@arjunkmrm/improving-rag-using-llms-as-re-ranking-agents-a6c66839dee5{:target="_blank"}  
[3] L. Martin, [RAG from scratch: Part 6. Query Translation (RAG-Fusion)](https://www.youtube.com/watch?v=77qELPbNgxA&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&index=6){:target="_blank"}  
[4] B. Ghosh, [Blueprint for Building Corrective RAG (CRAG)](https://medium.com/@bijit211987/blueprint-for-building-corrective-rag-crag-d6fbfeb7c98e){:target="_blank"} Medium, 2024, https://medium.com/@bijit211987/blueprint-for-building-corrective-rag-crag-d6fbfeb7c98e{:target="_blank"}  