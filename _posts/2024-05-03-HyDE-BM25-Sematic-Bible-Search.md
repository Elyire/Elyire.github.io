---
title: 간단히 구현해 보는 성경 의미 검색(Semantic Bible Search) 
date: 2024-05-03 17:00:00 +09:00
categories: [LLM, LangChain]
tags: [RAG, 의미 검색, 성경 검색]
pin: false
---
이번 포스팅에서는 RAG 기법을 이용하여 간단히 성경 의미 검색(Semantic Bible Search)을 구현해 보고자 합니다. 성경 검색은 일반적으로 두 가지 방식으로 이루어지는데, 하나는 성경의 특정 책, 장, 절을 직접 입력하여 해당 구절을 찾는 것이고, 다른 하나는 키워드를 입력하여 그 단어가 포함된 모든 구절을 검색하는 것입니다. 

반면, 이번 포스팅에서 다루고자 하는 성경 의미 검색(Semantic Bible Search)은 자연어로 원하는 구절을 찾는 방식입니다. 예를 들어 "사랑이 어떻게 행동하는지 설명하는 구절은 어디에 있나요?"와 같은 질문을 통해 관련 구절을 검색하는 것이죠. 이러한 방식은 기존의 키워드 검색과는 달리, 시스템이 사용자 질문의 의미를 파악함으로써 적절한 구절을 찾아주게 됩니다. 이를 통해 사용자는 보다 직관적이고 편리하게 원하는 성경 구절을 찾을 수 있습니다. 

## **LLM을 이용한 예비 검색결과 생성**

본 구현에서는 기존의 검색 기법과는 다른 접근 방식을 취했는데, 이는 HyDE(Hypothetical document embeddings)[1]에서 영감을 받았습니다. HyDE는 사용자 질문에 대한 가상의 답변을 생성하고, 이 가상 답변과 실제 문서 간의 유사도를 계산하여 관련 문서를 검색하는 기법입니다. 이번 성경 의미 검색(Semantic Bible Search) 구현에서는 HyDE의 개념 중 '사용자 질문에 대한 가상의 답변 생성'이라는 아이디어를 차용하였습니다. 

많은 테스트 결과, GPT와 같은 대형언어모델(LLM)들이 성경과 관련된 방대한 지식을 학습하고 있으며, 이에 대해 높은 정확도로 정보를 제공한다는 것을 발견할 수 있었습니다. 여기서는 이 점에 착안하여, 사용자의 자연어 질문에 대응하는 성경 구절을 LLM을 통해 예비적으로 생성하고 이를 활용하는 방식을 채택했습니다. 이는 마치 HyDE에서 사용자 질문에 대한 가상의 답변을 미리 생성하는 것과 유사한 접근 방식이라고 볼 수 있습니다. 

다만, HyDE에서는 가상 답변을 임베딩하여 이에 대응하는 유사한 문서를 검색하는 반면, 본 구현에서는 생성된 예비 검색결과에 대해 벡터 임베딩을 통한 유사도 계산을 사용하지 않았습니다(이 부분에 대해서는 뒤에서 좀 더 자세히 말씀드리도록 하겠습니다). 

```python
# 사용자 질문
question = "사랑이 어떻게 행동하는지 설명하는 구절은 어디에 있나요?"

# 예비 검색결과 생성을 위한 프롬프트 템플릿
template = """
Please list up the three Bible verses that are most relevant to the question **in Korean**, along with the verses immediately preceding and following them.  
Question: {question}

# Output format
- 성경구절과 이에 대응하는 말씀을 출력합니다.
- 성경 책 이름은 약어로 표기합니다 (요한복음 -> 요)
<example>
요1:1 태초에 말씀이 계시니라
롬1:17 복음에는 하나님의 의가 나타나서 믿음으로 믿음에 이르게 하나니 기록된 바 오직 의인은 믿음으로 말미암아 살리라 함과 같으니라
</example>

Verses:
"""

prompt_hyde = ChatPromptTemplate.from_template(template)
```

```python
llm_01 = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

# 예비 검색결과 생성을 위한 chain
generate_prelim_docs = (
    prompt_hyde 
    | llm_01 
    | StrOutputParser()
)

# 사용자 질문에 대한 예비 검색결과 생성
PrelimDocs = generate_prelim_docs.invoke({"question": question})
```

### 생성된 예비 검색결과
```
고전13:4 사랑은 오래 참고 사랑은 온유하며 투기하는 자가 되지 아니하며 사랑은 부풀지 아니하며
고전13:5 무례히 행치 아니하며 구하지 아니하며 성내지 아니하며 악한 것을 생각지 아니하며
고전13:6 불의를 기뻐하지 아니하며 진리와 함께 기뻐하고
고전13:7 모든 것을 참으며 모든 것을 믿으며 모든 것을 바라며 모든 것을 견디느니라

요일3:16 그가 우리를 위하여 목숨을 버리셨으니 우리가 이로써 사랑을 알고
요일3:17 누구든지 이 세상의 재물이 있고 형제의 궁핍함을 보고도 도와 줄 마음을 베풀지 아니하면 하나님의 사랑이 어찌 그 속에 거할까 보냐
요일3:18 자녀들아 우리가 말과 혀로만 사랑하지 말고 오직 행함과 진실함으로 하자

요일4:7 사랑하는 자들아 우리가 서로 사랑하자 사랑은 하나님께로부터 난 것이며 사랑하는 자마다 하나님께로부터 나서 하나님을 알고
요일4:8 사랑하지 아니하는 자는 하나님을 알지 못하나니 이는 하나님이 사랑이심이라
요일4:9 하나님의 사랑이 우리에게 이렇게 나타났으니 하나님이 자기의 독생자를 세상에 보내심은 그로 말미암아 우리를 살리려 하심이라
```

바로 위의 출력 블록은 `generate_prelim_docs` 체인을 `invoke`한 결과로, `"사랑이 어떻게 행동하는지 설명하는 구절은 어디에 있나요?"`라는 사용자 질문에 대해 LLM이 적절한 예비 검색결과를 생성한 것입니다. 

이를 보면 "그러면 성경 의미 검색은 그냥 LLM에게 관련 성경구절을 질의하고 답을 얻으면 되는 거 아닌가?"라는 의문이 들 수 있습니다. 하지만 단순히 LLM에게 "~와 관련된 성경구절이 뭐야?"라고 물어보고 얻은 답변을 성경 의미 검색(Semantic Bible Search)의 최종 결과로 사용하기에는 한 가지 문제점이 있습니다. 바로 LLM이 생성한 응답이 해당 구절의 정확한 한국어 번역을 제공하지 못한다는 점입니다. 

LLM의 응답 생성은 확률에 의존하기 때문에, 성경번역본과 유사하게 진행되다가도 어느 순간 다른 부분이 나타나게 됩니다. 경험상, 유명하고 널리 알려진 구절은 정확하게 출력하는 경우가 많지만, 그렇지 않은 구절들에서는 성경번역본과 유사하나, 완전히 일치하지 않는 경우가 자주 관찰됩니다. 다음 예시를 통해 이를 확인할 수 있습니다. 

> **고린도전서 13:4 (성경번역본)** 사랑은 오래 참고 사랑은 온유하며 <span style="color:blue">시기하지 아니하며 사랑은 자랑하지 아니하며 교만하지 아니하며</span>
>
>**고린도전서 13:4 (claude-3-sonnet)** 사랑은 오래 참고 사랑은 온유하며 <span style="color:blue">투기하는 자가 되지 아니하며 사랑은 부풀지 아니하며</span>
>
>**요한1서 3:17 (성경번역본)** <span style="color:blue">누가 이 세상의 재물을 가지고 형제의 궁핍함을 보고도 도와 줄 마음을 닫으면</span> 하나님의 사랑이 어찌 그 속에 <span style="color:blue">거하겠느냐</span>
>
>**요한1서 3:17 (claude-3-sonnet)** <span style="color:blue">누구든지 이 세상의 재물이 있고 형제의 궁핍함을 보고도 도와 줄 마음을 베풀지 아니하면</span> 하나님의 사랑이 어찌 그 속에 <span style="color:blue">거할까 보냐</span>

한글 성경에는 다양한 번역본이 존재합니다. 추측컨대, 이런 다양한 번역본들이 LLM의 학습데이터로 사용되었고, 이를 기반으로 확률적으로 출력하는 과정에서 선택되는 토큰들이 서로 뒤섞여 이런 현상이 발생하는 것이 아닐까 합니다. 

어차피 번역본이기 때문에 대략적인 의미만 유사하면 충분하다고 여길수도 있습니다. 그렇지만 원문의 의미를 최대한 반영하기 위해 번역 과정에서는 단어 선택, 구문 구조 배치 등이 신중히 고려되며, 여러 차례의 교정과 감수를 거쳐 최종 번역본이 탄생하게 됩니다. 따라서 LLM 출력에서 단어가 바뀌거나 구문 구조의 순서가 변경되는 것은 미묘한 의미 상의 왜곡을 초래할 수 있기 때문에 결코 가볍게 여길 부분은 아니라고 생각합니다.

즉, 성경 의미 검색(Semantic Bible Search)은 사용자의 자연어 질문을 받아 관련된 구절을 검색하고, 현재 한국교회에서 사용하고 있는 성경번역본의 정확한 번역 구절을 반환하는 것을 목표로 한다고 할 수 있습니다. 이를 위해서는 LLM 예비 검색결과를 통해 찾아낸 구절에 정확히 매칭이 되는 구절들을 성경 문서 청크들로부터 찾아와서 최종 답변 생성을 위한 맥락(context)으로 사용해야 합니다.  

## **BM25 retriever를 통한 context retrieval**

성경 의미 검색을 구현하는 과정에서, LLM 예비 검색결과에서 언급된 구절들을 활용하여 성경 문서 청크들 중 가장 관련성이 높은 구절들을 찾아내기 위해 가장 먼저, HyDE, 즉, 예비 검색결과를 임베딩하여 성경 문서 청크들 중에서 이에 대응하는 유사한 문서를 검색하는 방법을 시도했습니다. 

그러나 이 과정을 거치면 오히려 종종 사용자 질문에 대응하지 않는 엉뚱한 구절들을 포함하여 검색해오는 것을 확인할 수 있었습니다. LLM이 예비 검색결과 단계에서 질문에 잘 맞는 구절들을 찾아냈음에도 불구하고, 임베딩 벡터 유사도 계산을 통한 문서 검색 과정에서 오히려 관련성이 떨어지는 결과가 나타나는 현상이 관찰되었습니다. 

이는 예비 검색결과를 임베딩하고 이를 다시 성경 문서 청크의 임베딩과 매칭하는 과정에서 불필요한 복잡성이 야기되어 오히려 검색 품질이 저하되었기 때문일 것입니다. 예비 검색결과 자체가 이미 사용자 질의와 관련성이 높은 구절들을 포함하고 있는데, 이를 다시 임베딩하여 유사도 검색을 수행하는 것은 정보의 손실을 초래할 수 있기 때문입니다. 

그래서 이번에는 임베딩 벡터 기반 유사도 검색 대신, 전통적인 키워드 기반 검색 시스템에서 널리 사용되어 온 BM25 검색기를 적용하기로 했습니다. 즉, 예비 검색결과의 구절들을 성경 청크 문서들로 구성된 BM25 검색기에 쿼리로 전달해 이들과 가장 관련성이 높은 성경 구절들을 검색하는 방식을 채택하였습니다.

```python
# 성경 텍스트 파일 청킹 및 리스트화(BM25 retriever 만들기 위해)
loader = TextLoader(file_path="NKRV.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
split_docs = splitter.split_documents(docs)
```

```python
# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 3

# Retrieve
retrieval_chain = generate_prelim_docs | bm25_retriever

# 예비 검색결과에 대해 BM25 retriever 적용
retrieved_docs = retrieval_chain.invoke({"question": question}) 
```

사실 예비 검색결과에 기재되어 있는 성경구절 정보(책, 장, 절; 예: 요한복음 1장 1절)를 이용하여 성경 텍스트 파일에서 해당 구절을 검색해 그대로 가져오는 방법도 있습니다. 그럼에도 이러한 방법 대신 BM25 retriever를 적용한 이유는 구절 정보의 불완전성 혹은 오류에 유연하게 대처하기 위해서 입니다. 즉, LLM이 생성한 예비 검색결과에서 항상 정확하고 완전한 구절 정보를 출력하지 못할 가능성이 있고, 이런 상황이 벌어지더라도 구절 내용(구절 정보에 대응하는 구절 내용; 예: 태초에 말씀이 계시니라...)의 유사성을 기반으로 관련 구절들을 찾아낼 것을 기대하면서, 일종의 보완책으로 BM25 retriever를 사용한 것입니다.

예비 검색결과에 BM25 retriever를 적용하여 가져온 retrieved documents는 아래와 같습니다. 보시는 바와 같이 LLM이 생성한 예비 검색결과 구절 뿐만 아니라 이것의 전, 후 구절(예: 고린도전서 13:2, 요한일서 3:19-21, 요한일서 4:10-12)까지 함께 검색해서 가져온 것을 알 수 있습니다. 구절 내용의 유사성을 기반으로 이렇게 추가적으로 가져온 문서 정보는 추후, 생성 단계에서 LLM이 보다 풍부한 맥락을 바탕으로 답변을 생성하는 데 활용될 수 있습니다. 이를 통해 사용자에게 더 나은 검색 경험과 관련 정보를 제공할 수 있을 것으로 기대됩니다. 

### ***Retrieved documents***
```
[Document(page_content='고전13:2 내가 예언하는 능력이 있어 모든 비밀과 모든 지식을 알고 또 산을 옮길 
만한 모든 믿음이 있을지라도 사랑이 없으면 내가 아무 것도 아니요\n고전13:3 내가 내게 있는 모든 것으로 
구제하고 또 내 몸을 불사르게 내어 줄지라도 사랑이 없으면 내게 아무 유익이 없느니라\n고전13:4 사랑은 
오래 참고 사랑은 온유하며 시기하지 아니하며 사랑은 자랑하지 아니하며 교만하지 아니하며\n고전13:5 무례히 
행하지 아니하며 자기의 유익을 구하지 아니하며 성내지 아니하며 악한 것을 생각하지 아니하며\n고전13:6 불의를 
기뻐하지 아니하며 진리와 함께 기뻐하고\n고전13:7 모든 것을 참으며 모든 것을 믿으며 모든 것을 바라며 모든 
것을 견디느니라', metadata={'source': 'NKRV.txt'}),
Document(page_content='요일3:16 그가 우리를 위하여 목숨을 버리셨으니 우리가 이로써 사랑을 알고 우리도 
형제들을 위하여 목숨을 버리는 것이 마땅하니라\n요일3:17 누가 이 세상의 재물을 가지고 형제의 궁핍함을 보고도 
도와 줄 마음을 닫으면 하나님의 사랑이 어찌 그 속에 거하겠느냐\n요일3:18 자녀들아 우리가 말과 혀로만 사랑하지 
말고 행함과 진실함으로 하자\n요일3:19 이로써 우리가 진리에 속한 줄을 알고 또 우리 마음을 주 앞에서 굳세게 
하리니\n요일3:20 이는 우리 마음이 혹 우리를 책망할 일이 있어도 하나님은 우리 마음보다 크시고 모든 것을 아시기 
때문이라\n요일3:21 사랑하는 자들아 만일 우리 마음이 우리를 책망할 것이 없으면 하나님 앞에서 담대함을 얻고', 
metadata={'source': 'NKRV.txt'}),
Document(page_content='요일4:8 사랑하지 아니하는 자는 하나님을 알지 못하나니 이는 하나님은 사랑이심이라\n
요일4:9 하나님의 사랑이 우리에게 이렇게 나타난 바 되었으니 하나님이 자기의 독생자를 세상에 보내심은 그로 
말미암아 우리를 살리려 하심이라\n요일4:10 사랑은 여기 있으니 우리가 하나님을 사랑한 것이 아니요 하나님이 
우리를 사랑하사 우리 죄를 속하기 위하여 화목 제물로 그 아들을 보내셨음이라\n요일4:11 사랑하는 자들아 하나님이 
이같이 우리를 사랑하셨은즉 우리도 서로 사랑하는 것이 마땅하도다\n요일4:12 어느 때나 하나님을 본 사람이 없으되 
만일 우리가 서로 사랑하면 하나님이 우리 안에 거하시고 그의 사랑이 우리 안에 온전히 이루어지느니라',
metadata={'source': 'NKRV.txt'})]
```

## **성경 의미 검색(Semantic Bible Search) 결과**

이제 retrieved documents를 맥락(context)으로 활용하여 성경 의미 검색(Semantic Bible Search)의 최종 결과를 생성할 단계입니다. 아래는 검색 결과 생성에 사용된 프롬프트와 체인입니다.

프롬프트 템플릿의 `{context}` 부분에는 retrieved documents가 삽입되며, 프롬프트의 마지막에 `{question}`을 배치하여 LLM이 최종 답변(검색 결과)을 생성할 때 사용자의 질문 내용을 다시 한 번 상기할 수 있도록 했습니다. 생성된 결과는 사용자 질문과 가장 연관성이 높은 5개의 성경 구절을 출력하도록 했습니다. 이를 통해 사용자는 자신의 질문에 대한 가장 적절한 성경 구절이 무엇인지를 확인할 수 있게 됩니다.

```python
# 검색결과 생성을 위한 프롬프트 템플릿
template = """
# Instruction
Your task is to list up the top five, most relevant Korean Bible verses
related to the question based on the following context:
{context}

# Constraint
- When you list up the top five, most relevant Korean Bible verses,
you should quote the top five, most relevant verse in context, **exactly and verbatim**, without missing a single Korean word. 
- Be sure to get the contents of the context **verbatim** - don't make it up.
- If you can't find all the top five in your context, you can just output as many as you can. 
- Check to see if the question specifies the words Old Testament or New Testament, and even if it doesn't, figure out whether the context of the question wants you to look up Bible verses from the Old Testament, New Testament, or both, and generate an answer that responds accordingly.  

# Output format
- In the output, the numbering should follows the order of Bible chapters and verses.
- Don't explain, just output the numbered Bible verses.
- 성경구절과 이에 대응하는 말씀을 출력합니다.
- 성경 책 이름은 약어로 표기합니다 (요한복음 -> 요)
<example>
요1:1 태초에 말씀이 계시니라
롬1:17 복음에는 하나님의 의가 나타나서 믿음으로 믿음에 이르게 하나니 기록된 바 오직 의인은 믿음으로 말미암아 살리라 함과 같으니라
</example>

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm_02 = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)

# Semantic Bible Search 결과 출력을 위한 chain
final_rag_chain = (
    prompt
    | llm_02
    | StrOutputParser()
)
```

```python
results = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
```

### ***검색 결과***
```
1. 고전13:4 사랑은 오래 참고 사랑은 온유하며 시기하지 아니하며 사랑은 자랑하지 아니하며 교만하지 아니하며
2. 고전13:5 무례히 행하지 아니하며 자기의 유익을 구하지 아니하며 성내지 아니하며 악한 것을 생각하지 아니하며
3. 고전13:6 불의를 기뻐하지 아니하며 진리와 함께 기뻐하고
4. 고전13:7 모든 것을 참으며 모든 것을 믿으며 모든 것을 바라며 모든 것을 견디느니라
```

"사랑이 어떻게 행동하는지 설명하는 구절은 어디에 있나요?"라는 질문을 통해 찾고 싶었던 구절들이 잘 출력되었음을 확인할 수 있습니다. 

## **Summary**

이번 포스팅에서는 "HyDE(Hypothetical document embeddings)의 개념 중 '사용자 질문에 대한 가상의 답변 생성'이라는 아이디어"와 "BM25 retriever"를 활용해 retrieved documents를 얻고, 이를 맥락(context)으로 하여 최종 답변을 생성함으로써 성경 의미 검색(Semantic Bible Search, 자연어 질문을 통한 성경 구절 검색)을 구현하는 방법을 알아보았습니다.

혹시 관심이 있으신 분은 다음 링크에서 구현된 성경 의미 검색(Semantic Bible Search)을 사용해 보실 수 있습니다. 

👉 [**https://semantic-bible-search.streamlit.app/**](https://semantic-bible-search.streamlit.app/){:target="_blank"} 📖


## **References**

[1] L. Gao *et al.*, "Precise Zero-Shot Dense Retrieval without Relevance Labels", _arXiv_ **2022**, 2212.10496


