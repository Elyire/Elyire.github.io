---
title: 랭체인 LangChain Expression Language(LCEL)과 runnable
date: 2024-03-24 15:03:00 +09:00
categories: [LLM, LangChain]
tags: [LangChain, 랭체인]
pin: false
---


## **LangChain Expression Language(LCEL)과 runnable(러너블) 요소**

LangChain(랭체인)은 GPT나 Claude와 같은 LLM(Large Language Model) 기반 애플리케이션 개발을 위한 프레임워크다. 이름에 붙어 있는 'Chain'이라는 단어에서 유추할 수 있듯이 이 프레임워크에서는 LLM 애플리케이션 개발에 필요한 구성 요소들을 다양하게 연결 혹은 엮을 수 있는 도구들을 제공한다.

최근 LangChain이 대대적으로 업데이트되면서 **LangChain Expression Language(LCEL)**라는 개념이 도입되었는데, 간단히 말해 이는 **LLM 애플리케이션 구성 요소들을 쉽고 유연하게 조합할 수 있도록 해 주는 표현식 체계**라고 할 수 있다.

예를 들어 GPT-3.5 모델을 이용하여 LLM 애플리케이션을 구현할 때 반드시 필요한 기본 구성요소를 꼽자면 1) LLM 자체(GPT-3.5 모델)와 2) LLM의 출력을 가이드하기 위한 프롬프트를 들 수 있다. LCEL을 이용하면 이 시스템을 다음과 같이 간단히 chain으로 표현할 수 있다.

```python
chain = prompt | llm
```  

이 표현식에서 파이프 기호(`|`)는 두 구성 요소가 연결되어 있다는 것을 의미한다. 첫번째 요소의 출력이 두번째 요소의 입력으로 들어가는 식으로 연결된다. 이렇게 파이프로 연결될 수 있는 구성요소를 **runnable**이라고 부른다. 쉽게 말해 runnable은 **입력을 받아 출력을 생성할 수 있는 '실행가능'한 요소**라고 할 수 있다. 

runnable은 `Runnable` 프로토콜을 따라 구현되어 있어야 하며, 기본적으로 `invoke`, `batch`, `stream` 등과 같은 메서드들을 정의하고 있어야 한다. 예를 들어, `invoke` 메서드는 입력을 받아 해당 구성 요소를 실행하고 출력을 반환하는 역할을 한다 (runnable로 구성된 chain 역시 runnable이며 이러한 메서드들을 적용할 수 있다).  

파이프 기호(`|`)를 이용한 표현방식은 runnable 구성 요소들의 chaining을 직관적으로 파악할 수 있도록 해 준다. 때문에 이는 LangChain의 기본 아이디어에 매우 잘 부합하는 설계 원칙이라고 볼 수 있다.  

## **Prompt as a runnable**

재미있는 것은 일반적으로 텍스트 덩어리인 **Prompt**를 LCEL에서는 runnable 구성요소(component)로 취급한다는 것이다. 즉, 기존 LangChain의 prompt template 개념을 이용함으로써, 텍스트 덩어리인 prompt를 입력과 출력이 있는 요소로 변신시켜 chain의 구성요소인 runnable로 활용할 수 있게 한 것이다.

다음의 예시 코드를 살펴보자. `template`는 기본적으로 문자열이지만 `{topic}`과 `{question}`이라는 두 변수를 갖는 템플릿의 형태를 갖는다. `PromptTemplate.from_template()`를 사용하여 `template`로부터 `prompt`를 생성하면, `prompt`는 `PromptTemplate`의 인스턴스가 된다.
`prompt`를 출력해보면 `PromptTemplate` 객체의 정보를 확인할 수 있는데, `input_variables`에는 템플릿에서 사용된 변수들(`['question', 'topic']`)이 나열되고, `template`에는 원래의 템플릿 문자열이 그대로 들어 있다.

```python
template = """
당신은 {topic} 분야의 전문가입니다. 이 분야에 대한 사용자의 질문에 정확한 답변을 제공하세요.
사용자 질문:
{question}
"""
prompt = PromptTemplate.from_template(template)
```

```
PromptTemplate(input_variables=['question', 'topic'], template='\n당신은 {topic} 분야의 전문가입니다. 이 분야에 대한 사용자의 질문에 정확한 답변을 제공하세요.\n사용자 질문:\n{question}\n')
```

이렇게 생성된 `prompt` 인스턴스는 `runnable`이 된다. 즉, 다음과 같이 `invoke` 메서드를 통해 입력을 지정하여 이 `prompt`라는 `runnable`을 '실행'하면 템플릿의 `{topic}`과 `{question}` 자리에 각각 "스포츠"와 "2002년 월드컵 개최국은 어디인가요?"가 들어간 최종 프롬프트가 `PromptValue`의 인스턴스 형태로 출력되어 나온다. 

```python
prompt.invoke({"topic": "스포츠", "question": "2002년 월드컵 개최국은 어디인가요?"})
```

```
StringPromptValue(text='\n당신은 스포츠 분야의 전문가입니다. 이 분야에 대한 사용자의 질문에 정확한 답변을 제공하세요.\n사용자 질문:\n2002년 월드컵 개최국은 어디인가요?\n')
```

`prompt`를 '실행'함으로써 얻은 출력은 LLM의 입력으로 전달되어 응답 생성을 지시하는 프롬프트 역할을 하게 된다. 이를 통해 이 chain은 최종적으로 우리가 원하는 LLM의 응답을 얻을 수 있게 해준다. 전체 코드는 다음과 같다. 

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

template = """
당신은 {topic} 분야의 전문가입니다. 이 분야에 대한 사용자의 질문에 정확한 답변을 제공하세요.
사용자 질문:
{question}
"""

prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

chain = prompt | llm

chain.invoke({"topic": "스포츠", "question": "2002년 월드컵 개최국은 어디인가요?"})
```

```
AIMessage(content='2002년 월드컵은 한국과 일본에서 공동으로 개최되었습니다.')
```


## **Retriever as a runnable**

LCEL에서는 RAG(Retrieval Augmented Generation, 검색증강생성)에 사용되는 **Retriever** 또한 runnable 요소이며, 이 점을 활용하면 LangChain을 이용한 RAG 구현 중간 단계에서 retriever가 입력 쿼리와 관련된 문서들을 잘 검색하여 가져오는지를 체크해 볼 수 있다. Retriever는 입력으로 사용자 쿼리인 단일 문자열을 받고, 검색해 온 Document 리스트를 출력한다.   
하나의 쿼리에 대해서 retrieval을 테스트할 때에는 `invoke` 메서드를, 여러 개의 쿼리에 대해서는 `batch` 메서드를 사용하면 된다. `retriever` 객체를 생성하는 예시 코드는 Appendix를 참고하자.

```python
retriever.invoke("시스템 프롬프트는 어떻게 구성되나요?")
```

```python
retriever.batch(["시스템 프롬프트는 어떻게 구성되나요?", "delimiter의 역할은 무엇인가요?"])
```


## **Concluding Remarks**

이번 포스팅에서는 LangChain 프레임워크의 LCEL과 runnable의 기초적인 개념에 대해 알아보았다. LCEL에 대한 좀 더 자세한 설명[1], runnable 요소와 이들의 입/출력 타입[2] , 다양한 형태의 chain 구성과 병렬 처리 등을 가능하게 해주는 `Runnable` 클래스들[3]에 대해서는 각각 **랭체인LangChain 노트-LangChain 한국어 튜토리얼🇰🇷**을 참고하면 되겠다.  특히 사용자 정의 함수를 runnable로 만들어줌으로써 chain 구성에 있어 유연성과 확장성을 부여할 수 있는 `RunnableLambda` 등을 적재적소에 활용한다면 LLM 애플리케이션 개발의 지평을 넓히는데 도움이 될 것이다.


## References

[1] 테디노트, [CH01-03. LangChain Expression Language(LCEL), 랭체인LangChain 노트-LangChain 한국어 튜토리얼🇰🇷 ](https://wikidocs.net/233344){:target="_blank"}  
[2] 테디노트, [CH01-04. LCEL 고급, 랭체인LangChain 노트-LangChain 한국어 튜토리얼🇰🇷 ](https://wikidocs.net/233345){:target="_blank"}  
[3] 테디노트, [CH01-05. Runnable, 랭체인LangChain 노트-LangChain 한국어 튜토리얼🇰🇷](https://wikidocs.net/233346){:target="_blank"}  

본 포스팅은 LangChain 공식 문서 및 LangChain 한국어 튜토리얼 등을 참고하여 작성되었습니다. 


## Appendix

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(web_path="https://bit.ly/3VunCjC")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings()) 
retriever = vectorstore.as_retriever()
```

