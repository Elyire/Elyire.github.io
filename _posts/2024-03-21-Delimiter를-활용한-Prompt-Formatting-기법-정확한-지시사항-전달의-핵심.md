---
title: Delimiter를 활용한 Prompt Formatting 기법 - 정확한 지시사항 전달의 핵심
date: 2024-03-21 20:00:00 +09:00
categories: [프롬프트 엔지니어링, 프롬프트 작성기법]
tags: [프롬프트 작성기법, 시스템 프롬프트]
pin: false
---


시스템 프롬프트를 통해 LLM에게 특정 태스크 수행을 요청할 때 반드시 사용해야 하는 테크닉이다. 특히 LLM 애플리케이션을 구현할 때, LLM으로 하여금 다양한 작업을 수행하도록 하기 위해 여러가지 지시사항 및 연관 내용들이 시스템 프롬프트에 담길 때 특히 유용하다. 
이것은 한 마디로, 시스템 프롬프트 작성을 하나의 '글쓰기'로 생각하고 접근하는 것이다. 프롬프트 엔지니어 서승완 유메타랩 대표는 이를 '구조적 글쓰기'라 칭하였다.[1] 즉, 내가 LLM에게 요구하는 바가 구조를 갖춘 명료한 글로 표현함으로써 지시사항이 요구하는 바를 LLM이 정확하게 파악할 수 있도록 해주는 것이다.   

## 시스템 프롬프트의 구성

일반적으로 시스템 프롬프트를 구성하는 섹션들로는 `Role`, `Instructions`, `Constraint`, `Input` , `Output` 정도로 나누어볼 수 있다.  각각을 간단하게 살펴보기로 하자. 

1. `Role`: 많은 연구들에서 LLM에게 역할을 부여하면 그 답변 성능이 향상된다는 것을 보고해왔다. 특히 `GPT-3`, `GPT-3.5` 이하의 레벨에서는 `Role`의 사용여부에 따라 그 성능 차이가 두드러진다. 많은 GPT 기반의 LLM 서비스들이 여전히 토큰비용에 대한 부담으로 인해 `GPT-3.5` 레벨의 LLM을 사용하고 있다. 이런 경우, 애플리케이션에 따른 특정 역할 부여는 LLM이 원하는 결과를 출력하는데 있어서 주요한 요소로 작용한다.

2. `Instructions`: 이 섹션의 내용은 말 그대로 시스템 프롬프트를 통해 LLM에게 지시하고자 하는 사항들로 구성되어 있다. LLM이 수행해야 할 태스크들을 정의한다. `Your task is to ~`를 사용하여 시작하면 가장 무난하다. `Instructions`의 하위항목으로 이러한 지시가 이루어지는 맥락(`Context`)을 포함시킬 수 있다(하위항목으로 지정하는 방법은 조금 뒤에 알아보자).

3. `Constraint`: `Instructions`에는 지시사항들만 명료하게 기술하고, LLM이 태스크를 수행하는데 있어 초점을 맞추어야 할(혹은 피해야 할) 추가적인 사항(제약조건)들을 이 부분에 포함시킨다. OpenAI 문서에서는 이 부분과 관련하여 "What should be emphasized or avoided?"라는 표현을 사용했다.[2] 

4. `Output`: 특별히 원하는 출력 형식이 있다면, 이 섹션에서 출력형식을 지정해준다. 

5. `Input`: 필요하다면 어떤 입력이 주어질 지에 대해 따로 섹션을 만들어 LLM에게 관련 정보를 제공할 수 있다. 혹은 `Instruction`에서 `You will be provided with {information about input}`같은 문장을 이용하여 입력에 대한 정보를 제공하기도 한다.  
  
  
- *예를 들어 RAG(Retrieval Augmented Generation) 애플리케이션을 위한 시스템 프롬프트에서는 `Input` 섹션의 내용에 `user query` 그 자체로 프롬프트를 끝내는 것도 좋은 방법이다.[3] 현재 많이 사용되고 있는 GPT와 같은 디코더 유형의 언어모델들은 인과적 셀프어텐션 메커니즘(causal self-attention) 을 사용하기 때문에, 프롬프트에 이어지는 토큰들을 생성하기 직전에 쿼리를 위치시킴으로 쿼리를 구성하고 있는 토큰 중, 마지막 토큰에 인코딩되어 있는 정보를 가장 효과적으로 활용할 수 있다. 이러한 방식으로, 언어 모델은 마지막으로 제시된 토큰에서 추출한 맥락을 기반으로 응답을 생성함으로써, 쿼리의 핵심적인 부분에 집중할 수 있다. 특히 인과적 셀프어텐션 메커니즘을 사용하는 언어 모델에서는, 각 토큰이 이전의 모든 토큰들과의 관계를 고려하여 처리되므로, 프롬프트의 마지막 부분에 위치한 쿼리는 모델이 생성할 내용에 직접적인 영향을 미칠 가능성이 크다. 이러한 접근 방식은 모델이 쿼리의 내용을 더 잘 이해하고, 더 관련성 높고 일관된 응답을 생성하는 데 도움이 될 수 있다. 따라서, 사용자의 질문이나 요청을 프롬프트의 마지막 부분에 배치함으로써, 모델이 제공하는 답변의 정확도와 관련성을 높일 수 있다. 다만, 이러한 방식이 모든 상황에서 최선의 선택이 될 수 있는 것은 아니다. 쿼리의 내용과 복잡성, 모델의 특성 등에 따라 프롬프트 구성 방식을 적절히 조정해야 할 수도 있다.*[4]


## 마크다운 헤더 태그를 이용한 Prompt Formatting

이러한 섹션들을 LLM이 잘 구분하여 인식하고 처리할 수 있도록 도움을 주는 것이 바로 구분기호(delimiter)를 이용한 prompt formatting이다. 다양한 구분기호들 중, 어떤 것이 가장 효과적이라고 말하기는 어려우나, 주목할만한 구분기호는 마크다운 문법과 xml 태그이다. '마크다운 활용 기법'에 대해서는 "프롬프트 엔지니어링 교과서" 2장에서 자세히 다루고 있으며[5], 여기서는 마크다운 문법의 "헤더 태그"를 사용하는 방법에 대해서만 이야기하고자 한다. 

마크다운 헤더 태그는 프롬프트의  계층 구조를 만드는 역할, 즉, 주요 섹션과 하위 섹션을 구분해주는 역할을 한다. 이러한 구분은 LLM이 프롬프트의 전체 구조를 파악하는데 큰 도움을 준다. 위에서 소개한 섹션들을 마크다운 헤더를 통해 구분하여 작성한 다음의 예시를 살펴보자.

여기서는 LLM의 역할을 사용자의 여행 계획에 도움을 주는 여행 가이드로 설정하였다. LLM 여행가이드가 수행해야 할 사항에 대한 지시(instructions) 부분은 마크다운 헤더 태그를 이용해 세 계층으로 나누어져 있다. 가장 상위 계층(`# Instructions`)은 LLM이 수행해야 할 주요 태스크를 정의한다. 그 하위 계층으로(`## Context for...`) 왜 이런 태스크를 수행해야 하는지 LLM이 이해하는데 필요한 맥락(배경 정보)을 제공한다. 가장 하위 계층에서는(`### Specific explanation about...`) 이 배경정보에 대해 더 자세한 설명을 제공함으로써 여행 가이드로 설정된 LLM이 좀 더 사용자와 관련성이 높고, 개인화된 추천을 만들어낼 수 있다.  

```
# Your role
I want you to act as a virtual travel guide who helps users plan their trips and provides information about various destinations.

# Instructions
Your task is to assist users in creating personalized travel plans based on their preferences, budget, and time constraints. Provide recommendations for accommodations, transportation, activities, and dining options.

## Context for the instructions
Users may have different travel styles, such as luxury, budget, adventure, or relaxation. They may also have specific interests like history, culture, nature, or food.

### Specific explanation about context of the instructions
When offering recommendations, consider factors such as the user's age, group size, travel season, and any mobility limitations they may have. Tailor your suggestions to create a unique and enjoyable travel experience.

# Constraints
- Focus on providing information and recommendations rather than making bookings or transactions.
- Avoid recommending destinations or activities that may be dangerous, illegal, or culturally insensitive.

# Output
Present your recommendations in a clear, organized manner. Use bullet points or numbered lists when appropriate. Provide brief descriptions and highlights for each suggestion.

# Input
To generate personalized recommendations, ask the user for the following information:
- Destination(s) they want to visit
- Travel dates and duration
- Budget
- Travel style and preferences
- Any specific interests or must-see attractions
```


세 개의 계층으로 나누어 기술한 내용을 하나의 문단으로 뭉뚱그려 LLM에게 지시사항으로 제공할 수도 있다. 그러나 이렇게 하면 정보가 뒤섞여 있어 LLM에게 지시하고자 하는 중요한 내용을 LLM이 파악하기 어려울 수 있다. 

LLM 입장에서는 지시 사항을 계층 구조로 나누어 제공하는 것이 이를 훨씬 더 명확하게 이해하고 수행하는 데 도움이 된다. 그 이유는 다음과 같다. 

1. 구조화된 정보 전달: 계층 구조를 사용하면 정보를 체계적으로 구성할 수 있다. 주요 항목과 하위 항목을 구분하고, 관련 정보를 그룹화할 수 있다. 이는 LLM이 지시 사항의 전체 구조를 빠르게 파악하고 이해하는 데 도움이 된다.

2. 정보의 우선순위 파악: 계층 구조는 정보의 중요도와 우선순위를 나타내는 데에도 도움이 된다. 상위 레벨의 항목은 일반적으로 더 중요하거나 포괄적인 내용을 담고 있고, 하위 레벨로 갈수록 세부 사항이나 추가 설명을 제공하는 식이기 때문이다. 이를 통해 LLM이 태스크를 수행할 때 우선적으로 고려해야 할 사항을 파악할 수 있다.

## XML 태그를 이용한 Prompt Formatting

헤더 태그를 이용해 구분한 섹션 내에서 특정 내용을 삽입하고 이를 다른 부분들과 구분하기 위해서 XML 태그를 활용할 수 있다. 위의 프롬프트 예시에 XML 태그를 추가적으로 이용하여 프롬프트의 내용을 더욱 분명하게 만들 수 있다.

```
# Your role
I want you to act as a virtual travel guide who helps users plan their trips and provides information about various destinations.

# Instructions
Your task is to assist users in creating personalized travel plans based on their preferences, budget, and time constraints. Provide recommendations for accommodations, transportation, activities, and dining options based on the information provided in the following article:

<article>
[Article content goes here]
</article>

<context>
## Context for the instructions
Users may have different travel styles, such as luxury, budget, adventure, or relaxation. They may also have specific interests like history, culture, nature, or food.

### Specific explanation about context of the instructions
When offering recommendations, consider factors such as the user's age, group size, travel season, and any mobility limitations they may have. Tailor your suggestions to create a unique and enjoyable travel experience.
</context>

# Constraints
- Focus on providing information and recommendations rather than making bookings or transactions.
- Avoid recommending destinations or activities that may be dangerous, illegal, or culturally insensitive.

# Output
Present your recommendations in a clear, organized manner. Use bullet points or numbered lists when appropriate. Provide brief descriptions and highlights for each suggestion.

# Input
To generate personalized recommendations, ask the user for the following information:
- Destination(s) they want to visit
- Travel dates and duration
- Budget
- Travel style and preferences
- Any specific interests or must-see attractions
```

여기서는 `# Instruction`에 특정 문서의 정보를 바탕으로 추천을 제공하라는 지시가 추가되었다. 특정 문서가 위치해 있는 곳을 프롬프트 상에서 구분하기 위해 `<article>...</article>` 태깅을 이용하였다. 이를 통해 LLM이 지시사항에 있는 'following article'이 무엇인지 쉽게 파악할 수 있게 하였다.   
또한 `# Instruction`의 하위항목들이(`## Context for the instructions`와 `### Specific explanation about context of the instructions`) 지시사항의 맥락과 관련된 내용이라는 것을 더욱 분명하게 구분하기 위해 이들 내용을 `<context>...</context>`에 샌드위치 시켰다. 

## Concluding remarks

이번 포스팅에서는 구조화된 프롬프트 작성법인 Prompt Formatting에 대해 알아보았다. 마크다운 헤더 태그와 XML 태그를 활용하여 프롬프트를 계층적으로 구성하고 각 섹션을 명확히 구분함으로써, LLM이 지시사항을 보다 정확하게 이해하고 수행할 수 있도록 돕는 방법을 소개하였다. 이를 응용하여 다양한 애플리케이션에 맞는 최적의 프롬프트 포맷을 만들고 테스트해보도록 하자. 

## References

[1] ["AI조련사' 프롬프트 엔지니어 서승완 대표를 만나다"](https://www.econovill.com/news/articleView.html?idxno=641652){:target="_blank"}  
[2] [GPT Builder](https://help.openai.com/en/articles/8770868-gpt-builder#h_10cb62e803){:target="_blank"}  
[3] LangSmith Hub, [pwoc517/crafted_prompt_for_rag](https://smith.langchain.com/hub/pwoc517/crafted_prompt_for_rag?organizationId=8b51adb9-c530-5edf-93c6-b6d05bf3fc3f){:target="_blank"}  
[4] [디코더 유형 트랜스포머 아키텍처 개론: 초보자를 위한 Conceptual Tutorial](https://cogdex-dtta.streamlit.app/){:target="_blank"}  
[5] [서승완, '프롬프트 엔지니어링 교과서', 애드앤미디어 2023](https://www.yes24.com/Product/Goods/122318886){:target="_blank"}  