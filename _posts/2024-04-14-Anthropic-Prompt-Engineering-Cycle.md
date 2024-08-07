---
title: 프롬프트 개발 사이클(Prompt Development Lifecycle)
date: 2024-04-15 15:00:00 +09:00
categories: [프롬프트 엔지니어링, 프롬프트 작성 원리]
tags: [프롬프트 개발 사이클, 프롬프트 평가, 프롬프트 최적화]
pin: true
---
Claude의 개발사인 Anthropic에서 제공하는 프롬프트 엔지니어링 가이드 문서가 있습니다.[1] 이 문서는 서두에서 **프롬프트 개발 사이클(prompt development lifecycle)**에 대해 다룹니다. 그리 길지 않은 내용이지만 프롬프트 개발자들이 꼭 염두에 두어야 할 핵심들을 잘 짚어주고 있다고 생각하여 여기에 소개합니다. 원문의 큰 틀을 따라가되, 그대로 번역하지 않고 제 나름대로 내용을 추가하며 본 포스팅을 구성하였음을 밝힙니다. 원문을 읽기 원하시는 분들은 Reference 1번의 링크를 참고해 주세요. 

## **경험적 과학으로서의 프롬프트 엔지니어링**

문서에서는 프롬프트 엔지니어링을 AI 모델의 출력 성능을 향상을 위해 프롬프트를 반복적으로 테스트하고 개선하는 과정을 거치는 **경험적 과학empirical science**으로 정의합니다. 

화학공학의 열역학을 배우다 보면 경험적 상관관계(empirical correlation)라는 것이 나옵니다. 이는 물리화학 현상에 대한 이론적 모델링이 아닌, 실험을 통해 수집된 데이터를 바탕으로 도출한 상관관계식입니다. 

실험실 스케일의 작은 플라스크에서 일어나는 화학반응 현상은 우리가 교과서에서 배우는 방정식을 적용하여 해석할 수 있지만, 동일한 화학반응이라도 대형 화학 공정 플랜트로 스케일업(scale-up)하여 운전할 때 나타나는 현상들은 이상적인 방정식으로 설명하기 어려운 경우가 대부분입니다. 

이때는 특정 플랜트(즉, 특정 시스템)에서 일어나는 이 화학반응과 관련된 수많은 데이터를 얻은 후, 이를 바탕으로 이 플랜트 운전에 적용 가능한 경험적 상관관계식을 세우게 됩니다.

갑자기 화학공학에서 다루는 대형 화학 공정 플랜트를 언급한 이유는 이 경험적(empirical)이라는 용어 때문입니다. 프롬프트 엔지니어링에서도, 경험적으로 얻은 데이터들을 토대로 입력과 출력에 대한 상관관계 혹은 패턴을 얻어냄으로써 최적 프롬프트 설계를 달성한다는 점이 경험적 상관관계식 도출과 유사한 점이라는 생각이 들었습니다. 

사실 프롬프트 엔지니어링이란 매우 까다로운 작업입니다. 입력 프롬프트의 토큰 구성이 조금만 변하면 출력 결과가 크게 바뀌기도 합니다. 입력의 변화가 가져오는 출력의 변화가 항상 일정하지도 않습니다. 이는 대형언어모델(Large Language Model, LLM) 자체의 본질을 생각해보면 당연한 일인지도 모릅니다.  

LLM은 언어를 모델링하는 복잡한 비선형 시스템입니다. 입력은 계속되는 비선형 변환을 거칩니다. 게다가 마지막에서 그 출력이 확률적으로 결정됩니다. 그러니 입력의 변화에 따른 출력의 변화가 갑자기 커지기도 하고, 일정하지도 않게 됩니다. 어찌보면 우리가 제어할 수 있는 윈도우가 너무 작은 것일 수도 있습니다.  

이러한 시스템에서 최적의 출력을 얻기 위해서는 입력 프롬프트를 **'경험적으로, 체계적으로'** 반복해서 테스트할 수 밖에 없습니다. 프롬프트 엔지니어링을 '경험적 과학'으로 정의한 Anthropic의 시각은 타당하다고 생각됩니다.

## **프롬프트 엔지니어링의 핵심**

문서에서는 프롬프트 엔지니어링의 핵심이 단순히 '프롬프트를 잘 작성'하는데 있지 않다고 합니다. 오히려 프롬프트 엔지니어링 과정에서 대부분의 노력은 **강력한 평가 세트(테스트 케이스 세트)를 구축하고, 개발된 프롬프트를 해당 케이스들에 대해  테스트하고 개선하는 반복적인 과정**에 들어간다고 역설하고 있습니다. 평가(evaluations, 보통 evals라고 부름)의 중요성을 강조하고 있는 것입니다. 

사실 프롬프트를 개발할 때, 우리도 의식하지 못한 채 평가-개선 작업을 반복합니다. 하지만 이 과정이 체계적이지 못한 경우가 있습니다. 충분한 테스트 케이스를 마련하고 프롬프트를 다듬어 나가기 보다는, 먼저 프롬프트를 작성하고, 대략 원하는 결과가 출력되는지 체크하고, 원하는 결과가 나올 때까지 수정하고, 어느 정도 원하는 결과가 나오면 거기서 멈추는 경우가 바로 그것입니다. 개발된 프롬프트를 공들여 평가하기 보다는, 어떤 특정 효과를 가져다주는 기법, 문구 발견에 치중할 때가 더 많기도 합니다. 

Anthropic의 문서는 실제 프로덕션에 투입될 프롬프트 개발을 염두에 두고 쓰여진 것 같습니다. 이 짧은 문서는 처음부터 끝까지 평가(evals)의 중요성을 명시적으로든, 암시적으로든 계속해서 강조합니다. '평가'라는 말이 한국어 문장 사이에 있을 때 그 존재감이 그리 크게 드러나지 않는 경우가 종종 있습니다. 우리가 일상 속에서 '평가'라는 말을 다양한 의미로 너무 자주 사용하고 있어서 그런 것 같습니다(사견입니다). 그래서 본 포스팅에서는 이 존재감을 조금이라도 더 드러내고자 **평가Evals**라고 표기하도록 하겠습니다.  

## **프롬프트 개발 프로세스(Prompt Development Lifecycle)**

이 문서는 최적의 프롬프트 성능을 보장하기 위해 **원칙에 입각한principled**, **테스트 주도 개발 방식test-driven-development approach**을 권장합니다. 다시 말해, **평가Evals**에 기반한 체계적인고 일관된 접근 방식을 사용하라는 것입니다. 다음 그림은 이를 잘 보여주고 있습니다. 이제부터 각 단계에 대해 하나씩 알아보도록 하겠습니다. 

![](https://files.readme.io/49181ae-Prompt_eng_lifecycle.png)

### Step 1. **작업 정의 및 성공 기준 설정(Define the task and success criteria)**

가장 중요한 첫 단계는 **모델이 수행할 작업을 명확하게 정의**하고, 프롬프트 성능을 평가할 수 있는 명확하고 객관적인 **평가**(Evaluations, 이하 Evals) **기준**을 세우는 것입니다. 그리고 이를 바탕으로 **성공 기준Success Criteria**을 정의하는 것입니다. 

명백한 답이 나오는 케이스는 객관적인 **평가Evals 기준**을 적용하기 용이합니다. 예를 들어 수학 문제 풀이나 코딩 문제 풀이와 같이 정답이 명확한 경우, 모델이 생성한 출력과 정답을 비교하여 정확도를 측정할 수 있습니다. 그러나, 개방형 답변(open-ended answer)이 출력인 경우(예를 들어 철학자들의 명언을 해석해주는 태스크와 같이 장문의 출력이 생성되는 경우)는 객관적인 **평가Evals 기준**을 세우기 어려울 수 있습니다. 이 때는 장문의 출력 내에 반드시 포함되어야 할 핵심 내용의 리스트로 만들고 이를 기준으로 하여 생성된 출력을 평가할 수도 있습니다.

**성공 기준**은 프롬프트 엔지니어링의 목표, 즉 어떤 수준의 성능을 달성했을 때 해당 프롬프트를 최적화된 것으로 간주할 것인지를 정의합니다. 이는 평가 기준을 바탕으로 설정되며, 예를 들어 "정확도 90% 이상", "핵심 내용 포함률 90% 이상" 등과 같이 구체적인 목표치를 설정할 수 있습니다. **성공 기준**을 명확히 정의하면, 프롬프트 개발 과정에서 방향성을 잃지 않고 체계적으로 최적화를 진행할 수 있습니다. 

OpenAI 또는 Anthropic의 API를 사용하여 LLM 애플리케이션을 구축하는 경우에는 프롬프트 엔지니어링 시 성공 기준으로 고려해야 할 항목에 **지연시간**과 **가격**까지도 포함될 수 있습니다. 모델에 허용되는 응답시간은 어느 정도여야 하는지? 이는 애플리케이션의 실시간 요구사항과 사용자 기대치에 따라 달라질 수 있습니다. 그리고, 개발한 LLM 애플리케이션이 GPT-4나 Claude 3 Opus와 같은 고급 모델을 호출하는 API를 사용한다면, 입/출력 토큰 수를 프롬프트 단계에서 조정하여 배정된 예산에 부담을 주지 않아야 합니다.

이러한 작업 정의와 성공 기준 설정은 프롬프트 엔지니어링뿐만 아니라, LLM 기술을 실제 애플리케이션에 적용하고 배포하는 전체 도입 프로세스(adoption process)에서도 중요한 역할을 합니다. 도입 프로세스 전반에 걸쳐 명확하고 측정 가능한 성공 기준을 갖추면, 충분한 정보와 지식을 바탕으로 합리적이고 근거 있는 결정을 내릴 수 있고 올바른 목표를 향해 프롬프트를 최적화해 나갈 수 있습니다.

### Step 2. **테스트 케이스 개발(Develop test cases)**

 다음 단계는 개발자가 의도한 애플리케이션 사용 사례를 포괄하는 다양한 테스트 케이스 세트(test case sets)를 만드는 것입니다. 이 테스트 케이스 세트에는 전형적인 케이스와 특이 케이스(극단적이거나 예외적인 입력, edge case) 모두를 포함해야 합니다. 이를 통해 개발하고 있는 프롬프트의 견고성(robustness)을 확보할 수 있습니다. 사전에 잘 정의된 테스트 케이스(well-defined test cases upfront)를 구축해 놓음으로써 Step 1에서 설정한 **평가Evals** 및 성공 기준에 비추어 프롬프트의 성능을 객관적으로 측정할 수 있습니다.

프롬프트 엔지니어링에서 개발하는 프롬프트는 주로 시스템 프롬프트(System Prompt)를 의미합니다. 시스템 프롬프트는 모델이 수행해야 할 작업, 역할, 행동 방식 등을 설명하는 일종의 지시사항이나 가이드라인이라고 할 수 있습니다. 

테스트 케이스는 이렇게 설계된 시스템 프롬프트를 **평가Evals**하기 위한 **입력 데이터 세트**입니다. 개발자는 다양한 테스트 케이스를 통해 시스템 프롬프트가 의도한 대로 모델을 제어하고 작동시키는지 확인할 수 있습니다.

테스트 케이스를 만들 때는 다음 사항을 고려해야 합니다.

- 포괄성: 가능한 한 많은 사용 사례와 시나리오를 포함하여 프롬프트의 성능을 종합적으로 평가할 수 있어야 합니다.

- 다양성: 일반적인 입력부터 극단적이거나 예외적인 경우까지 다양한 유형의 입력을 테스트해야 합니다.

- 명확성: 각 테스트 케이스는 명확한 입력과 기대 출력을 정의하여 **평가Evals** 시 모호함이 발생하지 않도록 해야 합니다.

- 효율성: 테스트 케이스의 수는 적절해야 합니다. 너무 많으면 **평가Evals**에 시간이 오래 걸리고, 너무 적으면 충분한 검증이 이루어지지 않을 수 있습니다.

### Step 3. **예비 프롬프트 설계(Engineer the preliminary prompt)**

이제 프롬프트를 작성하는 단계입니다. 일단 초기(initial) 프롬프트를 작성합니다. 여기에는 다음과 같은 항목들이 포함될 수 있습니다. 

- 모델에게 시킬 작업을 설명하는 지시사항(Instructions) 

- 좋은 응답이 갖추어야 할 특성에 대한 설명(i.e. 응답의 길이, 어투나 문체, 전문성 여부 등)

- 기대하는 입력과 개발자가 원하는 출력 형식 지정 

- 입출력 예시 (i.e. 퓨샷 프롬프팅)

- 제약 조건(Constraints): 출력 내용이나 형식에 있어 강조하고 싶거나 피하고 싶은 사항들.

 예비 프롬프트가 완벽해야 할 필요는 없지만, 작업의 핵심을 담고 있어야 하며 모델이 이해할 수 있을 만큼 명확해야 합니다. 이 예비/초기 프롬프트는 프롬프트 개선(prompt refinement)의 시작점(starting point) 역할을 합니다. 이를 바탕으로 이후 테스트-평가-피드백을 통해 프롬프트를 지속적으로 개선해 나갈 수 있습니다.

### Step 4. **테스트 케이스를 이용한 프롬프트 테스트(Test prompt against test cases)** 
테스트 케이스(Step 2)를 입력으로 사용하여 예비 프롬프트(Step 3)가 어떤 모델의 응답을 이끌어 내는지 테스트하고 그 결과를 **평가Evals**하는 단계입니다. **평가Evals 방식**에는 태스크와 응답 특성에 따라 사람이 직접하는 평가, 정답키와의 비교(정답이 명확한 경우 코드를 이용), 미리 설정된 평가 기준(혹은 채점기준표)에 기반하여 언어모델이 채점하게 하는 방식 등이 있습니다.[2] 중요한 것은 개발한 프롬프트의 성능을 일관되고 체계적인 **평가Evals 방식**에 따라 평가하는 것입니다.

### Step 5. **프롬프트 개선(Refine prompt)** 

Step 4에서의 테스트 및 평가 결과 분석을 바탕으로, 개발 중인 프롬프트를 반복적으로 개선하는 단계입니다(iteratively refining). 이 과정을 통해 우리는 테스트 케이스에 대해 프롬프트의 성능을 개선하고, 성공 기준을 더 잘 충족시킬 수 있습니다. 프롬프트 개선을 위해서 다음과 같은 조치를 취할 수 있습니다. 

- 지시사항의 명확화: 모호하거나 혼동을 일으킬 수 있는 표현을 명확한 지시사항으로 수정합니다. **당신이 작성한 프롬프트의 지시사항을 주변 사람들이 읽어보고 바로 이해할 수 있다면, 언어모델도 그러할 것입니다.** 

- 추가 정보 제공: 작업 수행에 필요한 맥락, 백그라운드 지식, 제약 조건 등을 추가로 제공합니다. 

- 예시 보완: 필요하다면 다양한 유형의 입/출력 예시를 추가하여 언어모델의 이해를 돕습니다. 

- 불필요한 정보 제거: 작업 수행과 무관한 내용을 삭제하여 프롬프트의 간결성을 유지하도록 합니다. 

여기서 주의할 점은 프롬프트가 작은 수의 테스트 케이스에 대해서 너무 최적화되지 않도록 하는 것입니다. 이는 과적합(overfitting)과 일반화 능력의 저하를 초래할 수 있습니다. 프롬프트 개선 과정에서는 준비한 테스트 케이스 이외에도 다양한 실제 입력에 대한 성능을 검증하는 것이 중요합니다.  

### Step 6. **프롬프트 배포(Ship the polished prompt)**

테스트 케이스 전반에 걸쳐 좋은 성능을 보이고, 성공 기준을 충족하는 프롬프트가 완성되면, 이를 실제 애플리케이션 구현을 위해 배포합니다. 배포 이전에도 그렇지만 배포 이후에도 개발자가 예상하지 못했던 특이 케이스(edge case) 등에 대응할 준비를 해야 합니다. 프롬프트를 실제 환경에 배포할 때는 다음의 사항들을 고려해야 합니다. 

- 점진적 배포: 가능하다면 일부 사용자 그룹을 대상으로 먼저 베타 테스트를 진행하여 안정성과 성능을 확인한 후, 전체 사용자에게 배포하는 것이 안전합니다. 

- 사용자 피드백 수집: 실제 사용자들의 피드백을 적극적으로 수렴하여 개선 사항을 파악하고 프롬프트를 지속적으로 발전시켜 나가야 합니다.

- 유지 보수 체계 확립: 프롬프트의 버전 관리, 업데이트 절차, 롤백 계획 등을 사전에 수립하여 안정적인 운영이 가능하도록 합니다.

배포 이후에도 프롬프트 엔지니어링은 계속됩니다. 실제 사용 과정에서 발견되는 새로운 사례들을 테스트 케이스에 추가하고, 지속적인 모니터링과 분석을 통해 프롬프트를 개선해 나가는 것이 중요합니다. 최종 배포는 프롬프트 엔지니어링의 종착점이 아니라 새로운 출발점입니다.

## **Concluding Remarks**

이번 포스팅에서 살펴본 것처럼 성공적인 프롬프트 엔지니어링을 위해서는 태스크 정의(task definition)부터 배포, 운영에 이르기까지 체계적인 접근 방식이 필요합니다. 특히 초기 설계에 충분히 공을 들이고, 객관적인 테스트/평가와 반복적인 개선을 수행하는 것이 최적 프롬프트 개발의 핵심이라는 것을 꼭 기억해야 합니다.   

## **References**

[1] Anthropic User Guides, "Prompt Engineering" **2024**, Available: [https://docs.anthropic.com/claude/docs/prompt-engineering](https://docs.anthropic.com/claude/docs/prompt-engineering){:target="_blank"}  
[2] Anthropic Cookbook, "Building Evals" **2024**, Available: [https://github.com/anthropics/anthropic-cookbook/blob/main/misc/building_evals.ipynb](https://github.com/anthropics/anthropic-cookbook/blob/main/misc/building_evals.ipynb){:target="_blank"}  

## ***Appendix***
👉**Tip 1**💡: 프롬프트 개발 과정 전반에 걸쳐, 프롬프트 성능의 상한선을 설정하기 위해 가장 강력한(most capable) 모델과 제약없는 프롬프트 길이로 시작하는 것이 좋습니다. 일단 원하는 출력 품질을 달성하고 나면, 필요에 따라 프롬프트 길이를 줄이거나, 덜 capable한 모델을 사용하는 등의 접근을 통해 지연 시간(latency)과 비용을 최적화할 수 있습니다.  
  
👉**Tip 2**💡: **평가Evals**와 관련해 [OpenAI의 evals](https://github.com/openai/evals){:target="_blank"}도 참고해 주세요.