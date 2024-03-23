---
title: 240322 Jekyll Chirpy 설치 과정 중 발생한 이슈 해결
date: 2024-03-22 14:00:00 +09:00
categories: [블로그, Jekyll]
tags:
  [
    Jekyll,
    Chirpy,
  ]
---

# Github fork를 이용한 Jekyll Chirpy 설치

기본적으로 [다음의 블로그](https://jjikin.com/posts/Jekyll-Chirpy-%ED%85%8C%EB%A7%88%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-Github-%EB%B8%94%EB%A1%9C%EA%B7%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0(2023-6%EC%9B%94-%EA%B8%B0%EC%A4%80)/)에서 안내하는 방법을 따라 Chirpy 설치 진행

node.js는 홈페이지를 통해 설치한 상황(.msi 파일 이용)

상기 블로그에서 다음과 같이 안내하고 있음


>npm을 통해 node.js 모듈을 설치합니다.

```shell
npm install && npm run build
```

>node.js 모듈을 설치하지 않으면 `assets/js/dist/*.min.js Not Found` 에러 발생과 함께 블로그 기능이 정상적으로 동작하지 않습니다.

# 증상

내 경우에 블로그를 로컬에서 실행하면, 일단 실행을 되지만, 각 메뉴를 클릭해서 옮겨 다닐 때 `assets/js/dist/*.min.js Not Found` 에러가 계속해서 발생함

또한, 위의 명령(`npm install && npm run build`)을 실행할 경우, 다음과 같은 에러가 발생
```
C:\Projects_Y\GithubPagesBlog\Elyire.github.io>npm install && npm run build

up to date, audited 450 packages in 992ms

70 packages are looking for funding
  run `npm fund` for details

found 0 vulnerabilities

> jekyll-theme-chirpy@6.5.3 prebuild
> npx rimraf assets/js/dist


> jekyll-theme-chirpy@6.5.3 build
> NODE_ENV=production npx rollup -c --bundleConfigAsCjs

'NODE_ENV'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는
배치 파일이 아닙니다.
```

# 에러 해결

**GPT-4를 통해 이 에러의 해결방안을 조언 받음. 1번 cross-env 사용을 이용하여 문제가 해결됨**

### GPT-4의 응답

에러 메시지에 따르면 `NODE_ENV` 환경 변수 설정이 문제인 것으로 보입니다. Windows에서 `NODE_ENV=production`과 같이 환경 변수를 직접 설정하는 구문은 기본적으로 지원되지 않습니다. 이 문제를 해결하기 위한 몇 가지 방법이 있습니다:

1. **cross-env 사용**: `cross-env`는 다양한 OS에서 환경 변수를 설정할 때 발생하는 문제를 해결해주는 도구입니다. 이를 사용하기 위해서는 먼저 `cross-env`를 설치해야 합니다. 터미널에서 다음 명령어를 실행하세요:

   ```
   npm install --save-dev cross-env
   ```

   그런 다음, `package.json` 파일에서 `scripts` 섹션을 찾아 `build` 스크립트를 `cross-env NODE_ENV=production npx rollup -c --bundleConfigAsCjs`로 변경합니다. 예를 들어:

   ```json
   "scripts": {
     "build": "cross-env NODE_ENV=production npx rollup -c --bundleConfigAsCjs"
   }
   ```

**변경 후, `package.json` 파일을 저장하고, 레포지토리 루트에서 다시 npm을 통해 node.js 모듈을 설치하면 성공적으로 설치됨!**
```shell
npm install && npm run build
```

---

2. **Windows 배치 파일 사용**: `NODE_ENV`를 직접 설정하기 위해 Windows 배치 파일을 사용할 수도 있습니다. 예를 들어, 다음과 같은 배치 파일(`setenv.bat`)을 만듭니다:

   ```batch
   @echo off
   SET NODE_ENV=production
   npx rollup -c --bundleConfigAsCjs
   ```

   그런 다음, `package.json`의 `scripts` 섹션에서 `build` 스크립트를 `setenv.bat`으로 변경합니다.

3. **직접 명령어 실행**: 환경 변수를 직접 설정하는 대신, 명령 프롬프트에서 직접 환경 변수를 설정하고 `rollup`을 실행할 수도 있습니다. 명령 프롬프트에서 다음 명령어를 차례로 실행하세요:

   ```
   SET NODE_ENV=production
   npx rollup -c --bundleConfigAsCjs
   ```

이러한 방법 중 하나를 사용하여 `NODE_ENV` 환경 변수 설정 문제를 해결할 수 있습니다. 여전히 문제가 해결되지 않는다면, 프로젝트의 문서를 확인하거나, 필요한 경우 프로젝트의 이슈 트래커나 관련 커뮤니티에 질문을 남겨 보세요.


