![Document](https://img.shields.io/badge/Document-Korean-black.svg)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
[![DeepLearning](https://img.shields.io/badge/DeepLearning-Keras-red.svg)](https://keras.io)
[![KerasKorea](https://img.shields.io/badge/Community-KerasKorea-purple.svg)](https://www.facebook.com/groups/KerasKorea/)
[![KerasKorea](https://img.shields.io/badge/2018-Contributhon-green.svg)](https://www.kosshackathon.kr/)


# KEKOxTutorial 
전 세계의 멋진 케라스 문서 및 튜토리얼을 한글화하여 케라스x코리아를 널리널리 이롭게합니다. 🇰🇷

## 작업 방식
작업은 Github 저장소를 통해 진행되며, Issue와 Project를 활용합니다. 문서는 기본적으로 마크다운을 사용하며, 다른 포맷이 필요할 경우 다른 방식을 사용할 수 있습니다.

해당 문서와 관련해 내용을 참고한 곳이 있다면 **반드시 참고 링크 혹은 출처를 명시해야 합니다.**

파일 이름은 `이슈번호_문서이름`으로 생성합니다. ex) `09_keras_tutorial.md` ...

이미지 혹은 비디오 파일은 `media` 디렉터리에 모아둡니다. 파일 이름은 `이슈번호_사진번호`로 생성합니다. ex) `09_0.png`, `09_1.png`, `09_0.mov` ...

### 작업순서
1. Issue 중 자신이 시작하고자 하는 작업을 self-assign합니다. 혹은 자신이 assign 되어있는 이슈를 찾습니다.
2. 작업을 시작할 때 Project의 `Todo` column에 위치한 이슈를 `Doing`으로 이동합니다. 
3. 문서 작업을 마치면 Pull Request(PR)를 보냅니다. 
	* 작업을 시작할 때 이슈 번호와 동일한 이름의 브랜치를 생성해 작업한 뒤 작업을 완료하면 `master`로 PR을 요청합니다. ex) 만약 9번 이슈라면, `issue_09`으로 브랜치를 생성합니다. 
	
	```
	$ git checkout -b issue_09
	```
	 
	* Pull Request는 **각각의 작업단위(Issue 단위)** 의 commit으로 보냅니다.
	* Commit message는 `#이슈넘버: 내용`으로 작성합니다 (한글로 작성해도 무방합니다). ex) `#9: Add Keras Tutorial 09`, `#9: 오탈자 수정`
	* 참고 : [Git Style Guide](https://github.com/ikaruce/git-style-guide)

### 문서 작성
> 샘플 문서인 [01_guide_document](https://github.com/KerasKorea/KEKOxTutorial/blob/master/01_guide_document.md)를 참고해주세요.

* 모든 문서는 마크다운(\*.md)으로 작성합니다. 
	* 중요한 부분은 **\*\*bold\*\***, *\*italiic\** 등을 적절히 활용해 강조합니다.
	* 내용의 주제가 나뉘는 경우 대제목과 소제목으로 나눠 가독성을 높힙니다.
	* 코드는 코드 블럭으로 묶어서 나타냅니다.
* 스크린 샷이 필요한 경우 글자가 깨지지 않도록 큰 사이즈로 캡쳐합니다.
* 참고 링크는 하단에 모아둡니다.
* 모두가 함께 보는 글이 될테니 PR 전에 [맞춤법 교정](http://speller.cs.pusan.ac.kr) 하는 것을 추천합니다.

### 용어집
[Wiki 용어집 바로가기](https://github.com/KerasKorea/KEKOxTutorial/wiki/KEKOxTutorial-용어집)

* 용어집 테이블에 자주 사용되는 용어의 영문/한글을 작성해주세요.
* 헷갈리는 용어는 **이슈**에 [용어질문] 올린 뒤 결정되면 올려주세요.
* 이미 용어집에 등록된 단어가 아닌지 확인하고 추가해주세요.
* 프로젝트 후반에 용어는 일괄 수정하겠습니다.
* 자주 확인하고 업데이트해서 알찬 용어집을 만들어 보아요 📖

### 추가 사항
* 이슈는 프로젝트가 진행되면서 계속 추가됩니다.
* 작업하다 어려운 사항이 생기면 망설이지 말고 언제든 이슈 및 슬랙으로 물어보세요. 🤗
