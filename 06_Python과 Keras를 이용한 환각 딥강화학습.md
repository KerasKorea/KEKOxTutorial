## Python과 Keras를 이용한 환각 딥강화학습
원문 : [Hallucinogenic Deep Reinforcement Learning Using Python and Keras](https://medium.com/applied-data-science/how-to-build-your-own-world-model-using-python-and-keras-64fb388ba459)
> 본 문서는 [World Models 논문](https://arxiv.org/abs/1803.10122)을 튜토리얼 형식으로 설명한 것으로, World Models의 대략적인 내용을 학습한 후 읽는 것을 권장합니다. 본 문서에 대한 번역자의 추가 설명은 `인용구`로 표시하였습니다.  

* Keras
* LSTM
* World Models

### 소개

인공지능에 관심이 있다면, World Models을 주목해야 합니다.  
[World Models 블로그](https://worldmodels.github.io/)  
[World Models 논문](https://arxiv.org/abs/1803.10122)  

이 모델은 세 가지 이유에서 걸작이라고 할 수 있는데,  
1. 몇 가지 딥러닝/강화학습 테크닉을 결합해서 놀라운 성과를 보였습니다. - 그 유명한 카레이싱 강화학습 환경을 처음으로 해결한 에이전트입니다.  
2. 매우 쉽게 쓰여져 최신 AI 기술에 관심있는 사람들에게 좋은 학습자료가 됩니다.  
3. 여러분 스스로 솔루션을 코드로 작성해볼 수 있습니다.  

**이 포스팅은 해당 논문의 내용을 순서대로 실습하는 튜토리얼입니다.**  

이 글에서는 기술적인 세부사항들 뿐만 아니라 여러분의 컴퓨터에서 동작하는 에이전트를 만들어 볼 수 있도록 안내할 것입니다 (저는 논문의 저자는 아니지만, 단지 이 놀라운 논문을 저의 해석대로 여러분들에게 공유하는 것입니다). 

### Step 1: 문제 정의

우리는 2차원의 트랙에서 운전하는 강화학습 알고리즘(이하 '에이전트')를 완성할 것이고, 카레이싱 환경은 [OpenAI Gym](https://gym.openai.com/envs/CarRacing-v0/)을 기반으로 합니다.  

각 단위시간마다, 에이전트는 64x64 픽셀의 컬러 이미지(자동차와 주변 환경)를 입력받아 다음 단계에서 취할 행동을 결과값으로 도출해야 합니다 - 구체적으로 핸들방향(-1 ~ 1), 가속(0 ~ 1), 브레이크(0 ~ 1) 값을 의미합니다.  

이 때, 에이전트가 취한 행동은 시뮬레이션 환경에 전달되어 에이전트는 새로운 이미지를 입력받게 되고 사이클이 다시 시작되게 됩니다.  

에이전트는 매 단위시간마다 -0.1씩, 트랙타일을 지날 때마다 +1000/N (N은 트랙타일의 총 개수)씩 보상을 받습니다. 예를 들어, 에이전트가 732 프레임만에 트랙을 완주했다면, 1000 - 0.1*732 = 926.8점을 얻게 됩니다.  

![예시](https://cdn-images-1.medium.com/max/600/1*Yil5VbPHSoHcEOrnGOobUQ.gif)  

위의 예제는 처음 200 프레임 동안 [0, 1, 0]의 행동을 하고, 그 후 무작위로 움직이는 에이전트의 주행입니다. 좋지 않은 전략이라는 것이 눈에 보이죠?  

이 게임의 목표는 에이전트가 주변 정보를 활용해서 다음에 취할 최선의 행동을 학습하도록 하는 것입니다.  

### Step 2: 해결책  

논문의 저자들이 [World Models 블로그](https://worldmodels.github.io/)에 이 모델에 대해서 아주 잘 설명을 해놓았기 때문에, 저는 여기에서 세부사항까지 설명하지는 않을 것입니다. 대신에 모델의 각 부분들이 어떻게 결합되고, 왜 잘 동작하는지에 대한 설명에 집중하겠습니다.  

#### A Variational Autoencoder(VAE)  

#### A Recurrent Neural Network with Mixture Density Network output layer (MDN-RNN)  

#### Controller


### Step 3: 환경 설정

### Step 4: 무작위 데이터 생성

### Step 5: VAE 학습

### Step 6: RNN 데이터 생성

### Step 7: RNN 학습

### Step 8: 컨트롤러(Controller) 학습

### Step 9: 시각화

### Step 10: 환각 학습



