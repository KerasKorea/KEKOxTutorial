## Keras를 이용한 Multi-label 분류(Multi-label classification with Keras)
[원문 링크](https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/)
> 본 튜토리얼에서는 하나의 CNN을 사용해 옷의 종류, 색, 재질을 모두 분류하는 다중 라벨 분석을 다룹니다.

* Keras
* CNN
* Multi-label 
* Classification

----

해당 튜토리얼은 총 네 부분으로 나뉩니다. 

첫 번째 파트에서는 다중 라벨 분류를 위한 데이터 셋(그리고 빠르게 자신만의 데이터 셋을 구축하는 법)에 대해 알아보겠습니다. 

두 번째 파트에서는 다중 라벨 분류를 위해 사용할 케라스 신경망 아키텍처인 소형VGGNet에 대해 알아본 후, 
해당 신경망을 구현해 봅니다.  

세 번째 파트에서는 직접 구현한 소형VGGNet을 다중 라벨 데이터 셋에 대해 학습시켜 봅니다. 

마지막으로 학습된 신경망을 예시 이미지에 대해 테스트 해보는 것으로 본 튜토리얼을 마무리하고, 
다중 라벨 분석이 필요한 경우, 그리고 이 때 주의해야 할 점 몇가지를 알아보도록 하겠습니다. 
