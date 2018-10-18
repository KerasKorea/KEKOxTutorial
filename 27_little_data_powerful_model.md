# 작은 데이터셋으로 강력한 이미지 분류 모델 설계하기

원문: [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), Francois Chollet

번역: 허남규, [itsnamgyu@gmail.com](mailto:itsnamgyu@gmail.com)

이 글은 적은 양의 데이터를 가지고 강력한 이미지 분류 모델을 구축하는 방법을 소개합니다. 수백에서 수천 장 정도의 작은 데이터셋을 가지고도 강력한 성능을 낼 수 있는 모델을 만들어볼 것입니다.

저희는 세 가지 방법을 다룰 것입니다.

- 작은 네트워크를 처음부터 학습 (앞으로 사용할 방법의 평가 기준)
- 기존 네트워크의 *bottleneck feature* 사용
- 기존 네트워크의 상단부 레이어 *fine-tuning*

저희가 다룰 케라스 기능은 다음과 같습니다.

- `ImageDataGenerator`: 실시간 이미지 augmentation
- `flow`: `ImageDataGenerator` 디버깅
- `fit_generator`: `ImageDataGenerator`를 이용한 케라스 모델 학습
- `Sequential`: 케라스 내 모델 설계 인터페이스
- `keras.applications`: 케라스에서 기존 이미지 분류 네트워크 불러오기
- 레이어 동결 (freezing)을 이용한 fine-tuning

## 시작하기: 학습 이미지 2,000장

시작하기 전에 최상의 학습 환경을 조성해보겠습니다.

- 케라스, SciPy, PIL을 설치해주세요. 만약 NVIDIA GPU를 사용하실 분이라면 cuDNN을 설치해주세요. 저희는 적은 양의 데이터만 다룰 것이기 때문에 GPU가 꼭 필요하지는 않습니다.
- 이미지를 학습, 검증, 테스트 세트로 나누어 아래와 같이 준비해주세요. 확장자는 `.png` 또는 `.jpg`를 권합니다.

> GPU 텐서플로우를 사용하실 경우 conda를 이용해서 설치하실 것을 권합니다. [\[참고\]](https://towardsdatascience.com/stop-installing-tensorflow-using-pip-for-performance-sake-5854f9d9eb0c).

```
data/
	test/
		dogs/
			dog0001.jpg
			dog0002.jpg
			...
		cats/
			cat0001.jpg
			cat0002.jpg
			...
	train/
		...
	validation/
		...
```

고양이와 강아지 이외의 이미지 데이터를 찾으신다면 [Flickr API](https://www.flickr.com/services/api/)를 활용해보세요. 태그 기반 검색을 사용해서 라이센스 제약이 없는 이미지를 쉽게 찾을 수 있습니다.

이번 글에서는 사용할 데이터는 [캐글](https://www.kaggle.com/c/dogs-vs-cats/data)에서 가져온 것입니다. 학습 데이터로 1,000장의 고양이 사진과 1,000장의 강아지 사진을 가져왔습니다. 검증 데이터는 별도로 각각 400장을 추가로 가져왔습니다. 기존 캐글 데이터는 총 25,000장으로 이루어져 있지만, 적은 데이터로 효과적인 모델을 학습시켜보고자 일부만 가져왔습니다.

사실 1,000장의 사진을 가지고 복잡한 이미지 분류 모델을 학습시키는 것은 정말 어렵습니다. 하지만 실제 상황에서 자주 접하게 될 법한 문제입니다. 의료 영상 같은 경우 이 정도의 데이터를 구하는 것조차 거의 불가능하죠. 적은 데이터를 가지고 최대 성능을 쥐어짜는 것이야말로 데이터 과학자의 핵심 자질이라 할 수 있습니다.

![고양이와 강아지](https://blog.keras.io/img/imgclf/cats_and_dogs.png)

앞서 언급한 고양이 vs 강아지 대회를 캐글에서 개최한지 이제 근 5년이 되어가는데, 그때 당시만 하더라도 캐글에서는 다음과 같은 발표문을 냈습니다.

"In an informal poll conducted many years ago, computer vision experts posited that a classifier with better than 60% accuracy would be difficult without a major advance in the state of the art. For reference, a 60% classifier improves the guessing probability of a 12-image HIP from 1/4096 to 1/459. The current literature suggests machine classifiers can score above 80% accuracy on this task [\[참고\]](http://xenon.stanford.edu/~pgolle/papers/dogcat.pdf)."

몇 년 전만 하더라도 정확도 60%조차 달성하기 어려울 것이라고 전문가들은 단언했지만, 이제는 80% 정확도 정도 도달할 수 있을 것이라는 내용입니다.

당시에 집계된 [참가자 순위표](https://www.kaggle.com/c/dogs-vs-cats/leaderboard)를 살펴보시면 1등 팀이 거의 99% 정확도를 달성했습니다. 저희가 사용할 데이터는 캐글 데이터의 1/12 수준인 데다가 이미지 해상도 또한 정말 작습니다. 같은 성능을 내는 건 아무래도 쉽지 않겠죠.

# 데이터가 적은 문제에서 딥러닝 적용하기

저는 이런 말을 자주 듣게 됩니다: “deep learning is only relevant when you have a huge amount of data”. 데이터가 정말 많아야 딥러닝을 해볼 만하다는 것입니다. 전혀 엉뚱한 소리는 아닙니다. 컴퓨터가 알아서 데이터의 특성을 파악하려면 일반적으로 학습할 수 있는 데이터가 많아야 합니다. 이미지처럼 차원이 높은 데이터, 즉 복잡한 데이터는 더더욱 그렇습니다. 하지만 대표적인 딥러닝 모델인 CNN은 바로 이런 문제를 해결하기 위해 설계된 모델입니다—학습한 데이터가 적은 경우라도 말이죠. 별도의 데이터 조작 없이 적은 데이터를 가지고도 간단한 CNN을 처음부터 학습시켜보면 괜찮은 성능이 나오는 것을 확인할 수  있을 것입니다.

하지만 이 글에서 다루는 딥러닝 모델의 핵심 장점은 바로 재사용성입니다. 예를 들어 매우 큰 데이터셋을 가지고 학습된 이미지 분류 또는 음성인식 모델은, 조금만 수정을 가하면 아예 다른 문제 상황에서도 사용할 수 있게 됩니다. 더구나 최근에는 학습이 완료된 이미지 분류 모델이 많이 공개되어 누구나 이용할 수 있습니다. 이를 활용하면 아주 적은 데이터를 가지고도 강력한 이미지 분류 모델을 손쉽게 만들 수 있습니다.
