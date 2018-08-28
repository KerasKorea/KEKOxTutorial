## Keras와 텐서플로우를 이용한 손글씨 인식(Handwriting recognition using Tensorflow and Keras)
[원문 링크](https://towardsdatascience.com/handwriting-recognition-using-tensorflow-and-keras-819b36148fe5)
> 이 문서는 CNN 과 softmax classification loss 를 이용해서 영어 손글씨 데이터를 인식해보는 것에 대해서 설명합니다.

* CNN
* softmax classification loss

<br></br>
![figure1](https://cdn-images-1.medium.com/max/1600/1*mCjhzOF1Gsr7eURlIIozUg.gif)

### Introduction
작가 가 직접 작성한 각 문서를 분류하는 손글씨 인식은 사람마다 글씨 스타일의 큰 차이로 인해 어려운 문제입니다.
손글씨 인식에 대한 접근 방식은 언어의 독립적인 특징(feautre)를 추출하는 것입니다. 글씨의 독립적인 특징에는
글자의 굴곡(휘어진 정도- 쉽게 말하면 글씨체라고 생각하면 될 것 같네요.), 공백, b/w 문자가 있습니다. 그러고나서 작가 별로 구분할 수 있도록 SVM 과 같은 분류기를 사용합니다. 이 글에서는, 이러한 기능을 식별하는 딥러닝 기반 접근 방식을 보여드리고자 합니다. 우리는 손으로 쓴 작은 이미지 조각들을 CNN에 전달하고 손실함수로서 softmax classification loss를 사용해서 모델을 학습시킬 것입니다.

이 기술의 효과를 보여주기 위해서, 영어 손글씨 데이터를 분류하는 데 사용하기로 해요.

[여기](https://github.com/priya-dwivedi/Deep-Learning/blob/master/handwriting_recognition/English_작가_Identification.ipynb)에서 모든 코드를 확인할 수 있습니다.

#### Getting our data (데이터에 대한 설명)

IAM 손글씨 데이터베이스는 영어 손글씨 이미지 데이터 셋 중 가장 큰 데이터베이스입니다. IAM 손글씨 데이터베이스는 600 명 이상의 작가 가 쓴 1539 페이지 이상의 글자들을 스캔한 데이터 베이스입니다. 이 데모의 목적을 달성하기 위해서 우리는 가장 데이터가 많은 50명의 작가 들의 스캔된 글자데이터를 쓸 것입니다. 각 작가 들의 데이터베이스 콜렉션은 그들이 직접 쓴 문장들로 구성되어 있습니다. 여기에 어떤 wrtier 에 대한 예시가 있습니다:

![](https://cdn-images-1.medium.com/max/1600/1*5cigjKhwPhs66tlN5fSsRQ.png)
![sample sentences](https://cdn-images-1.medium.com/max/1600/1*1DgR4SFfDr7eGZwS0ppE9g.png)

신경망은 raw 데이터에 대해 많은 전처리를 필요로 하지 않습니다. 그래서 우리는 이 이미지들을 수정하지 않을 것입니다. 하지만 전체 이미지를 신경망에 전달하는 대신에, 우리는 작은 텍스트 조각들을 전달할 것입니다.

#### Generating patches of data (패치 데이터 만들기)

우리는 신경망이 개별 글쓴이의 글씨체를 이해하기를 바라며, 우리는 이 신경 네트워크가 텍스트 독립적이기를 원합니다(어떤 언어에 대해서도 동작할 수 있도록). 그래서 우리는 개별 문장이나 단어를 전달하는 대신에 임의의 텍스트 조각들을 전달할 것입니다. 이 작업은 모든 문장에서 113x113 크기의 패치를 무작위로 잘라서 수행합니다. 아래 이미지는 이러한 패치 8개의 콜라주입니다.

![Collage of 8 patches](https://cdn-images-1.medium.com/max/1600/1*5BoEd3eQwcO3zX5CPGao0w.jpeg)
Collage of 8 patches

생성 함수를 만들어서 각 문장으로 이동하여 이미지의 랜덤 패치를 생성할 것 입니다. 모든 이미지에 대해 생성할 수 있는 패치를 전체 패치의 30%로 패치 수를 제한합니다. 이 생성 함수에 대한 자세한 내용은 [Github repo](https://github.com/priya-dwivedi/Deep-Learning/blob/master/handwriting_recognition/English_Writer_Identification.ipynb) 를 참조하십시오.

#### Model and Results (모델과 결과)

이 작업을 위해 우리는 텐서플로우 백엔드를 사용하여 케라스에서 CNN 모델을 만들 것입니다. 우리는 여러개의 convolution 레이어와 maxpool 레이어가 있는 표준 CNN, 입출력 레이어를 연결시켜주는 dense 레이어, 그리고 소프트맥스 함수가 있는 최종 출력 레이어를 사용할 것입니다. RELU 활성화는 convolution과 dense 레이어 사이에 사용되었으며 모델은 Adam Optimizer를 사용하여 옵티마이징 합니다.

모델의 크기는 데이터 크기에 비례해야 합니다. 이 문제를 해결하기 위해서는 최대 풀 레이어 3개와 조밀한 레이어 2개가 충분했습니다. 아래 모델 요약을 참조하십시오.

![CNN Model Summary](https://cdn-images-1.medium.com/max/1600/1*g5Nof5Y_TLHOCm5kAssECQ.png)
CNN Model Summary

하이퍼 파라미터 튜닝 후 학습에 전혀 사용하지 않은 테스트 데이터셋에서 94%의 손실이 발생했습니다.

모델이 동일한 작가가 썼을 것이라고 판단한 아래 두 개의 패치를 보세요. 두 개의 "t"의 모양은 매우 비슷해서 그들이 같은 작가에 속한다는 것을 직관적으로 이해할 수 있을 것입니다.

![](https://cdn-images-1.medium.com/max/1600/1*c4U9dfqW9ZGfvSZyZbxMqA.png)
![](https://cdn-images-1.medium.com/max/2000/1*qq4ChRNr4qW9Dea0qNSzlw.png)


#### Summary (요약)
딥러닝을 사용한 손글씨 인식은 여러 가지 이유로 매우 강력한 기술입니다.
* 강력한 특징들을 알아서 잡아냅니다.
* 랜덤 패치로 모델을 학습시키는 것은 우리의 접근 방식은 모델은 똑같은 작가가 다른 언어의 글자를 썼어도 인식할 수 있도록 독립적이게 만듭니다.
* 높은 예측 정확도 덕분에 실제로도 사용 가능할 것입니다.

### 참고사이트
[원문의 reference - Deep Writer for Text Independent Writer Identification](https://arxiv.org/pdf/1606.06472.pdf)
[원문의 reference - IAM Handwriting database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
[Dense layer](https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/)
[tensorflow-hangul-recognition](https://github.com/IBM/tensorflow-hangul-recognition/blob/master/README-ko.md#2-이미지-자료-생성하기)
