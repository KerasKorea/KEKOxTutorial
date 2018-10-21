# 케라스와 함께하는 하이퍼 파라미터 최적화
딥러닝 모델에 적합한 하이퍼 파라미터를 찾는 것은 지루한 과정일 수 있습니다. 그럴 필요가 없어요~

![i](https://cdn-images-1.medium.com/max/800/1*6TywHHgv7Gqmmcg0TiJmFg.gif )

<h2>TL;DR</h2>

적절한 절차를 밟는다면, 주어진 예측 테스크에서 예술적인 하이퍼 파라미터를 찾는 것은 어렵지 않을 것입니다. 3가지 접근방법(**매뉴얼, 기계의 도움, 알고리즘적**)이 있지만, 이 글에서는 기계의 도움을 받는 방법에 초점을 맞출 것 입니다. 이 글에서는 내가 어떻게 하였는지 살펴보고, 그 방법이 효과가 있다는 것을 증명하고, 그리고 이것이 왜 작동하는지 이해할 수 있을 것입니다. 메인 규칙은 아주 간단합니다.

## 성능에 대하여
성능에 관한 첫번째 점은 정확성 문제에 관한 것으로써 측정 가능한 모델의 성능의 방법에 관한 문제입니다. F1(*자동차 경주*)점수와 같은 예시를 보면, 1%로 이기고 99%로 지는 예측을 한다면, 0%로 이기는 것이 f1 점수에 정확해집니다.이것은 f1 점수가 "모두 0", "모두 1", "no true positives"과 같은 코너 케이스에 특별한 처리를 하면 조절이 가능합니다. 그러나 이것은 큰 분야이며, 이 글의 범위가 아닙니다.그래서 저는 이 문제가 하이퍼 파라미터 최적화의 중요한 부분이라는 것을 이야기 하고 싶습니다.우리는 이 분야에서 아주 많은 연구를 하고 있지만, 그 연구는 기본적인 내용보다는 알고리즘에 초점을 맞추고 있습니다. 실제로, 당신이 세상에서 성능이 좋은(*종종 매우 복잡한*) 알고리즘을 찾을 수도 있습니다. 이 알고리즘은 말이 되지 않으며, 그것은 "실제 세계" 문제를 다루는데 큰 도움이 되지 않을 것입니다.

실수를 만들지 마세요; 
**성능 메트릭을 정확하게 파악하더라도**
(소리지르는중). 모델을 최적화하는 과정에서 어떤 일이 일어나는지 고려해야 합니다. 우리는 학습 데이터도, 평가 데이터도 있습니다. 검증 졀과를 검토하고, 이를 기반으로 학습시킨다면, 우리는 검증 데이터에 대한 훈련결과를 갖게됩니다. 이제 우리는 머신의 바이어스의 곱인 학습 결과와 예측 결과가 있습니다. 다시말해, 우리가 얻은 모델은 잘 알려진 특성이 없으며, 일반화하는 것에 비중이 가있습니다. 그래서 이 점을 명심하는 것이 매우 중요합니다.

하이퍼 파라미터 최적화에 대한 더 진화된 완전 자동(비지도) 접근 방식의 핵심은 이러한 두가지 문제를 해결하는 것입니다. 이 두가지가 해결되면(*그렇게 할 수 있는 방법이 있어요!*) 결과 메트릭스는 단일 점수로 구현해야 합니다.그런 다음 이 점수는 하이퍼 파라미터 최적화 프로세스가 최적화되는데 메트릭이 사용됩니다.그렇지 않으면, 세계의 어떤 알고리즘도 도움이 되지 않을 것입니다. 왜냐하면 우리가 추구하는 것(성능) 이외의 어떤 것에 최적화 될 것이기 때문입니다. 우리는 또 뭘 할 수 있을까요? 예측 작업이 구체화 되는 테스크를 수행할 모델입니다. 하나의 사례(주로 주제를 다루는 논문의 경우)에만 해당되는 모델이 아니라 모든 종류의 예측 작업에 대한 모든 종류의 모델입니다.그것이 바로 Keras와 같은 솔루션이 우리가 할 수 있는 것입니다. 그리고 Keras와 같은 도구를 사용하는 과정의 일부를 자동화하려는 시도는 그 아이디어를 수용해야합니다.

## 어떤 도구를 사용했을까요?

이 기사의 모든 내용에서 모델에 Keras를 사용했고, 제가 만든 하이퍼 매개 변수 최적화 솔루션 인 Talos를 사용했습니다. 새로운 구문을 도입하지 않고 기존의 Keras를 그대로 사용하는 것이 이점입니다. 그것은 고통스러운 반복 대신 재미있게 며칠이 걸렸던 일을 몇 분 안에 할 수 있습니다.

여러분이 직접 사용해 볼수 있습니다.

`pip install talos`

또는 [이곳](https://github.com/autonomio/talos)에서 문서와 코드를 볼 수 있습니다.

하지만 제가 공유하고자 하는 내용과 제가 만들고자 하는 특징은 위의 도구와 관련이 없습니다. 여러분은 여러분이 좋아하는 방식으로 동일한 절차를 따를 수 있습니다.

자동화 된 하이퍼 매개 변수 최적화 및 관련 도구를 사용하는 것에서 가장 중요한 문제 중 하나는 일반적으로 익숙한 방식에서 점점 멀어져가는 경향이 있다는 것입니다. (모든 복잡한 문제가 그렇듯이) 하이퍼 매개 변후 최적화를 성곡적으로 예측하는 핵심 키는 인간과 기계 간의 협력을 포용하는 데 있습니다. 모든 실험은 (딥러닝의) 연습과 기술 (이 경우 Keras)에 대해 더 많이 배울 수있는 기회입니다. 프로세스 자동화를 통하여 그 기회를 놓치지 마세요. 동시에 우리는 그 과정에서 노골적으로 중복되는 부분을 제거할 수 있어야 합니다. Jupyter에서 shift-enter를 몇 백 번하고 각 반복 사이를 1 ~ 2 분씩 기다리는 것을 생각해 보세요. 요약하자면, 이 시점에서 목표는 올바른 모델을 찾는 완벽하게 자동화된 접근방식이 아니라, 인간에게 부담이 되는 절차적 중복을 최소화 하는 것이어야 합니다. 기계는 기계적으로 작동하지 않고 스스로 작동합니다. 다양한 모델 구성의 결과를 하나씩 분석하는 대신 수천 개 또는 수십만 개씩 분석하고자 합니다. 하루에 80,000초 이상이 소요되며, 그 시간에 제가 관여할 필요없이 수많은 매개 변수를 다룰 수 있습니다. 

## Let’s Get Scannin’

예를들어, 저는 먼저 이 기사에서 다룬 실험 동안 사용한 코드를 제공할 것입니다. 제가 사용한 데이터셋은 [위스콘신 유방암](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) 데이터 세트입니다.

```python
# first we have to make sure to input data and params into the function
def breast_cancer_model(x_train, y_train, x_val, y_val, params):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    model.add(Dense(10, input_dim=x_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer='normal'))
    
    model.add(Dropout(params['dropout']))
    
    # if we want to also test for number of layers and shapes, that's possible
    hidden_layers(model, params, 1)
   
    # then we finish again with completely standard Keras way
    model.add(Dense(1, activation=params['last_activation'],
                    kernel_initializer='normal'))
    
    model.compile(loss=params['losses'],
                  # here we add a regulizer normalization function from Talos
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  metrics=['acc', fmeasure])
    
    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)
    
    # finally we have to make sure that history object and model are returned
    return history, model
```

케라스 모델이 정의되면 초기 매개변수 경계를 결정해야 합니다. 그런 다음 사전은 단일 순열이 한 번 선택되고 선택된 순열을 무시되는 방식으로 프로세스에 공급됩니다.

```python
# then we can go ahead and set the parameter space
p = {'lr': (0.5, 5, 10),
     'first_neuron':[4, 8, 16, 32, 64],
     'hidden_layers':[0, 1, 2],
     'batch_size': (2, 30, 10),
     'epochs': [150],
     'dropout': (0, 0.5, 5),
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shape':['brick','long_funnel'],
     'optimizer': [Adam, Nadam, RMSprop],
     'losses': [logcosh, binary_crossentropy],
     'activation':[relu, elu],
     'last_activation': [sigmoid]}
```

손실, 최적화 도구 및 스캔에 포함하려는 활성화함수에 따라 먼저 케라스에서 해당 함수/클래스를 가져와야 합니다. 이제 모델과 파라미터가 준비되었으니 이제 실험을 시작할 때입니다.

```python
# and run the experiment
t = ta.Scan(x=x,
            y=y,
            model=breast_cancer_model,
            grid_downsample=0.01, 
            params=p,
            dataset_name='breast_cancer',
            experiment_no='1')
```

다음 섹션에서 제공하는 통찰력과 관련된 파라미터 사전의 파라미터만 변경했기 때문에 더 이상의 코드를 공유하지는 않을 것입니다. 완벽을 기하기 위해 기사의 끝에 저는 코드와 함께 노트에 대한 링크를 공유하려고 합니다.

이 실험의 첫 번째 라운드에는 많은 순열이 (총 180,000개 이상의) 있기 때문에, 저는 무작위로 전체 중 1%만을 뽑았습니다. 그리고 1,800개의 순열을 남깁니다.

![i](https://cdn-images-1.medium.com/max/800/1*CMOICFCpP-3bbuNWTTJ_BA.png)

![i](https://cdn-images-1.medium.com/max/800/1*ZB-qBLNXY5RVIc9uEN5gYw.gif)

이 경우, 나는 2015년에 생산된 MacBook Air를 통하여 실행했으며, 위의 시간은 제가 친구를 만나 커피 한 잔 (또는 두 잔)을 마실 수있는 시간처럼 보입니다.

## Hyperparameter Scanning Visualized

이 기사에서는 위스콘신 유방암 데이터 세트를 사용였고, 최적의 매개 변수 또는 데이터 세트에 대한 사전 지식이없는 것으로 가정하여 실험을 진행하였습니다. 하나의 열을 삭제하고 나머지를 모두 변형하여 각 피쳐의 평균이 0이고 표준 편차가 1이되도록 데이터 세트를 준비했습니다.

1,800번의 순열을 처음 실행한 후, 결과를 살펴보고 매개변수 공간을 어떻게 제한(또는 변경)할지 결정해야 합니다.

![i](https://cdn-images-1.medium.com/max/800/1*yDrJuNQUgEcb8_L2F9ve-g.png)

간단한 순위 순서 상관 관계에서 lr(학습 속도)가 성능 메트릭에 가장 큰 영향을 미친다는 것을 알 수 있습니다. 이 경우 성능 지표는 val_ac(유효성 정확도)입니다. 이 데이터셋의 경우 Positive(양) 수가 많기 때문에 val_ac가 정상입니다.

![i](https://cdn-images-1.medium.com/max/1000/1*XgAAoiQ14z8vz-_PMvEVIA.png)

드롭아웃을 확인하는 또 다른 방법은 커널 밀도 평가입니다. 여기에서는 드롭 아웃 0 또는 0.1 및 val_acc (0.6 마크 전후)에서  val_acc이 약간 더 높아지는 경향을 확인할 수 있습니다. 

![i](https://cdn-images-1.medium.com/max/800/1*beQLnaRNZv35EGWrcuXSjA.png)

다음 스캔의 첫 번째 작업 항목은 높은 드롭아웃 비율을 모두 제거하고 0에서 0.2 사이의 값에 초점을 맞추는 것입니다. 다음으로 learning rate를 더 자세히 살펴 보겠습니다. learning rate는 최적화 프로그램에서 하나의 스케일로 정규화됩니다. 여기서 1은 해당 최적화 알고리즘의 Keras 기본값을 나타냅니다.

![i](https://cdn-images-1.medium.com/max/800/1*7S0ppxrlH-r8Q9i8ILOW6A.png)

