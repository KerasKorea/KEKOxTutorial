'''케라스로 Deep Dreaming 하기
원문 : https://github.com/keras-team/keras/tree/master/examples/deep_dream.py
> 현 스크립트는 케라스를 이용해 입력 이미지의 특징들을 pareidolia 알고리즘으로 분석, 강화시켜
> 마치 꿈, 환각같은 이미지 형태로 출력해주는 튜토리얼입니다. 

* deepdream
* keras
* CNN

현 스크립트를 실행하기 위해선:
```
python deep_dream.py path_to_your_base_image.jpg prefix_for_results
```
예 :
```
python deep_dream.py img/mypic.jpg results/dream
```
'''
from __future__ import print_function

from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import scipy
import argparse

from keras.applications import inception_v3
from keras import backend as K

# 입출력 이미지 경로 설정합니다.
parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix

# 아래 코드는 마지막 손실에서 가중치와 활성화를 최대로 하는 계층들의 이름입니다
# 이제 최대화를 시도해봅시다.
# 밑의 설정들을 수정해서 새로운 시각효과를 얻어봅시다.
settings = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}


def preprocess_image(image_path):
    # 이미지들을 열어서 적절한 tensor에 resize, format 해주는 함수
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # 하나의 tensor를 검증 이미지로 변환해주는 함수
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

K.set_learning_phase(0)

# 실험하실 placehorder(입력 데이터)기반으로 InceptionV3 network를 설계합니다. 
# 해당 모델은 ImageNet으로 선행학습된 가중치를 가져올 겁니다.
model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)
dream = model.input
print('Model loaded.')

# 각 핵심 계층의 상징적인 결과를 가져옵니다(고유한 이름을 부여해야 합니다.).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# 손실을 정의합니다.
loss = K.variable(0.)
for layer_name in settings['features']:
    # 계층의 특징들에 대한 L2 norm을 손실에 추가합니다.
    assert (layer_name in layer_dict.keys(),
            'Layer ' + layer_name + ' not found in model.')
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    # 손실에서 경계부분을 제외한 픽셀만 포함시키도록 artifacts(예술작품?) 경계를 피해줍니다.
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# 손실에 대해 실제 'dream' 모델의 기울기를 계산합니다.
grads = K.gradients(loss, dream)[0]
# 기울기들을 표준화합니다
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

# 주어진 입력이미지의 기울기들과 손실 값을 검색하는 함수를 설정합니다.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


"""진행과정

- 원본 이미지를 불러옵니다.
- 아주 작은것부터 가장 큰 것까지,
- 여러가지의 처리 구조를 정의합니다(예: 이미지 형태)
- 원본 이미지를 가장작은 규모로 크기 변경합니다. 
- 모든 계층 구조를 위해, 가장 작은 단위에서 시작합니다.(예, 현재 척도):

    - 기울기 상승 진행
    - 이미지를 다음 규모로 업그레이드
    - 업그레이드시 손실된 세부정보를 재입력
- 원래 크기로 돌아갔을 때, 정지합니다.

업그레이드 동안 손실된 세부정보를 얻기 위해, 그저 원본 이미지를 가져와,
축소, 확장하고, 그 결과를 원래 (크기 변경된) 이미지와 비교합니다
"""


# 아래 하이퍼파라미터들을 사용하면 새로운 효과들을 얻을 수 있습니다.
step = 0.01  # 기울기 상승 step의 크기
num_octave = 3  # 기울기 상승을 실행할 때, 계층 구조의 수(?)
octave_scale = 1.4  # 계층들 간 비율
iterations = 20  # 계층마다 (기울기) 상승 step의 횟수
max_loss = 10.

img = preprocess_image(base_image_path)
if K.image_data_format() == 'channels_first':
    original_shape = img.shape[2:]
else:
    original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)

save_img(result_prefix + '.png', deprocess_image(np.copy(img)))