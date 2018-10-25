'''케라스로 신경 스타일로 바꾸기

현 스크립트를 실행하기 위해선:
```
python neural_style_transfer.py path_to_your_base_image.jpg \
    path_to_your_reference.jpg prefix_for_results
```
예 :
```
python neural_style_transfer.py img/tuebingen.jpg \
    img/starry_night.jpg results/my_result
```
추가 파라미터들은 :
```
--iter, 스타일 변화를 수행하는 반복횟수를 지정함.(고정값은 10)
--content_weight, 컨텐츠 손실에 대한 가중치 (고정값은 0.025)
--style_weight, 스타일 손실에 대한 가중치 (고정값은 1.0)
--tv_weight, 전체 변화 손실에 대한 가중치 (고정값은 1.0)
```

속도를 위해, GPU에서 현 스크립트를 실행하길 권합니다.

예시에 대한 결과 : https://twitter.com/fchollet/status/686631033085677568

# 상세설명

스타일 변형은 기존 이미지와 동일한 '콘텐츠'를 사용하여
이미지를 생성하지만 다른 이미지의 '스타일'을 갖도록 생성합니다.

손실 함수의 최적화를 통해 얻을 수 있으며, 
이는 3가지 요소를 가집니다.: "style loss", "content loss", "total variation loss"

 - 전체 변화 손실(total variation loss)은 조합 이미지의 픽셀들 사이에
local 공간 연속성을 부과하여 시각적 일관화를 제공합니다. 

 - 스타일 손실(The style loss)은 딥러닝이 시행되는 구간으로, 이는 deep CNN을 사용하여
  정의합니다. 정확히는, (ImageNet으로 훈련된) 다른 계층들에서 추출된, 
  기본 이미지의 표현값들의 Gram matrix와 스타일 기준 이미지간의 L2 거리의 합으로 구성됩니다. 
   일반적인 아이디어는 색상/질감 정보를 다양한 공간적 척도(scale)
   (꽤나 큰 구조 -- 언급된 계층의 깊이로 정의)로 수집하는 겁니다.

 - 콘텐츠 손실(The content loss)은 기본 이미지 특징과 결합 이미지 간의 L2 거리로,
 생성된 이미지가 원본에 가까워 지도록 해줍니다.

# 참고 자료
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
'''

from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

# 서로 다른 손실 요소들의 가중치들을 가져옵니다.
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# 생성할 이미지의 차원을 설정합니다.
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

# 아래 함수는 이미지를 열어서 크기를 조정하고 적절한 tensor로 format시킵니다.

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# 아래 함수는 tensor를 검증 이미지로 전환시킵니다.

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # 평균 픽셀을 기준으로 zero중심 제거합니다.
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 주어진 이미지를 tensor형식으로 갖습니다.
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# 생성될 이미지를 갖도록 설정합니다.
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# 3가지 이미지를 1개의 keras tensor로 결합시킵니다.
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

# (갖고있는)3가지 이미지를 입력으로하는 VGG16 네트워크를 구성합니다.
# 해당 model은 미리 ImageNet으로 학습된 가중치를 갖습니다.
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('Model loaded.')

# 각 핵십 계층의 상징적 결과들을 갖습니다(그 결과들에 이름을 지정합니다.)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# 신경 스타일 손실을 계산합니다.
# 먼저 4가지 함수들을 정의할 필요가 있습니다.

# 이미지 tensor의 gram matrix(feature-wise 외적연산)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# 스타일 손실은 생성된 이미지에서 기존 이미지의 스타일을 유지하도록 설계됩니다.
# 스타일 기준 이미지와 생성된 이미지에서 특징에 대한 gram matrix(스타일 수집)를 기반으로 합니다.
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# 보조 손실 함수는 생성된 이미지에서 
# 기존 이미지의 '콘텐츠'를 유지하도록 설계되었습니다.
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# 3번째 손실함수인 전체 변화 손실(total variation loss)은 
# 생성된 이미지를 지역적으로 일관성있게 유지되도록 설계되었습니다.
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# 이런 손실 함수들을 단일 숫자(스칼라)로 결합시킵니다.
loss = K.variable(0.)
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

# 손실에 의해 생성된 이미지의 기울기를 갖습니다.
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# Evaluator 클래스는 '손실'과 '기울기'라는 2가지 별도 함수들을 통해 
# 한번으로 손실과 기울기를 검색하는 동시에 계산을 할 수 있도록 합니다.
# 왜 그렇게 진행했냐면 scipy.optimize는 손실과 기울기에 대한 별도의 함수를
# 요구하지만 따로 계산할 경우 비효율적일수 있기 때문입니다.
class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# 신결 스타일 손실을 최소화하기 위해
# 생성된 이미지 픽셀들을 scipy기반으로 최적화(L-BFGS)합니다.
x = preprocess_image(base_image_path)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # 현재 생성된 이미지를 저장합니다.
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))