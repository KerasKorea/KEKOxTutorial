'''Keras를 이용해 신경망으로 낙서하기

# 본 스크립트 사용법

## Arguments
```
--nlabels:              mask 이미지안에 영역(색깔)의 수
--style-image:          학습하고 싶은 스타일이 있는 이미지
--style-mask:           스타일 이미지를 위한 시맨틱 라벨(semantic labels)
--target-mask:          목표 이미지를 위한 시맨틱 라벨(사용자가 만든 낙서)
--content-image:        (선택) 학습하고 싶은 컨텐츠가 있는 이미지
--target-image-prefix:  생성된 목표 이미지를 위한 저장 경로
```

## Example 1: 스타일 이미지, 스타일 라벨 그리고 목표 시맨틱 라벨을 사용해 낙서하기
```
python neural_doodle.py --nlabels 4 --style-image Monet/style.png \
--style-mask Monet/style_mask.png --target-mask Monet/target_mask.png \
--target-image-prefix generated/monet
```

## Example 2: 스타일 이미지, 스타일 라벨, 목표 시맨틱 라벨 그리고 옵션인 컨텐츠 이미지를 사용해 낙서하기
```
python neural_doodle.py --nlabels 4 --st를yle-image Renoir/style.png \
--style-mask Renoir/style_mask.png --target-mask Renoir/target_mask.png \
--content-image Renoir/creek.jpg \
--target-image-prefix generated/renoir
```

# 참고자료

- [Dmitry Ulyanov's blog on fast-neural-doodle]
    (http://dmitryulyanov.github.io/feed-forward-neural-doodle/)
- [Torch code for fast-neural-doodle]
    (https://github.com/DmitryUlyanov/fast-neural-doodle)
- [Torch code for online-neural-doodle]
    (https://github.com/DmitryUlyanov/online-neural-doodle)
- [Paper Texture Networks: Feed-forward Synthesis of Textures and Stylized Images]
    (http://arxiv.org/abs/1603.03417)
- [Discussion on parameter tuning]
    (https://github.com/keras-team/keras/issues/3705)

# 소스코드 자료

Example images can be downloaded from
https://github.com/DmitryUlyanov/fast-neural-doodle/tree/master/data
'''
from __future__ import print_function
import time
import argparse
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from keras import backend as K
from keras.layers import Input, AveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications import vgg19

# 커맨드 라인에 입력할 설정들
parser = argparse.ArgumentParser(description='Keras neural doodle example')
parser.add_argument('--nlabels', type=int,
                    help='number of semantic labels'
                    ' (regions in differnet colors)'
                    ' in style_mask/target_mask')
parser.add_argument('--style-image', type=str,
                    help='path to image to learn style from')
parser.add_argument('--style-mask', type=str,
                    help='path to semantic mask of style image')
parser.add_argument('--target-mask', type=str,
                    help='path to semantic mask of target image')
parser.add_argument('--content-image', type=str, default=None,
                    help='path to optional content image')
parser.add_argument('--target-image-prefix', type=str,
                    help='path prefix for generated results')
args = parser.parse_args()

style_img_path = args.style_image
style_mask_path = args.style_mask
target_mask_path = args.target_mask
content_img_path = args.content_image
target_img_prefix = args.target_image_prefix
use_content_img = content_img_path is not None

num_labels = args.nlabels
num_colors = 3  # RGB
# 목표 시맨틱 라벨을 기반으로 이미지 사이즈를 결정
ref_img = img_to_array(load_img(target_mask_path))
img_nrows, img_ncols = ref_img.shape[:2]

total_variation_weight = 50.
style_weight = 1.
content_weight = 0.1 if use_content_img else 0

content_feature_layers = ['block5_conv2']
# 생성이 더 잘되도록, 스타일 피쳐(style feature)용 컨볼루션 레이어를 더 많이 사용합니다.
style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']


# 이미지를 읽고 사전처리를 해주는 함수
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def kmeans(xs, k):
    assert xs.ndim == 2
    try:
        from sklearn.cluster import k_means
        _, labels, _ = k_means(xs.astype('float64'), k)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(xs, k, missing='raise')
    return labels


def load_mask_labels():
    '''목표 및 스타일 시맨틱 라벨을 불러오기
        m개의 라벨/색을 지닌 시맨틱 이미지(nr x nc)를 4D boolean 텐서로 불러오기.
    주의 : 
        (1, m, nr, nc)는 'channels_first'에, (1, nr, nc, m)는 'channels_last'에 사용되는 형태
    '''
    target_mask_img = load_img(target_mask_path,
                               target_size=(img_nrows, img_ncols))
    target_mask_img = img_to_array(target_mask_img)
    style_mask_img = load_img(style_mask_path,
                              target_size=(img_nrows, img_ncols))
    style_mask_img = img_to_array(style_mask_img)
    if K.image_data_format() == 'channels_first':
        mask_vecs = np.vstack([style_mask_img.reshape((3, -1)).T,
                               target_mask_img.reshape((3, -1)).T])
    else:
        mask_vecs = np.vstack([style_mask_img.reshape((-1, 3)),
                               target_mask_img.reshape((-1, 3))])

    labels = kmeans(mask_vecs, num_labels)
    style_mask_label = labels[:img_nrows *
                              img_ncols].reshape((img_nrows, img_ncols))
    target_mask_label = labels[img_nrows *
                               img_ncols:].reshape((img_nrows, img_ncols))

    stack_axis = 0 if K.image_data_format() == 'channels_first' else -1
    style_mask = np.stack([style_mask_label == r for r in range(num_labels)],
                          axis=stack_axis)
    target_mask = np.stack([target_mask_label == r for r in range(num_labels)],
                           axis=stack_axis)

    return (np.expand_dims(style_mask, axis=0),
            np.expand_dims(target_mask, axis=0))


# 이미지를 위해 텐서형 변수들을 생성합니다.
if K.image_data_format() == 'channels_first':
    shape = (1, num_colors, img_nrows, img_ncols)
else:
    shape = (1, img_nrows, img_ncols, num_colors)

style_image = K.variable(preprocess_image(style_img_path))
target_image = K.placeholder(shape=shape)
if use_content_img:
    content_image = K.variable(preprocess_image(content_img_path))
else:
    content_image = K.zeros(shape=shape)

images = K.concatenate([style_image, target_image, content_image], axis=0)

# 시맨틱 라벨을 위해 텐서형 변수들을 생성합니다.
raw_style_mask, raw_target_mask = load_mask_labels()
style_mask = K.variable(raw_style_mask.astype('float32'))
target_mask = K.variable(raw_target_mask.astype('float32'))
masks = K.concatenate([style_mask, target_mask], axis=0)

# 이미지와 작업용 변수인 인덱스 상수들
STYLE, TARGET, CONTENT = 0, 1, 2

# 이미지 모델(image_model), 시맨틱 모델(mask_input)을 생성하고 피쳐용으로 레이어 출력값을 사용
# 이미지 모델은 VGG19
image_model = vgg19.VGG19(include_top=False, input_tensor=images)

# 나열된 풀링 레이어으로 시맨틱 모델 표현합니다.
mask_input = Input(tensor=masks, shape=(None, None, None), name='mask_input')
x = mask_input
for layer in image_model.layers[1:]:
    name = 'mask_%s' % layer.name
    if 'conv' in layer.name:
        x = AveragePooling2D((3, 3), padding='same', strides=(
            1, 1), name=name)(x)
    elif 'pool' in layer.name:
        x = AveragePooling2D((2, 2), name=name)(x)
mask_model = Model(mask_input, x)

# 이미지 모델과 시맨틱 모델에서 피쳐들을 수집합니다.
image_features = {}
mask_features = {}
for img_layer, mask_layer in zip(image_model.layers, mask_model.layers):
    if 'conv' in img_layer.name:
        assert 'mask_' + img_layer.name == mask_layer.name
        layer_name = img_layer.name
        img_feat, mask_feat = img_layer.output, mask_layer.output
        image_features[layer_name] = img_feat
        mask_features[layer_name] = mask_feat


# 손실 함수를 정의합니다.
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram


def region_style_loss(style_image, target_image, style_mask, target_mask):
    '''
    (boolean형) 마스크로 지정된 하나의 공통 영역에 대해
    스타일 이미지와 목표 이미지 사이의 스타일 손실값을 계산합니다.
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 2 == K.ndim(style_mask) == K.ndim(target_mask)
    if K.image_data_format() == 'channels_first':
        masked_style = style_image * style_mask
        masked_target = target_image * target_mask
        num_channels = K.shape(style_image)[0]
    else:
        masked_style = K.permute_dimensions(
            style_image, (2, 0, 1)) * style_mask
        masked_target = K.permute_dimensions(
            target_image, (2, 0, 1)) * target_mask
        num_channels = K.shape(style_image)[-1]
    num_channels = K.cast(num_channels, dtype='float32')
    s = gram_matrix(masked_style) / K.mean(style_mask) / num_channels
    c = gram_matrix(masked_target) / K.mean(target_mask) / num_channels
    return K.mean(K.square(s - c))


def style_loss(style_image, target_image, style_masks, target_masks):
    '''
    모든 영역에서 스타일 이미지와 목표 이미지 사이의 스타일 손실값을 계산
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 3 == K.ndim(style_masks) == K.ndim(target_masks)
    loss = K.variable(0)
    for i in range(num_labels):
        if K.image_data_format() == 'channels_first':
            style_mask = style_masks[i, :, :]
            target_mask = target_masks[i, :, :]
        else:
            style_mask = style_masks[:, :, i]
            target_mask = target_masks[:, :, i]
        loss += region_style_loss(style_image,
                                  target_image, style_mask, target_mask)
    return loss


def content_loss(content_image, target_image):
    return K.sum(K.square(target_image - content_image))


def total_variation_loss(x):
    assert 4 == K.ndim(x)
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 전체 손실값은 컨텐츠, 스타일, 전체 변화 손실의 가중치를 계산한 합산.
# 각 개별 손실 함수는 이미지/시맨틱 모델로부터 추출한 피쳐들을 사용합니다.
loss = K.variable(0)
for layer in content_feature_layers:
    content_feat = image_features[layer][CONTENT, :, :, :]
    target_feat = image_features[layer][TARGET, :, :, :]
    loss += content_weight * content_loss(content_feat, target_feat)

for layer in style_feature_layers:
    style_feat = image_features[layer][STYLE, :, :, :]
    target_feat = image_features[layer][TARGET, :, :, :]
    style_masks = mask_features[layer][STYLE, :, :, :]
    target_masks = mask_features[layer][TARGET, :, :, :]
    sl = style_loss(style_feat, target_feat, style_masks, target_masks)
    loss += (style_weight / len(style_feature_layers)) * sl

loss += total_variation_weight * total_variation_loss(target_image)
loss_grads = K.gradients(loss, target_image)

# 효율성 검사를 위한 평가 클래스
outputs = [loss]
if isinstance(loss_grads, (list, tuple)):
    outputs += loss_grads
else:
    outputs.append(loss_grads)

f_outputs = K.function([target_image], outputs)


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

# 반복 최적화로 이미지를 생성합니다.
if K.image_data_format() == 'channels_first':
    x = np.random.uniform(0, 255, (1, 3, img_nrows, img_ncols)) - 128.
else:
    x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.

for i in range(50):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # 현재 생성된 이미지를 저장합니다.
    img = deprocess_image(x.copy())
    fname = target_img_prefix + '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))