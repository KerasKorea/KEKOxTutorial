# GANs을 이용한 Single Image Super Resolution — Keras

원문: (Medium) [**Single Image Super Resolution Using GANs — Keras**](https://medium.com/@birla.deepak26/single-image-super-resolution-using-gans-keras-aca310f33112), Deepak Birla

## 문서 소개

> 이 문서는 Super Resolution을 기존의 보간법(interpolation)과 같은 방법이 아닌 GANs을 사용하여 성능을 올린 SRGAN의 내용과 함께 keras코드로 쉽게 설명합니다.

- Super Resolution
- GANs
- Perceptual loss


**Image Super Resolution**:
Image super resolution은 화질 저하를 최소로 유지하면서 작은 이미지의 크기를 증가시키거나, 저해상도 이미지에서 얻은 풍부한 디테일에서 고해상도 이미지를 복원하는 것으로 정의할 수 있습니다. 이 문제는 주어진 저해상도 이미지에 대한 여러 솔루션이 있기 때문에 상당히 복잡합니다. 이것은 위성 및 항공 이미지 분석, 의료 이미지 처리, 압축 이미지/비디오 강화 등과 같은 수많은 응용 프로그램을 가지고 있습니다.

**Problem Statement(문제점)**:
저해상도 이미지에서 고해상도 이미지를 복구하거나 복원하기 위해서 다음과 같이 할 수 있습니다. 노이즈 감소(noise-reduction), 업스케일링 이미지(up-scaling image) 및 색상 조정(color adjustments)을 포함하는 이미지 개선과 같은 많은 방법이 있습니다.


> 2020 새마음 새뜻으로 이든이가 해쪄욤 
> Translator: [박정현](https://github.com/parkjh688)  
> Translator email : parkjh688@gmail.com
> Translator twitter : @edensuperb
