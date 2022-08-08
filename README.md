# 이모티콘 추천기 ver.2


## 프로젝트 소개 </br>
### 개요
![introduce1](./etc/%08introduce1.png)
<b> 이모티콘(emoticon) </b></br> 
문자를 이용하여 만들어낸 감정을 나타내는 기호들을 일컫는다.

최근 이모티콘이 일상 소통 방식으로 자리매김하게 되면서, 이모티콘의 중요성은 점점 증가하고있다. </br></br>
![introduce2](./etc/introduce2.png)
그러나, 이모티콘을 일일히 찾아서 구매하는 것, 상황에 맞는 이모티콘을 적재적소에 활용할 수 있도록 하는 과정은 번거로운 과정이다.

</br>
이러한 문제를 해결하기 위해 텍스트만 입력해도 이모티콘이 출력되어 쉽게 이모티콘을 사용할 수 있도록 하는 프로젝트를 진행하였다. </br></br>


### 기존 이모티콘 ver.1과의 차이점 </br>
pytorch와 pytorch ignite를 사용하여 코드의 가독성을 높이고, argparser를 이용하여 CLI환경에서도 쉽게 hyperparameter tuning할 수 있도록 하였다. </br></br>

### 프로젝트 구조 </br>
- <b>(data)</b> - 사용자가 생성한 뒤, 말뭉치를 저장하는 directory </br>
- <b>emoji</b> - 이모티콘 파일들과 입력 문장 분류 결과에 따른 이모티콘을 출력하는 기능 포함
- <b>preprocessing</b> - tokenization, shuffling, 파일 병합 등 전처리 과정 포함
- <b>train.py</b> model 학습 시 실행해야하는 파일. 실행 시 argparse를 통한 hyperparameter tuning 담당
- <b>main.py</b> 학습된 model을 바탕으로 텍스트 입력 후 이모티콘 추천이 실행되는 파일
- <b>emoji_classification</b>
    * <b>models</b> - 문장 분류 모델, fasttext 모델 등 모델 저장 directory
    * <b>init.py</b>
    * <b>loader.py</b> - torchtext 0.10.0 version 사용, train(valid) data를 model에 load
    * <b>trainer.py</b> - pytorch ignite를 사용, 문장 분류 model의 train,validation 과정 포함
    * <b>utility.py</b> - 기타 다양한 기능 포함</br>

- etc - 프로젝트 소개 파일 포함
---
</br>

## 사용 방법 
1. data 디렉토리를 생성하여 말뭉치를 저장한 뒤, preprocssing의 shuf.sh -> tokenization으로 기본적인 preprocessing 진행 (그 외의 과정은 말뭉치에 따라 사용자가 수행)
2. train.py 로 분류 모델 생성
3. main.py로 문장 분류 및 이모티콘 추천받기 실행
---
</br>

## 실행 영상
[![reslut_video](https://img.youtube.com/vi/CoZry6k7vOw/0.jpg)](https://youtu.be/CoZry6k7vOw)