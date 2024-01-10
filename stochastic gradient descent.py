#확률적 경사 하강법
#무작위로 선택된 하나의 데이터 포인트(또는 작은 배치)를 사용하여 경사를 조금씩 내려감 
#에포크 epoch: 전체 훈련 데이터셋에 대해 학습이 완료된 각 반복을 의미
#손실함수 loss function: 알고리즘이 얼마나 잘못되었는지 나타내는 척도. 손실 함수는 미분 가능한 연속된 함수여야 한다.
#로지스틱 손실 함수 Logistic loss fuction /이진 교차 엔트로피 손실 함수 Binary Cross-entropy loss function
#양성 클래스(1) 손실: -log(예측 확률) -> 1에서 멀어질수록 손실은 아주 큰 양수가 됨.
#음성 클래스(0) 손실: -log(1-예측 확률) -> 0에서 멀어질수록 손실은 아주 큰 양수가 됨.

#SGD Classifier

#데이터 프레임 생성
import pandas as pd

fish =  pd.read_csv('https://bit.ly/fish_csv')

#입력 데이터 지정
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

#타겟 데이터 지정
fish_target = fish['Species'].to_numpy()

#훈련 테스트 데이터 나누고 섞기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42) 

#데이터 표준화 전처리
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#SGD Class
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) #0.773...
print(sc.score(test_scaled, test_target)) #0.775

#점진적 학습
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) #0.815...
print(sc.score(test_scaled, test_target)) #0.85 

#조기 종료 Early Stopping: 에포크가 진행될수록 훈련 세트 점수는 증가하지만 테스트 세트는 일정 횟수를 기점으로 감소(과대적합 시작되는 곳)

#에포크당 세트 점수 기록
import numpy as np

sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target) # 생선 목록 만들기

#훈련 반복 진행
for _ in range(0,300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
    
# 훈련 데이터 그래프 그리기
import matplotlib.pyplot as plt
plt.plot(test_score)
plt.plot(train_score)
plt.show() #100번째 이후로 격차가 벌어지므로 에포크 100으로 맞추고 다시 수행

#다시 진행
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
# tol: 허용 오차 tolerance 알고리즘이 수렴 조건을 만족하는지 판단하는 데 사용되는 기준입니다. 
# 수렴은 알고리즘이 최적의 결과에 충분히 가까워졌다고 판단되어 더 이상의 학습이 필요 없을 때를 의미
#tol=None:알고리즘은 내부의 다른 중단 조건(최대 반복 횟수 max_iter)에 도달할 때까지 학습을 계속   
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) #0.957...
print(sc.score(test_scaled, test_target)) #0.925
