import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#데이터 불러오기
fish =  pd.read_csv('https://bit.ly/fish_csv')
fish.head()
print(pd.unique(fish['Species']))

#입력 데이터 지정
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])

#타겟 데이터 지정
fish_target = fish['Species'].to_numpy()

#훈련 테스트 데이터 나누고 섞기
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42) 

#데이터 표준화 전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#K-NN Classify
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target)) #0.89...
print(kn.score(test_scaled, test_target)) #0.85

#다중 분류 multi-class classification
print(kn.classes_) #문자열로 된 타깃값 그대로 사용 가능하나 알파벳 순으로 바뀜
print(kn.predict(test_scaled[:5])) # 테스트 샘플 첫 5개 타깃값 예측

#테스트 샘플 확률 예측
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

#최근접 이웃 클래스 확인
distances, indexes = kn.kneighbors(test_scaled[3:4]) # 4번째 샘플만 확인
print(train_target[indexes]) 

#로지스틱 회귀 Logistic Regression
#시그모이드 함수 Sigmoid Function 
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z,phi)
plt.show()
 
#로지스틱 회귀로 이진 분류 수행
#불리언 인덱싱 Boolean Indexing: True, False로 행 선택
#예시
char_arr = np.array(['A', 'B', 'C', 'D', 'E']) 
print(char_arr[[True,False,True,False,False]]) # 'A' , 'C'

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

#로지스틱 회귀 모델 훈련
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))
print(lr.coef_, lr.intercept_) #계수 확인

#z값 출력
z_value = lr.decision_function(train_bream_smelt[:5])
print(z_value)

#시그모이드 함수 라이브러리
from scipy.special import expit
print(expit(z_value)) # predict_proba 2번째 열과 값이 동일

#로지스틱 회귀 다중 분류 수행
lr = LogisticRegression(C=20, max_iter=1000) # C: 규제 제어 파라미터(작을수록 규제가 커짐 기본값 1) max_iter: 반복 횟수
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target)) #0,93...
print(lr.score(test_scaled, test_target)) #0.925
print(lr.predict(test_scaled[:5])) # 테스트 세트 처음 5개

proba = lr.predict_proba(test_scaled[:5]) # 5개 확률 예측
print(np.round(proba, decimals=3))

# 선형 방정식 크기 출력
print(lr.coef_.shape, lr.intercept_.shape) #(7, 5) (7,) 데이터 특성 5개(열) z값 7개(행)

#소프트맥스 함수: 다중 z값을 확률로 변환
# 모든 z값에 대하여 e^z(x) 을 계산 후 더함 / 각각 exp 값을 모두 더한 값으로 나눔 / 총합은 1

#각각의 z값 구하기
z_values = lr.decision_function(test_scaled[:5])
print(np.round(z_values, decimals=2)) 

#소프트맥수 함수 구하기
from scipy.special import softmax
proba = softmax(z_values, axis=1)
print(np.round(z_values,decimals=3))