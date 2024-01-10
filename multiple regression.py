#다중 회귀 Multiple Regression 평면 학습
#특성 공학 Feature Engineering: 기존의 특성을 사용해 새로운 특성을 만들어 내는 작업(농어 높이 X 농어 길이)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

#pandas 이용해서 데이터 불러오기
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 세트 테스트 세트로 나누고 섞기
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

#Transformer
poly = PolynomialFeatures()
poly.fit([[2,3]]) #훈련(fit)을 해야 변환(Transform)가능
print(poly.transform([[2,3]])) # [[1. 2. 3. 4. 6. 9.]]
#각 항을 제곱한 항 추가, 곱한 항 추가, 1은 계수 중 하나

#1을 없애는 transformer
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2,3]])
print(poly.transform([[2,3]])) # [[2. 3. 4. 6. 9.]]

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)

test_poly = poly.transform(test_input)

#다중 회귀 모델 훈련
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target)) # 0.99
print(lr.score(test_poly, test_target)) # 0.97

#특성 추가
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input) # 훈련셋 훈련
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)

lr.fit(train_poly, train_target)
print(lr.score(train_poly,train_target)) #0.9999...
print(lr.score(test_poly,test_target)) # -144 -> 특성의 개수를 늘리면 효과가 확실해지지만 그만큼 훈련셋에 아주 심한 과대적합이 되어버림

#규제 regularisation
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

#릿지 회귀 Ridge Regression: 계수를 제곱한 값을 기준으로 규제
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) # 0.98
print(ridge.score(test_scaled, test_target)) #0.97  정상화

# alpha 매개변수
# alpha 값이 크면: 규제 강도도 커지므로 계수 값을 줄이고 과소적합을 유도
# alpha 값이 작으면: 계수 줄이는 역할 줄어들고 선형회귀 모델과 유사해지므로 과대적합 가능성 높음

#hyperparameter
# alpha 값처럼 사전에 사람이 지정해줘야 하는 피라미터
#머신러닝 라이브러리에서는 클래스와 메서드의 매개변수

#alpha 값 찾기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       #ridge 모델 생성
       ridge = Ridge(alpha=alpha)
       #ridge 모델 훈련
       ridge.fit(train_scaled, train_target)
       #훈련 점수 테스트 점수 저장
       train_score.append(ridge.score(train_scaled, train_target))
       test_score.append(ridge.score(test_scaled, test_target))
       
#alpha list 그래프 그리기
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.show()

#그래프 분석하기
#파란색 석은 train 주황색 선은 test
#두 그래프가 가장 가깝고 test의 값이 가장 큰 -1 이 alpha 값 -> 10^-1 = 0.1
#따라서 alpha 값은 0.1

#최종 ridge 모델 훈련
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) #0.98
print(ridge.score(test_scaled, test_target)) # 0.97

#라쏘 회귀 Lasso Regression: 절댓값을 기준으로 규제
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target)) #0.98
print(lasso.score(test_scaled, test_target)) # 0.98

#alpha 값 찾기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       #Lasso 모델 생성
       lasso = Lasso(alpha=alpha, max_iter=10000)
       #Lasso 모델 훈련
       lasso.fit(train_scaled, train_target)
       #훈련 점수 테스트 점수 저장
       train_score.append(lasso.score(train_scaled, train_target))
       test_score.append(lasso.score(test_scaled, test_target))
       
#alpha list 그래프 그리기
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.show() #최적 값 10^1 = 10

#최종 Lasso 모델 훈련
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target)) #0.98
print(lasso.score(test_scaled, test_target)) # 0.98

#계수가 0인 값 찾기
print(np.sum(lasso.coef_ == 0)) # 40 -> 55개 넣었는데 40개 0이므로 15개만 사용됨
