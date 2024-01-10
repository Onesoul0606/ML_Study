import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 세트 테스트 세트로 나누고 섞기
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42) 

# 크기 -1 지정: 나머지 원소 개수로 모두 채우기 -> 배열의 전체 원소 개수 외우지 않아도 됨
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

#K-NN Regression training
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)
print(knr.predict([[50]]))

#산점도 표시
distances, indexes = knr.kneighbors([[50]]) #50cm 농어 이웃 구하기
plt.scatter(train_input, train_target) # 훈련 세트 산점도
plt.scatter(train_input[indexes], train_target[indexes], marker="D") #훈련 세트 중 이웃 샘플만 다시 그리기
plt.scatter(50,1033,marker="^") #50cm 농어
plt.show()
print(np.mean(train_target[indexes])) # 이웃 샘플 타깃의 평균(모델이 예측했던 값과 일치)

#다시 산점도 그리기
distances, indexes = knr.kneighbors([[100]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker="D")
plt.scatter(100,1033,marker="^")
plt.show()

# 선형회귀 Linear Regression
lr = LinearRegression()
lr.fit(train_input, train_target) # 선형 회귀 모델 훈련
print(lr.predict([[50]]))

# 함수에 대입? coef_, intercept_ (model parameter)
# 1차 함수 y = ax + b에서 a = coef_ b = intercept_
print(lr.coef_, lr.intercept_)

#산점도와 1차 방정식 그래프 그리기
plt.scatter(test_input, test_target) # 훈련 세트 산점도
plt.plot([15,50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_]) # 15에서 50까지 1차 방정식 그래프 그리기
plt.scatter(50,1241.8, marker="^") # 50cm 농어 데이터
plt.show()

#전반적인 과소적합?
print(lr.score(train_input,train_target)) #0.93...
print(lr.score(test_input, test_target)) #0.82...

#다항회귀 Polynomial Regression ax^2 + bx + c
#넘파이 브로드캐스팅
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
print(train_input.shape, test_poly.shape) # 데이터셋 크기 확인

#타겟값은 어떤 그래프를 그리던 변하지 않기 때문에 바꿔 줄 필요도 없다
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
print(lr.coef_, lr.intercept_)

#다항 회귀 그래프 그리기
point = np.arange(15, 50) # 구간별 직선을 위한 정수 배열
plt.scatter(train_input, train_target) # 산점도 그리기
plt.plot(point, 1.01*point**2 -21.6*point + 116.05) # 15에서 49까지 2차 방정식 그래프 그리기
plt.scatter([50], [1574], marker="^") #50cm 농어
plt.show()

# 결정 계수 평가 (좋아졌지만 아직 과소적합 남아있음)
print(lr.score(train_poly, train_target)) # 0.970....
print(lr.score(test_poly, test_target)) # 0.977...