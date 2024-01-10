import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

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


plt.scatter(perch_length, perch_weight) # 값 불러오기
plt.xlabel("length") # x축 이름 지정 
plt.ylabel("weigh") # y축 이름 지정
plt.show()

# 훈련 세트 테스트 세트로 나누고 섞기
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42) 

#배열 확인 (4, )
test_array = np.array([1,2,3,4]) 
print(test_array.shape)

# .reshape: 바꾸려는 배열의 크기 지정 4, -> 2,2
test_array = test_array.reshape(2,2) 
print(test_array.shape)

# 크기 -1 지정: 나머지 원소 개수로 모두 채우기 -> 배열의 전체 원소 개수 외우지 않아도 됨
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)
print(train_input.shape, test_input.shape)

# k-최근접 이웃 회귀 모델 훈련
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target)) # 결정 계수 값(0.99....)

# 결정 계수 coefficient of determination R^2
# R^2 = 1 -  (target - predict)^2 / (target - mean)^2

# 테스트 평균 절댓값 오차 구하기 (mean absolute error)
test_prediction = knr.predict(test_input) #테스트 세트 예측
mae = mean_absolute_error(test_target, test_prediction) #테스트 절댓값 오차 계산
print(mae)

# 과대적합(overfitting) vs 과소적합(underfitting)
# overfitting = train set > test set 
# underfitting = train set < test set OR both low 
print(knr.score(train_input, train_target)) #0.96...
print(knr.score(test_input, test_target)) #0.99...

# overfitting solution: 이웃 개수 늘리기
# underfitting solution: 이웃 개수 줄이기
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target)) # 0.98...
print(knr.score(test_input, test_target)) # 0.97...
