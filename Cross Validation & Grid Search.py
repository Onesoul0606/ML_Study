#검증 세트 Validation Set
#Train Set 60% + Test Set 20% + Validation Set 20%

#데이터프레임 정리
import pandas as pd
import plotly as px

wine = pd.read_csv('https://bit.ly/wine-date')
data = wine[["alcohol", "sugar", "pH",]].to_numpy()
target = wine[['class']].to_numpy()

#세트 나누기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42) 

#훈련세트에서 검증 세트 나누기
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
print(sub_input.shape, val_input.shape)  

#세트 훈련
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target)) #0.997
print(dt.score(val_input,val_target)) #0.866 overfitting

#교차 검증 Cross Validation
#검증 세트를 떼어 내어 평가하는 과정을 여러 번 반복후 평균을 내어 최종 검증 점수 산출
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target) #5-fold cross validation
print(scores) #key 순서: 모델 훈련 시간, 모델 검증 시간, 교차 검증 점수. 최종 점수는 각 교차 검증 점수의 평균

#평균 구하기
import numpy as np

print(np.mean(scores['test_score'])) #0.857

#분할기 Splitter: 교차 검증시 훈련 세트 섞기
#위 과정과 동일한 코드
from sklearn.model_selection import StratifiedKFold

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))

#Splitter를 이용해서 훈련 세트 섞기
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#n_split: 폴드 개수, shuffle:데이터 분할전 섞기 cv: 교차 검증 설정, 데이터 분할 정의
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))

#하이퍼파라미터 튜닝 Hyperparameter Tuning
#동시의 모든 최적의 매개변수값을 찾아야 한다. 즉, 한 매개변수의 최적값을 찾은 후 그거에 맞춰서 다른 매개변수 최적값을 찾으면 값이 함께 달라진다.

#그리드 서치 Grid Search: 하이퍼파라미터 탐색과 교차 검증을 한 번에 수행해줌
#예시
from sklearn.model_selection import GridSearchCV

params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=1)
# GridSearchCV의 cv 기본 값:5 n_jobs:병렬 실행에 사용할 CPU 코어 수 -1 은 모든 코어
gs.fit(train_input, train_target)
#그리드 서치는 훈련이 끝나면 훈련 모델 중 검증 점수가 가장 높은 모델의 매개변수 조합으로 전체 훈련 세트에서 자동으로 다시 모델을 훈련
dt = gs.best_estimator_ #best_estimator_: 가장 최적 모델 저장 속성
print(gs.best_params_) #best_params_: 가장 최적의 매개변수 저장 속성
print(gs.cv_results_['mean_test_score']) #cv_result_['mean_test_score']: 각 매개변수 교차 검증의 평균 점수
best_index = np.argmax(gs.cv_results_['mean_test_score']) #argmax:어떤 함수를 최대로 만드는 매개변수
print(gs.cv_results_['params'][best_index]) #가장 큰 값의 인덱스 추출 후 params에 저장된 키에 저장된 매개변수 출력

#복잡한 예시 
params =    {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001), #총 9개
             'max_depth' : range(5,20,1), #총 15개 
             'min_samples_split' : range(2, 100, 10) #총 10개
             } # 9*15*10 = 1350 교차 검증 횟수 X 5-폴드 교차 = 6,750모델
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_estimator_) #(max_depth=14, min_impurity_decrease=0.0004, min_samples_split=12, random_state=42)
print(np.max(gs.cv_results_['mean_test_score'])) #0.8683865773302731'

#랜덤 서치 Random Search: 매개변수값의 범위나 간격을 미리 정하기 어렵거나 매개변수 조건이 너무 많마 그리드 서치 수행 시간이 오래 걸릴 때
#scipy중 uniform, randint 라이브러리
from scipy.stats import uniform, randint #주워진 범위에서 값을 고르게 뽑는 라이브러리. uniform은 실수, randint는 정수

rgen = randint(0,10)
print(rgen.rvs(10))
print(np.unique(rgen.rvs(1000),return_counts=True)) #return_counts=True: 중복 횟수 카운트
ugen = uniform(0,1)
print(ugen.rvs(10))

#탐색 매개변수 딕셔너리 만들기
params = {'min_impurity_decrease': uniform(0.0001,0.001),
          'max_depth': randint(20,50),
          'min_samples_split': randint(2,25),
          'min_samples_leaf': randint(1,25),
          }
from sklearn.model_selection import RandomizedSearchCV #랜덤 서치 클래스
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score'])) #0.8695428296438884
dt = gs.best_estimator_
print(dt.score(test_input, test_target)) #0.86



