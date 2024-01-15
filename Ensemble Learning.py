#앙상블 학습 알고리즘: 정형 데이터 머신러닝의 끝판왕
#정형 데이터 Structured Data: 구조화된 데이터 엑셀, 데이터베이스 등
#비정형 데이터 Unstructured Data: 구조화 되지 않은 데이터. 텍스트, 이미지, 음악 등

#랜덤 포레스트 Random Forest: 랜덤한 결정 트리의 숲
#부트스트랩 샘플 Bootstrap Sample: 데이터 내에서 중복 가능하게 샘플을 뽑는 것. 기본적으로 훈련 세트와 크기가 같음.
#노드 분할 시 무작위로 고른 다음 최선의 분할을 탐색

#데이터 불러오기
import pandas as pd
import numpy as np

wine = pd.read_csv('https://bit.ly/wine-date')

#데이터프레임 정리
data = wine[["alcohol", "sugar", "pH",]].to_numpy()
target = wine[['class']].to_numpy()

#세트 나누기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42) 
#test_size: 테스트 세트 비율 기본값 25%(0.25)

#교차 검증 수행
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42, n_jobs= -1)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
#return_train_score:훈련 점수 포함 여부
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) #0.9973541965122431 0.8905151032797809 overfitting

#랜덤 포레스트의 특성 중요도: 각 트리의 특성 중요도를 취합한 것
rf.fit(train_input, train_target)
print(rf.feature_importances_) #[0.23167441 0.50039841 0.26792718]
#결정 트리의 특성 중요도와 결과는 같지만 수치는 약간 균등해짐 -> 과대 적합 줄이고 일반화 성능 높임

#OOB out of bag: 부트스트랩 샘플에서 사용되지 않은 샘플 -> 부트스트랩 샘플로 훈련한 결정 트리를 평가할 수 있음
rf = RandomForestClassifier(oob_score=True, random_state=42, n_jobs= -1)
rf.fit(train_input, train_target)
print(rf.oob_score_)

#엑스트라 트리 Extra Tree
#random forest VS extra tree: 엑스트라 트리는 부트스트랩 샘플 사용x, 전체 샘플 사용, 노드 무작위 분할(splitter = random)
# 성능은 떨어질 수 있으나 많은 트리를 앙상블하기 때문에 과대적합 완화 및 검증 세트 점수 높임.
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) #0.9974503966084433 0.8887848893166506

#엑스트라 특성 중요도
et.fit(train_input, train_target)
print(et.feature_importances_) #[0.20183568 0.52242907 0.27573525] 결정트리보다 균등해짐

#그레디언트 부스팅 Gradient Boosting:깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블
#과대적합에 강하고 높은 일반화 성능
#gradient descent, logistic loss function, mean of squared error 사용
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) #0.8881086892152563 0.8720430147331015 

#히스토그램 기반 그레디언트 부스팅 Histogram-based Gradient Boosting: 정형 데이터를 다루는 머신러닝 알고리즘 중 가장 인기가 많음
#입력 특성을 256개로 나눔 -> 노드 분할 시 최적의 분할을 매우 빠르게 탐색 가능
#256개 중 하나를 떼어 놓음 -> 누락된 값들 사용 -> 따로 전처리 과정 필요x 
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) #0.9321723946453317 0.8801241948619236 과대 적합 잘 억제, 그레디언트보다 조금 더 성능 좋음

hgb.fit(train_input, train_target)
print(rf.feature_importances_) #[0.23167441 0.50039841 0.26792718] 조금 더 골고루
print(hgb.score(test_input,test_target)) #0.8723076923076923

#XG Boost library
from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) #0.9558403027491312 0.8782000074035686

#LightGBM library
from lightgbm import LGBMClassifier
lgb = XGBClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) #0.9558403027491312 0.8782000074035686