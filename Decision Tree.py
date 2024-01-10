
#데이터 불러오기
import pandas as pd
import plotly as px

wine = pd.read_csv('https://bit.ly/wine-date')
print(wine.head())
print(wine.info()) #데이터 타입과 누락 데이터 확인
print(wine.describe()) #데이터의 간략한 통계 확인(최소 최대 평균값 등)

#데이터프레임 정리
data = wine[["alcohol", "sugar", "pH",]].to_numpy()
target = wine[['class']].to_numpy()

#세트 나누기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42) 
#test_size: 테스트 세트 비율 기본값 25%(0.25)
print(train_input.shape, test_input.shape) # 세트 개수 확인

#데이터 표준화 전처리
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#로지스틱 회귀 모델 훈련(레드 와인(0) 화이트 와인(1) 이진 분류)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target)) #0.780...
print(lr.score(test_scaled, test_target)) #0.777... underfitting
print(lr.coef_, lr.intercept_)

#결정 트리 Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) #0.996...
print(dt.score(test_scaled, test_target)) #0.859... overfitting

#트리 모델 다이어그램 출력 및 단순화
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
# plot_tree(dt)
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
# max_depth: 리프 노트 개수, filled: 클래스 별 색깔 칠하기, feature_names:특성 이름 전달
plt.show() 

#불순도 Gini Impurity
#계산법: 1 - (음성 클래스 비율^2 + 양성 클래스 비율^2) # 0~1 지니 불순도 0: 순수 노드 Pure Node
#결정 트리 모델은 부모 노드 Parent Node 와 자식 노드 Child Node 의 불순도 차이가 가능한 크도록 성장시킴. 
#정보이득 information gain: 그 차이값(정보 이득을 최대가 되도록 데이터를 나눔)
#계산법: 자식 노드 불순도를 샘플 개수의 비례해서 더한 후 부모 노드 불순물에서 빼기
# 부모의 불순도 - (왼쪽 노드 샘플 수/부모의 샘플 수) X 왼쪽 노드 불순도 - (오른쪽 노드 샘플 수/부모의 샘플 수) X 오른쪽 노드 불순도
## 엔트로피 불순도 Entropy Impurity: 제곱 대신 log_2를 사용하여 계산

#가지치기 Pruning
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) #0.845
print(dt.score(test_scaled, test_target)) #0.841

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show() #데이터 값이 표준화된 값으로 나와서 직관적이지 않다

#전처리 전 데이터로? 결정 트리는 전처리할 필요가 없다!
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target)) #0.845
print(dt.score(test_input, test_target)) #0.841

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show() #당도나 알코올 도수 등 값이 그대로 도출되어 직관적이다

#특성 중요도: 어떤 특성이 구별하는데 있어서 가장 유용했는지 나타내는 것
print(dt.feature_importances_) #[0.12345626 0.86862934 0.0079144 ] 알코올, 당도, pH 순. 당도가 가장 유용했다. 합치면 1