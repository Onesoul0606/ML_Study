'''
주성분 분석Principal Component Analysis PCA
-차원 Dimension과 차원축소 Dimension Reduction 
-주성분 분석:데이터에 있는 분산이 가장 큰 방향을 찾는 것
주성분 벡터: 주성분 방향과 원점을 지나는 벡터
'''

#데이터셋 불러오기(코랩버젼)
#!wget bit.ly/fruits_300 -O fruits_300.npy
import numpy as np
fruits = np.load('fruits_300.npy') #3차원 배열: 샘플 개수, 너비, 높이
fruits_2d = fruits.reshape(-1,100*100) #2차월 배열로 변환: 샘플 개수, 너비*높이

#PCA 클래스
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)
print(pca.components_.shape) #components_:PCA 클래스가 찾은 주성분

#각 클러스터 이미지 출력
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1): #arr:배열, ratio: 이미지 비율
  n = len(arr) #n은 샘플 개수
  #한 줄에 10개씩 이미지 그리기. 샘플 개수를 10으로 나눠서 전체 행 개수 계산
  rows = int(np.ceil(n/10)) # np.ceil 함수는 올림 처리
  #행이 1개면 열의 개수는 샘플 개수. 그렇지 않으면 10개
  cols = n if rows < 2 else 10
  fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False) #squeeze=False는 항상 axs 배열을 2차원 배열로 유지
  for i in range(rows):
    for j in range(cols):
      if i*10 + j < n:
        axs[i,j].imshow(arr[i*10+j],cmap='gray_r')
      axs[i,j].axis('off')
  plt.show()
draw_fruits(pca.components_.reshape(-1,100,100)) #원본데이터에서 가장 분산이 큰 순서대로 나타낸 것

#원본 데이터 차원 줄이기
print(fruits_2d.shape)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

#원본 데이터 재구성
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)
fruits_reconstruct = fruits_inverse.reshape(-1,100,100)
for start in [0,100,200]:
  draw_fruits(fruits_reconstruct[start:start+100])
  print("\n")

#설명된 분산 Explained Variance: 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값
print(np.sum(pca.explained_variance_ratio_))
plt.plot(pca.explained_variance_ratio_)

#다른 알고리즘과 함께 사용
#로지스틱 회귀 모델
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
target = np.array([0]*100 + [1]*100 + [2]*100)

#교차 검증 수행
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score'])) # 0.997...
print(np.mean(scores['fit_time'])) #훈련 소요 시간: 1.08...

#PCA로 축소한 데이터와 비교
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score'])) # 1.0 정확도 상승
print(np.mean(scores['fit_time'])) #0.0256... 시간 단축

#분산 비율 조정
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
print(pca.n_components_) #2
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)#데이터 크기 확인 (300,2)
#2개의 특성으로 교차 검증
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score'])) # 0.993...  #특성 2개만으로 정확도 99퍼 달성
print(np.mean(scores['fit_time'])) #훈련 소요 시간: 0.0318...

#차원 축소된 데이터로 k-평균 알고리즘으로 클러스터 찾기
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))
#이미지 출력
for label in range(0,3):
  draw_fruits(fruits[km.labels_ == label])
  print('\n')

#산점도로 시각화
for label in range(0,3):
  data = fruits_pca[km.labels_ == label]
  plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()