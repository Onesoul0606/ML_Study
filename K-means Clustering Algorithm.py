"""
K-means/Cluster Centre/Centroid
진짜 샘플은 랜덤으로 섞여 있어서 어디까지가 사과,파인애플, 바나나인지 알 수 없음
작동방식:
1. 무작위로 k개의 클러스터 중심 정하기
2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정.
3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경.
4. 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복.
"""

#데이터셋 불러오기(코랩버젼)
#!wget bit.ly/fruits_300 -O fruits_300.npy
import numpy as np
fruits = np.load('fruits_300.npy') #3차원 배열: 샘플 개수, 너비, 높이
fruits_2d = fruits.reshape(-1,100*100) #2차월 배열로 변환: 샘플 개수, 너비*높이

#K_means Class
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(km.labels_)
print(np.unique(km.labels_, return_counts=True)) #클러스터별 개수 확인

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
draw_fruits(fruits[km.labels_==0])
draw_fruits(fruits[km.labels_==1])
draw_fruits(fruits[km.labels_==2])

#클러스터 중심
#Kmeans 클래스가 최종적으로 찾은 클러스터 중심은 cluster_centers_ 속성에 저장
draw_fruits(km.cluster_centers_.reshape(-1,100,100),ratio=3)
#.transform: 훈련 데이터 샘플에서 클러스터 중심까지 거리로 변환
print(km.transform(fruits_2d[100:101])) #[[3393.8136117  8837.37750892 5267.70439881]]
#이 샘플은 첫 번째(0) 클러스터와 가장 길이가 찗으므로 0번 레이블에 속할 듯
print(km.predict(fruits_2d[100:101]))
draw_fruits(fruits[100:101])
#n_iter:최적의 클러스터를 찾을 때까지 알고리즘이 반복한 횟수 
print(km.n_iter_)

#최적의 k 찾기(실전용)
#엘보우 방법 Elbow Method
#이너셔 Inertia: 클러스터 중심과  샘플 사이의 거리 제곱 합(샘플들이 얼마나 가까이 모여있는지)
#클러스터 개수 증가 -> 이너셔 감소 -> 어느 지점에서 감소하는 속도가 급격히 깎임. 그 모양이 엘보우

#이너셔 계산
intertia = []
for k in range(2,7):
  km = KMeans(n_clusters=k, random_state=42)
  km.fit(fruits_2d)
  intertia.append(km.inertia_)
plt.plot(range(2,7), intertia)
plt.show()