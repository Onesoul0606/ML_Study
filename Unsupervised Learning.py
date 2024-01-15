#데이터셋 불러오기(코랩버젼)
#!wget bit.ly/fruits_300 -O fruits_300.npy
import numpy as np
import matplotlib.pyplot as plt

#과일 이미지 출력
fruits = np.load('fruits_300.npy')
print(fruits.shape)
print(fruits[0,0,:]) # 1행에 있는 픽셀 100개 출력
plt.imshow(fruits[0], cmap='gray') #cmap: colour map /gray 배경 물체 반전
plt.show()
plt.imshow(fruits[0], cmap='gray_r') #gray_r: 원본 그대로 출력
plt.show()

#파인애플, 바나나 이미지 출력
fig, axs = plt.subplots(1,2) #subplot(): 그래프 여러개 띄우기
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()

#픽셀값 분석
#100*100 2차원 배열을 10000개의 1차원 배열로 바꾸기
#fruits라는 큰 배열을 세 부분으로 나누고, 각 부분을 별도의 2차원 배열로 변환
apple = fruits[0:100].reshape(-1,100*100) #fruits 배열의 1부터 100까지를 -1(자동 크기 할당)로 100*100 크기로 2차원 배열을 만듦
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
#샘플 평균값 구하기
print(apple.mean(axis=1)) #axis(배열축): 2차원 배열에서 0 = 행 1 = 열

#히스토그램(값의 빈도를 나타낸 그래프) 그리기
plt.hist(np.mean(apple, axis=1), alpha=0.8) #alpha: 투명도(1보다 아래로)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])#범례 만들기
plt.show()

#픽셀별 평균값 구하기(모양이 비슷한 사과, 파인애플 구분하기)
fig, axs = plt.subplots(1,3, figsize=(20,5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

#픽셀 평균 모든 샘플 합친 이미지 출력
apple_mean = np.mean(apple, axis = 0).reshape(100,100)
pineapple_mean = np.mean(pineapple, axis = 0).reshape(100,100)
banana_mean = np.mean(banana, axis = 0).reshape(100,100)
fig, axs = plt.subplots(1,3,figsize=(20,5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

#평균값과 가까운 사진 고르기(절댓값 오차 사용): fruits 모든 배열에서 apple_mean을 뺀 절댓값의 평균
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)
#오차값이 가장 적은 샘플 100개 골라보기
apple_index = np.argsort(abs_mean[:100])   # argsort:작은 거에서 큰 순서로 나열
fig, axs =  plt.subplots(10, 10, figsize=(10,10) ) #10x10 사이즈의 서브 플롯 생성, 전체 플롯의 크기를 10x10 인치로 설정
for i in range(10):
  for j in range(10):
    axs[i,j].imshow(fruits[apple_index[i*10 + j]],cmap='gray_r')
    axs[i,j].axis('off') #각 서브플롯의 축을 비활성화. 서브플롯 주위의 눈금, 레이블 제거.
plt.show()

#군집 clustering 클러스터 cluster
#군집: 비슷한 샘플끼리 그룹으로 모으는 작업(대표적인 비지도 학습 작업 중 하나)
#클러스터: 군집 알고리즘에서 만든 그룹