import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# 对数据集的dataFrame进行操作，记录每次每个点属于哪个类
def assignment(df, centroids, colmap):
    for i in centroids.keys():
        # 计算每个点对每个聚类中心的距离，并分别导入为df的distance_from_0 ~ distance_from_k列中
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    # 将每个点所属于的聚类中心标出，0, 1, 2, ..., k
    # 先取df中distance_from_0 ~ distance_from_k列，每一行的最小值的索引（索引为distance_from_0 ~ distance_from_k）
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_'))) #去除最小值索引的无效字符串，并转为整型
    # 对属于不同聚类中心的点，赋值对应的颜色，用于绘图
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

# 根据当前聚类组的所有数据，更新对应的聚类中心
def update(df, centroids):
    for i in centroids.keys():
        # 对df中'closest'列中，分别属于0~k类的点，求平均
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

# k-means++：利用概率来确定初始中心点，防止初始中心点过于聚集
def centroids_plus(df, k):
    df1 = df
    center={}
    # 聚类中心第一个点,从df1的所有点中随机取一个
    first_center = np.random.randint(0, df1.shape[0])
    center[0] = [df1.iloc[first_center][0], df1.iloc[first_center][1]]
    # 确定后面的初始聚类中心点
    for i in range(1, k):
        # 计算每个点到前一个聚类中心点的距离
        distance = np.sqrt( (df1['x'] - center[i-1][0]) ** 2 + (df1['y'] - center[i-1][1]) ** 2 )
        # 将距离归一化成概率/权重
        weight = distance/distance.sum()
        # 按照概率/权重大小，随机选择后一个聚类中心点，即df1的第n_center行对应的点
        num = random.random()
        total = 0
        n_center = -1
        while total<num:
            n_center += 1
            total += weight[n_center]
        center[i] = [df1.iloc[n_center][0], df1.iloc[n_center][1]]
    return center

def main():
    # step 0.0: generate source data
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })

    k = 3

    # k-means 确定初始聚类中心点
    # centroids[i] = [x, y]
    #centroids = {
    #   i: [np.random.randint(0, 80), np.random.randint(0, 80)]
    #   for i in range(k)
    #}

    #k-means++ 确定初始聚类中心点
    centroids = centroids_plus(df, k)

    # step 0.2: assign centroid for each source data
    # for color and mode: https://blog.csdn.net/m0_38103546/article/details/79801487
    # colmap = {0: 'r', 1: 'g', 2: 'b', 3: 'm', 4: 'c'}
    colmap = {0: 'r', 1: 'g', 2: 'b'}
    df = assignment(df, centroids, colmap)

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()

    for i in range(10):
        #key = cv2.waitKey()
        plt.close()

        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(df, centroids)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):
            break

if __name__ == '__main__':
    main()

