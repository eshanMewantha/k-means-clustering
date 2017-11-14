import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""Create random data points"""
x = np.random.randn(50) * 10
y = np.random.randn(50) * 10

temp = []
for x_, y_ in zip(x, y):
    temp.append([x_, y_])

X = np.array(temp)

"""Train K-Means model"""
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print("\nCluster Centers : " + str(kmeans.cluster_centers_))
print("\nReference labels of data points after clustering : \n" + str(kmeans.labels_))

"""Predict the cluster reference of new data points"""
predictions = kmeans.predict([[0, 0], [4, 4]])
print("\nCluster reference of the new data point [0,0] = " + str(predictions[0]))
print("\nCluster reference of the new data point [4,4] = " + str(predictions[1]))

"""Visualize clustered data points"""
x_blue, y_blue, x_red, y_red, x_green, y_green = ([] for i in range(6))

for index, value in enumerate(kmeans.labels_):
    if value == 0:
        x_blue.append(x[index])
        y_blue.append(y[index])
    elif value == 1:
        x_red.append(x[index])
        y_red.append(y[index])
    elif value == 2:
        x_green.append(x[index])
        y_green.append(y[index])
plt.plot(x_red, y_red, 'ro')
plt.plot(x_blue, y_blue, 'bo')
plt.plot(x_green, y_green, 'go')
plt.show()
