from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.axes(projection='3d')


# draw sphere
u, v = np.mgrid[0:2*np.pi:40j, 0:2*np.pi:40j]
x1 = np.cos(u)*np.sin(v)
y1 = np.sin(u)*np.sin(v)
z1 = np.cos(v)
ax.plot_wireframe(x1, y1, z1, color="0.5",linewidth=0.1)   #绘制网格


data=np.random.rand([10, 4])
#x,y,z分别表示数据的三个维度
x = data[:,0]
y = data[:,1]
z = data[:,2]
label=data[:,3]
color=['Blues','autumn','cool','YlGnBu']   #提前预定不同类别的颜色
k=np.size(np.unique(label))
for i in range(1,k+1):
    test=np.where(label==i)    #拿到属于i类的数据下标
    ax.scatter(x[test], y[test], z[test], c=z[test], marker='o',s=1,cmap=color[i-1],alpha=0.5)


plt.show()
