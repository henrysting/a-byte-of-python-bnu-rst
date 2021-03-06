扩展：绘图模块 matplotlib
============================


Matplotlib基于数值计算模块Numpy，克隆了许多Matlab中的函数，能够帮助用户轻松地获得高质量的二维图形。
Matplotlib模块支持的图表类型非常之多，几乎能胜任任何绘图任务。
但对于特定数据，选取合适的图表类型来表达数据的内涵非常重要。
为了方便科学计算和数据分析的初学者，我们这里给出一个简单的示意图来告诉大家如何选取合适的图表类型。

|image3|

下面我们来通过实际的例子来学习matplotlib绘图模块。

二维图像
------------

范例：绘制函数

.. code:: python

   import numpy as np
   import matplotlib.pyplot as plt

   x = np.linspace(0, 10, 1000)
   y = np.sin(x) ; z = np.cos(x**2)
   plt.figure()
   plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
   plt.plot(x,z,"b--",label="$cos(x^2)$")
   plt.legend(loc=3)
   plt.show() #保存图像可用 plt.savefig('fig.jpg')
   plt.close()

程序运行结果如下：

|image0|

Plot参数
~~~~~~~~
plot 命令中包含丰富的选项，可以灵活的调节数据点和连线的各种属性。

-   **alpha** : float
-   **color or c**: any matplotlib color
-   **label** : any string , 图注名称
-   **linestyle or ls**: [ '-' | '--' | '-.' | ':' | 'steps' | ...]
-   **linewidth or lw**: float value (points, 0.3527mm )
-   **marker** [ '+' | ',' | '.' | '1' | '2' | '3' | '4' ]
-   **markersize or ms** : float
-   **zorder**: any number 叠放顺序

颜色
~~~~
matplotlib 包含8种基本颜色，可以用单个字母表示：

*  蓝色： 'b' (blue)
*  绿色： 'g' (green)
*  红色： 'r' (red)
*  青色： 'c' (cyan)
*  洋红： 'm' (magenta)
*  黄色： 'y' (yellow)
*  黑色： 'k' (black)
*  白色： 'w' (white)

也可以用数字表示灰度，如 0.75 ([0,1]内任意浮点数)。
如果需要更多颜色，还可以采用RGB表示法： 由红色、绿色和蓝色的值组成的十六进制符号来定义，如 '#2F4F4F' 或 (0.18,0.31,0.31)

|image1|

坐标轴定制
~~~~~~~~~~

.. code:: python

   import matplotlib.pyplot as plt
   plt.title('sine function demo')
   plt.xlabel('time(s)')
   plt.ylabel('votage(mV)')
   plt.xlim([0.0,5.0])
   plt.ylim([-1.2,1.2])
   #plt.hold('on') # 保持之前plot的结果
   #这里在python 3.6.9测试时已经默认使用hold为true的结果,在Python3已经移除plt.hold('on')这一选项,
   # 因此我们将上一句注释掉
   plt.grid('on') # 添加网格
   plt.text(4,0,'$\mu=100$') # 文本
   plt.axis('equal') # 等比例坐标轴
   plt.ylim(plt.ylim()[::-1]) # 翻转Y轴
   plt.gca().invert_yaxis() # 翻转Y轴
   plt.axvline(x=0)# 竖直参考线
   plt.axhline(y=0,color='k'  )# 水平参考线
   plt.show()

极坐标
~~~~~~

.. code:: python

   import numpy as np
   import matplotlib.pyplot as plt

   r = np.arange(0, 3.0, 0.01)
   theta = 2 * np.pi * r
   ax = plt.subplot(111, polar=True)
   ax.plot(theta, r, color='r', linewidth=3)
   ax.set_rmax(2.0)
   ax.grid(True)
   ax.set_title("polar plot")
   plt.show()

程序运行结果如下：

|image2|


直方图
~~~~~~

.. code:: python

   import numpy as np
   import matplotlib.mlab as mlab
   import matplotlib.pyplot as plt

   mu = 100 # mean of distribution
   sigma = 15 # standard deviation of distribution
   x = mu + sigma * np.random.randn(10000)
   num_bins = 50
   # the histogram of the data
   n, bins, patches = plt.hist(x, num_bins, normed=1,
   facecolor='green', alpha=0.5)
   y = mlab.normpdf(bins, mu, sigma) # add a 'best fit' line
   plt.plot(bins, y, 'r--')
   plt.show()

程序运行结果如下：

|image4|

散点图
~~~~~~

.. code:: python

   import matplotlib.pyplot as plt
   import numpy as np

   n = 150
   x = np.random.rand(n,3)
   c = np.random.rand(n,3)
   plt.scatter(x[:,0], x[:,1], s=x[:,2]*500, alpha=0.5, color=c)
   plt.show()

程序运行结果如下：

|image5|

柱状图
~~~~~~

.. code:: python

   from matplotlib.ticker import FuncFormatter
   import matplotlib.pyplot as plt
   import numpy as np

   x = np.arange(4)
   money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]

   def millions(x, pos):
       'The two args are the value and tick position'
       return '$%1.1fM' % (x * 1e-6)

   formatter = FuncFormatter(millions)

   fig, ax = plt.subplots()
   ax.yaxis.set_major_formatter(formatter)
   plt.bar(x, money)
   plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
   plt.show()

程序运行结果如下：

|image6|

多子图
~~~~~~

.. code:: python

   subplot(numRows, numCols, plotNum)
   plt.subplot(221) # 第一行的左图
   plt.subplot(222) # 第一行的右图
   plt.subplot(212) # 第二整行
   plt.show()
   ax1 = plt.subplot(211) # 创建子图1
   ax1.plot(x,y)
   ax2 = plt.subplot(212) # 创建子图2
   ax2.plot(x,y)

.. code:: python

   #多子图示例
   import numpy as  np
   import matplotlib.pyplot as  plt
   x = np.linspace(0,  10 , 1000)
   y1 = np.sin (x) ; y2 = np.cos (x**2); y3=np.tan  (x)
   ax1 = plt.subplot(211 ) 
   ax1.plot(x,y1)
   ax2 = plt.subplot(223 ) 
   ax2.plot(x,y2)
   ax3 = plt.subplot(224 ) 
   ax3.plot(x,y3)
   plt.show()

|image15|

热力图
~~~~~~

.. code:: python

   #热力图
   import numpy as np
   import matplotlib.pyplot as  plt
   def bbfunc(lam,T):
      h=6.626e-34
      c=2.99792e+8
      k=1.3806e-23
      lam=lam*1e-9
      ddd=2*h*c*c/(lam*lam*lam*lam*lam)/(np.exp(h*c/(lam*k*T))-1)
      return ddd
   n = 100
   lam = np.linspace(0,2000,n) 
   T  = np.linspace(4000,6000,n)
   X, Y = np.meshgrid(lam, T)
   Z = bbfunc(X,Y)
   plt.imshow(Z, cmap=plt.get_cmap('jet'))
   plt.colorbar()
   plt.show()
   plt.imshow(Z[::  -1],  cmap=plt.get_cmap('jet'), extent=[0, 2000,  4000, 6000])

|image16|


colormap
~~~~~~~~

|image7|

.. code:: python

   #查看可用色表
   import pylab as pl
   pl.colormaps()
   #查看色表内容

   pl.cm.hot(0.001)
   pl.cm.hot(0.999)
   pl.cm.hot(0.5)
   pl.cm.hot(0.5, 0.5)

三维作图
---------

.. code:: python

   from matplotlib import pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D
   import numpy as np

   fig = plt.figure()
   ax = Axes3D(fig)
   data = np.random.random([100,3])
   np.random.shuffle(data)
   ax.scatter(data[:,0],data[:,1],data[:,2], marker='o')
   plt.show()

程序运行结果如下：

|image8|

三维曲面
~~~~~~~~

.. code:: python

   from mpl_toolkits.mplot3d import Axes3D
   import matplotlib.pyplot as plt
   import numpy as np

   cmap = plt.cm.jet
   fig = plt.figure()
   ax = fig.gca(projection='3d')
   X = np.arange(-5, 5, 0.25)
   Y = np.arange(-5, 5, 0.25)
   X, Y = np.meshgrid(X, Y)
   Z = np.sin(np.sqrt(X**2 + Y**2))
   ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap )
   ax.set_zlim(-1.01, 1.01)
   plt.show()

程序运行结果如下：

|image9|

等高线图
~~~~~~~~

.. code:: python

   import matplotlib.pyplot as plt
   import numpy as np

   plt.figure()
   X = np.arange(-5, 5, 0.25)
   Y = np.arange(-5, 5, 0.25)
   X, Y = np.meshgrid(X, Y)
   Z = np.sin(np.sqrt(X**2 + Y**2))
   levels = np.arange(-1,1,0.25)
   cs = plt.contour(X, Y, Z, levels)
   plt.clabel(cs,inline=1,fontsize=8)
   plt.axis('equal')
   plt.show()

程序运行结果如下：

|image10|

三维投影
~~~~~~~~

.. code:: python

   from mpl_toolkits.mplot3d import axes3d
   import matplotlib.pyplot as plt
   from matplotlib import cm

   fig = plt.figure()
   ax = fig.gca(projection='3d')
   X, Y, Z = axes3d.get_test_data(0.1)
   ax.plot_surface(X, Y, Z, rstride=8,cstride=8, alpha=0.3)
   cset = ax.contour(X, Y, Z, zdir='z', offset=-100)
   cset = ax.contour(X, Y, Z, zdir='x', offset=-40)
   cset = ax.contour(X, Y, Z, zdir='y', offset=40)
   plt.show()

程序运行结果如下：

|image11|

mplot3d 函数
~~~~~~~~~~~~

.. code:: text

   plot3D：三维控件绘图
   plot_surface： 三维网格曲面
   plot_trisurf： 三维三角曲面
   plot_wireframe：三维线图
   quiver： 矢量图
   quiver3D： 三维矢量图
   scatter: 散点图

三维球面
~~~~~~~~

方法一：

.. code:: python

   from mpl_toolkits.mplot3d import Axes3D
   import matplotlib.pyplot as plt
   import numpy as np

   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   u = np.linspace(0, 2 * np.pi, 100)
   v = np.linspace(0, np.pi, 100)
   x = 10 * np.outer(np.cos(u), np.sin(v))
   y = 10 * np.outer(np.sin(u), np.sin(v))
   z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
   ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')
   plt.show()

程序运行结果如下：

|image12|

方法二：

.. code:: python

   from mpl_toolkits.mplot3d import Axes3D
   import matplotlib.pyplot as plt
   import numpy as np

   fig = plt.figure()
   ax = fig.gca(projection='3d')
   u, v = np.ogrid[0:2*np.pi:20j, 0:np.pi:20j]
   x=np.cos(u)*np.sin(v)
   y=np.sin(u)*np.sin(v)
   z=np.cos(v)
   ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.3)
   plt.show()

程序运行结果如下：

|image13|


动画绘制
-----------

动画模块 animation
~~~~~~~~~~~~~~~~~~
matplotlib 中包含生成动画的子模块animation，我们通过下面的例子来看一下它的用法。

.. code:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.animation as ani
   fig = plt.figure()
   x = np.arange(0, 2*np.pi, 0.01) # x-array
   line, = plt.plot(x, np.sin(x))
   def animate(i):
       line.set_ydata(np.sin(x+i/10.0)) # update the data
       return line
   ani.FuncAnimation(fig, animate, np.arange(1, 200), interval=25, blit=True)
   plt.show()


实时动画
~~~~~~~~~
 
.. code:: python

   import pylab as pl
   import numpy as np
   pl.ion() #实时绘图
   pl.show()
   x = np.arange(0,2*np.pi,0.01)
   line, = pl.plot(x,np.sin(x))
   for i in np.arange(1,200):
       line.set_ydata(np.sin(x+i/10.0))
       pl.pause(0.05)
   pl.ioff() #关闭实时绘图

程序运行结果如下：

|image14|

保存动画
~~~~~~~~
生成可以播放的动画通常需要额外的视频编码器。
这里我们先保存制作动画所需的单帧图像。

.. code:: python

   #图片保存
   #程序运行前，先在该文件目录下新建一个文件夹ani
   import pylab as  pl
   import numpy as  np
   x = np.arange(0,2*np.pi,0.01)
   for i in np.arange(200):
      pl.figure()
      pl.plot(x,np.sin (x+i /10.0))
      pl.savefig("/home/user/Desktop/助教/program/ani/{:0>3d}.png".format(i))
      #换成你想保存的绝对路径不会出错，注意不同操作系统下斜杠与反斜杠区别

接下来使用imageio进行动画文件制作

.. code:: python

   #动画文件制作imageio
   #注意先运行上一个程序生成完图片再运行该程序生成动画
   import imageio
   import os
   def main():
      img_folder= "ani"
      files = os.listdir(str(os.getcwd()) + "/" +img_folder)
      frames = []
      for file in files:
         img_path= img_folder+ "/" + file
      img_path= os.path.join(img_folder, file)
      frames.append(imageio.imread(img_path))
      imageio.mimsave("ani.gif", frames, 'GIF', duration=0.1)
   if __name__ == "__main__":
      main()




.. |image0| image:: ../pic/matplotlib/figure_1.png
.. |image1| image:: ../pic/matplotlib/figure_2.png
   :width: 400 px
.. |image2| image:: ../pic/matplotlib/figure_3.png
.. |image3| image:: ../pic/matplotlib/figure_4.png
.. |image4| image:: ../pic/matplotlib/figure_5.png
.. |image5| image:: ../pic/matplotlib/figure_6.png
.. |image6| image:: ../pic/matplotlib/figure_7.png
.. |image7| image:: ../pic/matplotlib/figure_8.png
.. |image8| image:: ../pic/matplotlib/figure_9.png
.. |image9| image:: ../pic/matplotlib/figure_10.png
.. |image10| image:: ../pic/matplotlib/figure_11.png
.. |image11| image:: ../pic/matplotlib/figure_12.png
.. |image12| image:: ../pic/matplotlib/figure_13.png
.. |image13| image:: ../pic/matplotlib/figure_14.png
.. |image14| image:: ../pic/matplotlib/figure_16.png

.. |image15| image:: ../pic/matplotlib/subplot.png
.. |image16| image:: ../pic/matplotlib/heatmap.png
