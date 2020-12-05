扩展：科学计算 numpy
==============================

科学计算常用模块

-  **Numpy**：Arrays manipulation library
   科学计算的必装模块，几乎所有的其他科学模块都依赖于它
-  **Scipy**：扩展的科学计算模块
-  **PyGSL**：C/C++语言中著名的科学计算函数库GNU Scientific Library(GSL)的python版
-  **Sympy**：符号计算模块
-  **StatLib**：统计学工具箱
-  **Escript/Finley**：偏微分方程求解
-  **Parallel Python**：并行计算模块


Numpy是Python的扩展模块，提供了矩阵、线性代数、傅里叶变换等的解决方法。
NumPy包含：N维矩阵对象、线性代数运算功能、傅里叶变换、Fortran代码集成的工具、C++代码集成的工具。


SciPy是基于Numpy构建的一个集成了多种数学算法和方便的函数的Python模块。
SciPy的子模块需要单独import。
Scipy包含：constants物理和数学常数、fftpack快速傅立叶变换程序、integrate
积分和常微分方程求解器、interpolate拟合和平滑曲线、
linalg线性代数、optimize最优路径选择、signal信号处理、sparse稀疏矩阵和以及相关程序、special特殊函数、stats统计上的函数和分布等。

Numpy数组
---------

Python中用列表(list)保存一组值，可以用来当作数组使用，不过由于列表的元素可以是任何对象，因此列表中所保存的是对象的指针。这样为了保存一个简单的[1,2,3]，需要有3个指针和三个整数对象。对于数值运算来说这种结构显然比较浪费内存和CPU计算时间。
此外Python还提供了一个array模块，可以直接保存数值，但是不支持多维，也没有各种运算函数，因此也不适合做数值运算。
NumPy的诞生弥补了这些不足，NumPy提供了两种基本的对象：ndarray(N-dimensional
array object)和ufunc(universal function object)。
ndarray(下文统一称之为数组)是存储单一数据类型的多维数组，而ufunc则是能够对数组进行处理的函数。

Numpy数组创建
-------------

.. code:: python

   >>> import numpy as np
   >>> a = np.array([1, 2, 3, 4], dtype='int32')
   >>> a = np.array([[3,4,5],[3,6,7]])
   np.arange(0,1,0.1) np.zeros(2,3) np.ones(5)
   np.linspace(0, 1, 12) np.logspace(0, 2, 20)

**函数式创建：**

.. code:: python

   def func(i, j):
       return (i+1) * (j+1)
   a = np.fromfunction(func, (9,9))

np.array 与 list 的区别
-----------------------

.. code:: python

   >>> a=range(5)
   >>> a + 1
   >>> a * 2
   >>> a + a
   >>> a > 3
   >>> np.array(a)

   >>> b=np.arange(5)
   >>> b + 1
   >>> b * 2
   >>> b + b
   >>> b > 3
   >>> b / 2
   >>> list(b)

一维数组取样
------------

.. code:: python

   >>> a = np.arange(10)
   >>> a[5] # 用整数作为下标可以获取数组中的某个元素
   >>> a[3:5] # 用范围作为下标获取数组的一个切片，包括a[3]不包括a[5]
   >>> a[:5] # 省略开始下标，表示从a[0]开始
   >>> a[:-1] # 下标可以使用负数，表示从数组后往前数，array([0, 1, 2, 3, 4, 5, 6, 7, 8])
   >>> a[2:4] = 100,101 # 下标还可以用来修改元素的值
   >>> a[1:-1:2] # 范围中的第三个参数表示步长， 2表示隔一个元素取一个元素

二维数组取样
------------

.. code:: python

   >>> a = np.arange(10).reshape(2,-1)
   >>> a
   array([[0, 1, 2, 3, 4],
          [5, 6, 7, 8, 9]])
   >>> a[1,1] #单个元素
   6
   >>> a[1] #整行
   array([5, 6, 7, 8, 9])
   >>> a[:,2] #整列
   array([2, 7])
   >>> a[0][::2] #抽取某行特定元素
   array([0, 2, 4])

条件取样
--------

.. code:: python

   >>> a = np.arange(10).reshape(-1,2)
   >>> a[a[:,1]>3]
   array([[4, 5],
          [6, 7],
          [8, 9]])
   >>> a[a[:,1]%3==0]
   array([[2, 3],
          [8, 9]])
   >>> a[(a[:,1]>3)*(a[:,1]%3==0)]

数组排序
--------

argsort函数返回数组值从小到大的索引

.. code:: python

   >>> x = np.array([3,1,2])
   >>> np.argsort(x)
   >>> x[np.argsort(x)] # 排序后的数组
   >>> x=np.array([[0,3],[4,2]])
   >>> np.argsort(x, axis=1) # 排序每行
   >>> a[a[:,1].argsort()] # 按第二列排序

数学数组方法
------------

.. code:: python

   >>> a = np.arange(6).reshape(2,3)
   >>> a.shape 
   (2, 3)
   >>> a.dtype 
   dtype('int32')
   >>> a.reshape(3,2)      # 改变数组维度
   >>> a.ravel()           # 展开数组
   >>> a.repeat(2,axis=0)  # 复制元素

数组合并
--------

.. code:: python

   >>> a = np.array([1, 2, 3])
   >>> b = np.array([2, 3, 4])
   >>> np.r_[a,b]
   >>> np.hstack((a,b))
   array([1, 2, 3, 2, 3, 4])
   >>> np.vstack((a,b))
   array([[1, 2, 3],
          [2, 3, 4]])
   >>> np.c_[a,b]
   array([[1, 2],
          [2, 3],
          [3, 4]])

数据存储
--------

很多时候我们需要将程序运算得到的数据进行存储，Numpy为我们提供了存储数据的函数。格式如下：

.. code:: python

   numpy.savetxt(fname, X, fmt='%.18e',delimiter=' ', newline='\n', header='',footer='', comments='# ')

   >>> x = y = z = np.arange(0.0,5.0,0.5)
   >>> np.savetxt('test.out', x, delimiter=',')
   # X is an array
   >>> np.savetxt('test.out', (x,y,z))
   # x,y,z equal sized 1D arrays
   >>> np.savetxt('test.out', x, fmt='%6.4f')you
   # use exponential notation

数据读取
--------

既然更够存储数据，那一定也有读取之前已经存储的数据的方法。函数的格式如下：

.. code:: tex

   numpy.loadtxt(fname, dtype=<type 'float'>,comments='#', delimiter=None,
           converters=None, skiprows=0, usecols=None,unpack=False, ndmin=0)

   让我们来读取刚才已经存储的数据
   
.. code:: python

   >>> data = np.loadtxt('test.out', dtype = float)
   >>> data = np.loadtxt('test.out', usecols=[1])

和math函数比较
--------------

Python本身其实自带math库以用于一般的数学计算，Numpy中的函数是针对数组设计的，且更为快速和强大，这里我们来弄清楚二者的具体区别。
将下面的代码保存为脚本并执行。

.. code:: python

   import time, math
   import numpy as np
   n = 1e+6
   x = range(int(n))
   start = time.clock()
   for i in x:
       tmp = math.sin(i/n)
   print("math.sin:", time.clock() - start)
   x = np.array(x)/n
   start = time.clock()
   np.sin(x)
   print("numpy.sin:", time.clock() - start)
