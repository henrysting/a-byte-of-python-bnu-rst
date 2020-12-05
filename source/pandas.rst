扩展：数据处理模块 pandas
============================

Numpy主要处理结构化数据, 适合用于数据量较小并且比较规则的时候。
Pandas能够更灵活、方便地处理大量一致性不好、需要清理的数据。
Pandas是基于NumPy构建的, 支持 CSV，Excel，JSON, HTML, SQL，HDF5等多种数据格式。
Panas包含两个主要数据结构：Series和DataFrame。下面我们分别来介绍。

一维数据表(Series)
---------------------

Series是一种类似干一维数组的对象,它由一组数据(各种NumPy数据类型以及一组与之相关的数据标签(即索引)组成。
仅由一组数据,使用 ``pandas.Series`` 即可产生最简单的Series。

Series的字符串表现形式为:索引在左边,值在右边。由于我们没有为数据指定索引,于是会自动创建一个0到N-1(N为数据的长度）的整数型索引。
你可以通过Series的values和index属性获取其数组表示形式和索引对象。

以下是表的内容：

.. code:: text

    USA 3.3
    JP 1.3
    CH 14.0
    IN 13.5
    dtype: float64

.. code:: python 

    >>> import pandas as pd
    >>>  people = [3.3,1.3,14,13.5]
    >>>  a = pd.Series(people)
    >>>  print("The Series is:\n", a)
    The Series is:
    0     3.3
    1     1.3
    3    13.5
    dtype: float64
    >>>  print("The values are: \n", a.values)
    The values are:
    [ 3.3  1.3 14.  13.5]
    >>>  print("The index is?\n", a.index)
    The index is?
    RangeIndex(start=0, stop=4, step=1)
    >>>  print(a[1])
    1.3
    >>>  print(a[:2])
    0    3.3
    1    1.3
    dtype: float64
    >>>  country = ["USA","JP","CHN","IND"]
    >>>  b = pd.Series(people,country)
    >>>  print(b[1])
    1.3
    >>>  print(b["USA"])
    3.3
    >>>  print(b[:"CHN"])
    USA     3.3
    JP      1.3
    CHN    14.0
    dtype: float64

使用字典创建并使用索引筛选

.. code:: python 

    >>> import pandas as pd 
    >>> data = {"USA":3.3, "JP":1.3, "CH":14, "IN":13.5}
    >>> c = pd.Series(data)
    >>> print(c)
    USA     3.3
    JP      1.3
    CH     14.0
    IN     13.5
    dtype: float64
    >>> print(c["CH"])
    14.0
    >>> d = pd.Series(data, index=["USA","CH"])
    >>> print(d)
    USA     3.3
    CH     14.0
    dtype: float64


一维数据表(Series)-创建

.. code:: python

    import pandas as pd
    people = [3.3,1.3,14,13.5]
    test = pd.Series(people, index=[5, 3, 10, 6])
    print(test)
    print(test[5])
    print(test[:2])

.. code:: text

    5      3.3
    3      1.3
    10    14.0
    6     13.5
    dtype: float64
    3.3
    5    3.3
    3    1.3
    dtype: float64

一维数据表(Series)-创建2

.. code:: python

    #使用字典进行创建

    import pandas as pd
    data = {"USA":3.3, "JP":1.3, "CH":14, "IN":13.5}
    c = pd.Series(data)
    print(c)
    print(c["CH"])
    #使用索引筛选内容
    d = pd.Series(data, index=["USA","CH"])
    print(d)

    #不连续数字索引
    people = [3.3,1.3,14,13.5]
    test = pd.Series(people, index=[5, 3, 10, 6])
    print(test)
    print(test[5])
    print(test[:2])

这里的索引分为两种：

隐式索引：默认的数据行(列)编号, 如numpy.array 

显式索引：明确给出的数据行(列)标签，如pandas.series

索引器(indexer)
------------------

loc: 使用显式索引(标签), label based indexing

iloc：使用隐式索引(位置), positional indexing 

ix：前两种索引的混合模式，主要用在DataFrame中.为了在DataFrame的行上进行标签索引，引入了专门的索引字段ix。
它使你可以通过NumPy式的标记法以及轴标签从DataFrame中选取行和列的子集。

.. code::python

    import pandas as pd
    people = [3.3,1.3,14,13.5]
    test = pd.Series(people, index=[5, 3, 10, 6])
    test.loc[3]
    test.loc[:3]
    test.iloc[3]
    test.iloc[:3]

从上到下结果分别为
.. code:: text

    14.0

    5 3.3
    3 14.0
    dtype: float64

    13.5

    5 3.3
    3 14.0
    dtype: float64

Series 更新
-----------
.. code:: python

    import pandas as pd
    s1=pd.Series([1,2,3])
    s2=pd.Series([4,5,6])
    s3 = pd.Series([4,5,6],index=[3,4,5])
    s1.append(s2)
    #正常通过,因为没有检查索引一致性
    s1.append(s2, verify_integrity=True) 
    #报错:
    '''
    ValueError: Indexes have overlapping values: Int64Index([0, 1, 2], dtype='int64')
    '''
    s1.append(s3)
    s1.append(s2, ignore_index=True)
    #正常更新


二维数据表(DataFrame)
---------------------

DataFrame是一个表格型的数据结构,它含有一组有序的列,每列可以是不同的值类型(数值、字符串、布尔值等）。
DataFrame既有行索引也有列索引，它可以被看做由Series组成的字典(共用同一个索引）。跟其他类似的数据结构相比
(如R的data. frame),DataFrame中面向行和面向列的操作基本上是平衡的。其实，DataFrame中的数据是以一个或多个二维块存放的
(而不是列表、 字典或别的一维数据结构）。有关DataFrame内部的技术细节远远超出了本章节所讨论的范围。

注意:虽然Dataframe是以二维结构保存数据的 ，但你仍然可以轻松地将其表示为更高维度的数据
(层次化索引的表格型结构，这是pandas中许多高级数据处理功能的关键要素 ）。

构建Dataframe的办法有很多，最常用的一种是直接传入一个由等长列表或NumPy数组
组成的字典.结果DataFrame会自动加上索引(跟Series一样)，且全部列会袚有序排列:


.. code:: python

    import numpy as np
    data = {'CHN':{'COUNTRY':'China', 'POP': 1398, 'AREA': 9597,'IND_DAY': '1949-10-01'}},
    'IND':{'COUNTRY':'India', 'POP': 1351, 'AREA': 3287,},
    'USA':{'COUNTRY':'US', 'POP': 329, 'AREA': 9833, 'IND_DAY': '1776-07-04'}}
    df = pd.DataFrame(data)
    df.index  # 行标签 
    print(data['CHN']) #  按列索引 
    print(df.loc['POP']))    # 按行索引

.. code:: text

    {'CHN': {'COUNTRY': 'China', 'POP': 1398, 'AREA': 9597, 'IND_DAY': '1949-10-01'}, 'IND': {'COUNTRY': 'India', 'POP': 1351, 'AREA': 3287}, 'USA': {'COUNTRY': 'US', 'POP': 329, 'AREA': 9833, 'IND_DAY': '1776-07-04'}}
    {'COUNTRY': 'China', 'POP': 1398, 'AREA': 9597, 'IND_DAY': '1949-10-01'}
    CHN    1398
    IND    1351
    USA     329
    Name: POP, dtype: object

如果指定了列序列，则DataFrame的列就会按照指定顺序进行排列:

二维数据表(DataFrame)-创建1

.. code:: python

    import pandas as pd
    s = pd.Series([1,2,3,4,5])
    print("S=\n", s)
    print()
    df = pd.DataFrame(s, columns=['digits'])
    print("df=\n", df)

.. code:: text

    S=
    0    1
    1    2
    2    3
    3    4
    4    5
    dtype: int64

    df=
        digits
    0       1
    1       2
    2       3
    3       4
    4       5

二维数据表(DataFrame)-创建2

在通过字典创建的时候，如果有的值并不存在，则自动用NaN填充。Nan在算术运算中会自动对齐不同索引的数据。

.. code:: python

    import pandas as pd
    data = {'CHN':{'COUNTRY':'China', 'POP': 1398, 'AREA': 9597,'IND_DAY': '1949-10-01'}},\
    'IND':{'COUNTRY':'India', 'POP': 1351, 'AREA': 3287},\
    'USA':{'COUNTRY':'US', 'POP': 329, 'AREA': 9833, 'IND_DAY': '1776-07-04'}}
    df = pd.DataFrame({"COU": country, "PEO":people})
    print("df = \n", df)

    # 在通过字典创建的时候，如果有的值并不存在，则自动用NaN填充，例如：

    dl = [{"a":1, "b":1}, {"b":2, "c":2}, {"c":3, "d":3}]
    df = pd.DataFrame(dl)
    print("df = \n", df)

两次结果分别为:

.. code:: text

    df = 
        COU   PEO
    0  USA   3.3
    1   JP   1.3
    2  CHN  14.0
    3  IND  13.5

    df =
        a    b    c    d
    0  1.0  1.0  NaN  NaN
    1  NaN  2.0  2.0  NaN
    2  NaN  NaN  3.0  3.0

二维数据表(DataFrame)-创建:通过Numpy二维数组创建

.. code:: python

    import numpy as np

    df = pd.DataFrame(np.zeros([5,3]),columns=["A", "B", "C"], index=["a", "b", "c", "d", "e"])
    print("df=\n",df)

.. code:: text

    df=
        A    B    C
    a  0.0  0.0  0.0
    b  0.0  0.0  0.0
    c  0.0  0.0  0.0
    d  0.0  0.0  0.0
    e  0.0  0.0  0.0

这里可以处理的数据类型:

.. code:: text

    object, 字符串类型
    int, 整型
    float,  浮点型 
    datetime, 时间类型 
    bool, 布尔型

数据筛选

另一种常见的数据形式是嵌套字典(也就是字典的字典).它就会被解释为:外层字典的键作为列，内层键则作为行
索引,我们也可以对该结果进行转置:

.. code:: python

    import pandas as pd
    data = {'CHN':{'COUNTRY':'China', 'POP': 1398, 'AREA': 9597,'IND_DAY': '1949-10-01'},
    'IND':{'COUNTRY':'India', 'POP': 1351, 'AREA': 3287,},
    'USA':{'COUNTRY':'US', 'POP': 329, 'AREA': 9833, 'IND_DAY': '1776-07-04'}}
    df = pd.DataFrame(data=data, index=pd.Series(['POP','AREA'])).T 
    print(df['POP']) #返回列
    print(df[1:2]) #返回行 
    print(df[1:2][:2])
    print(df['POP'][3:6])
    print(df[3:6]['POP'])
    print(df.iloc[1]) #返回单列数据
    print(df.iloc[1:3]) #返回切片列数据，相当于data.loc[[1,2,3]] 
    #print(df.loc[:4,['POP']]) #返回指定行的指定类
    #ps:这句话在python3.9跑不通,现在不知道怎么改
    print(df.iloc[:2,1:3]) #返回特定行特定列的数据

.. code:: text

    CHN    1398
    IND    1351
    USA     329
    Name: POP, dtype: int64
        POP  AREA
    IND  1351  3287
        POP  AREA
    IND  1351  3287
    Series([], Name: POP, dtype: int64)
    Series([], Name: POP, dtype: int64)
    POP     1351
    AREA    3287
    Name: IND, dtype: int64
        POP  AREA
    IND  1351  3287
    USA   329  9833
        AREA
    CHN  9597
    IND  3287

基于numpy的运算
----------------------

Pandas基于Numpy，运算结果保留索引和列标签，而且自动对齐索引，没有数据的位置自动用NaN填充.

.. code:: python

    import numpy as np
    import pandas as pd
    s1 = pd.Series({"A": 1, "B":2, "D":4, "E":5}, name="ONE")
    print(s1)
    print(np.sqrt(s1))

    s2 = pd.Series({ "D":4, "E":5, "F":6}, name="TWO")
    print(s1 + s2)
    print(s1.add(s2, fill_value=100))

.. code:: text

    A    1
    B    2
    D    4
    E    5
    Name: ONE, dtype: int64
    A    1.000000
    B    1.414214
    D    2.000000
    E    2.236068
    Name: ONE, dtype: float64
    A     NaN
    B     NaN
    D     8.0
    E    10.0
    F     NaN
    dtype: float64
    A    101.0
    B    102.0
    D      8.0
    E     10.0
    F    106.0
    dtype: float64

运算2

.. code:: python

    import pandas as pd
    import numpy as np
    A1 = np.random.randint(10, size=(3,5))
    df1 = pd.DataFrame(A1, columns=list("ABCDE"))
    print("df1 = \n", df1)

    df2 = df1 - df1.iloc[1] #按行计算
    print("\n df2 = \n", df2)

    df3 = df1.subtract(df1["B"], axis=0) #按列运算
    print("\n df3 = \n", df3)

.. code:: text

    df1 = 
        A  B  C  D  E
    0  5  7  6  6  7
    1  5  2  5  0  6
    2  1  9  4  7  4

    df2 =
        A  B  C  D  E
    0  0  5  1  6  1
    1  0  0  0  0  0
    2 -4  7 -1  7 -2

    df3 =
        A  B  C  D  E
    0 -2  0 -1 -1  0
    1  3  0  3 -2  4
    2 -8  0 -5 -2 -5

绘图
-----

.. code:: python

    import pandas as pd
    data = {'CHN':{'COUNTRY':'China', 'POP': 1398, 'AREA': 9597,'IND_DAY': '1949-10-01'}},
    'IND':{'COUNTRY':'India', 'POP': 1351, 'AREA': 3287,},
    'USA':{'COUNTRY':'US', 'POP': 329, 'AREA': 9833, 'IND_DAY': '1776-07-04'}}
    df = pd.DataFrame(data=data, index=['POP','AREA']).T 
    df.loc['China'][6:].plot() 
    #绘图 
    import pylab as pd
    pd.show()
    df.iloc[127:135,6:].T.plot() 
    #绘多图 
    df.iloc[127:135,6:].T.plot(logy=True)
    style=['s-','o-','^-'],color=['b','r','y'],linewidth=[2,1,1]

合并数据
---------

``concat()`` ,  ``append()`` ,  ``merge()`` 一般都是用来连接两个或者多个DataFrame对象。
其中，  ``concat()`` ,  ``append()`` 默认用来纵向连接DataFrame对象，  ``merge()`` 用来横向连接DataFrame对象。

合并数据concat

.. code:: python

    import pandas as pd
    s1 = pd.Series(list("ABC"), index =[1,2,3])
    s2 = pd.Series(list("DEF"), index =[4,5,6]) 
    s =  pd.concat([s1, s2])
    print(s)
    df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['A','B'])
    df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['A','B']) 
    #df3 = pd.concat([df1, df2])
    df3 = pd.concat([df1, df2], ignore_index=True) 
    df4 = pd.concat([df1, df2], keys=["C", "D"])
    print(df1)
    print("\n")
    print(df2)
    print("\n")
    print(df3)
    print("\n")
    print(df4)

.. code:: text

    1    A
    2    B
    3    C
    4    D
    5    E
    6    F
    dtype: object
    A  B
    0  a  1
    1  b  2


    A  B
    0  c  3
    1  d  4


    A  B
    0  a  1
    1  b  2
    2  c  3
    3  d  4


        A  B
    C 0  a  1
    1  b  2
    D 0  c  3
    1  d  4

注意到， 因为 ``concat()`` 保留了每个子DataFrame的index， 所以合并之后的DataFrame中， 每个index出现了两次。
我们可以通过设置 ``ignore_index=False`` 来解决这个问题.

合并数据merge


.. code:: python

    import pandas as pd
    df1 = pd.DataFrame([['a', 1], ['b', 2],['c',3]], columns=['A','B'])
    df2 = pd.DataFrame([['c', 3,  2], ['d', 4, 5]], columns=['A','B','C'])
    df3 = pd.concat([df1, df2], sort=True)
    print(df1)
    print(df2)
    print("\n")
    print(df3)
    df3 = pd.merge(df1,df2)
    print("\n")
    print(df3)
    df3 = pd.merge(df1,df2,how='outer')
    print("\n")
    print(df3)
    

.. code:: text

    A  B
    0  a  1
    1  b  2
    2  c  3
    A  B  C
    0  c  3  2
    1  d  4  5


    A  B    C
    0  a  1  NaN
    1  b  2  NaN
    2  c  3  NaN
    0  c  3  2.0
    1  d  4  5.0


    A  B  C
    0  c  3  2


    A  B    C
    0  a  1  NaN
    1  b  2  NaN
    2  c  3  2.0
    3  d  4  5.0




