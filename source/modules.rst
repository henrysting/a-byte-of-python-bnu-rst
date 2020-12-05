核心：模块
============

在上一章，你已经了解了如何在你的程序中通过定义一次函数工作来重用代码。那么如果你想在你所编写的别的程序中重用一些函数的话，应该怎么办？正如你可能想象到的那样，答案是模块（Modules）。

编写模块有很多种方法，其中最简单的一种便是创建一个包含函数与变量、以 ``.py`` 为后缀的文件。这个文件被称为脚本。

另一种方法是使用撰写 Python
解释器本身的本地语言来编写模块。举例来说，你可以使用 `C语言 <http://docs.python.org/3/extending/>`__\ 来撰写 Python
模块，并且在编译后，你可以通过标准 Python 解释器在你的 Python 代码中使用它们。

一个模块可以被其它程序导入并运用其功能。我们在使用 Python 标准库的功能时也同样如此。模块名与脚本的文件名相同，例如我们编写了一个
名为Items.py的脚本，则可在另外一个脚本中用 ``import Items`` 语句来导入它。首先，我们要了解如何使用标准库模块。

案例 (保存为 ``module_using_sys.py``):

.. code:: python

   import sys

   print('The command line arguments are:')
   for i in sys.argv:
       print(i)

   print('\n\nThe PYTHONPATH is', sys.path, '\n')

输出：

.. code:: console

   $ python module_using_sys.py we are arguments
   The command line arguments are:
   module_using_sys.py
   we
   are
   arguments


   The PYTHONPATH is ['/tmp/py',
   # 内容过多，未全部显示
   '/Library/Python/2.7/site-packages',
   '/usr/local/lib/python2.7/site-packages']



**它是如何工作的**

首先，我们通过 ``import`` 语句_导入 ``sys``
模块。基本上，这句代码将转化为我们告诉 Python
我们希望使用这一模块。 ``sys`` 模块包含了与 Python
解释器及其环境相关的功能，也就是所谓的 *系统* 功能（ *sys* tem）。

当 Python 运行 ``import sys`` 这一语句时，它会开始寻找 ``sys``
模块。在这一案例中，由于其是一个内置模块，因此 Python 知道在哪里找到它。

如果它不是一个已编译好的模块，即用 Python 编写的模块，那么 Python
解释器将从它的 ``sys.path`` 变量所提供的目录中进行搜索。如果找到了对应模块，则该模块中的语句将在开始运行，并 *能够* 为你所使用。在这里需要注意的是，初始化工作只需在我们 *第一次* 导入模块时完成。

``sys`` 模块中的 ``argv`` 变量通过使用点号予以指明，也就是 ``sys.argv`` 这样的形式。它清晰地表明了这一名称是 ``sys``
模块的一部分。这一处理方式的另一个优点是这个名称不会与你程序中的其它任何一个 ``argv`` 变量冲突。

``sys.argv`` 变量是一系列字符串的 *列表（List）*（列表将在\ :doc:`后面的章节 <./07.data_structures>` \ 予以详细解释）。具体而言，\ ``sys.argv``
包含了 *命令行参数（Command Line Arguments）* 这一列表，也就是使用命令行传递给你的程序的参数。

如果你正在使用一款 IDE 来编写并运行这些程序，请在程序菜单中寻找相关指定命令行参数的选项。

在这里，当我们运行 ``python module_using_sys.py we are arguments`` 时，我们通过 ``python`` 命令来运行 ``module_using_sys.py``
模块，后面的内容则是传递给程序的参数。 Python 将命令行参数存储在 ``sys.argv`` 变量中供我们使用。

在这里要记住的是，运行的脚本名称在 ``sys.argv`` 的列表中总会位列第一。因此，在这一案例中我们将会有如下对应关系：\ ``'module_using_sys.py'``
对应 ``sys.argv[0]``\ ，\ ``'we'`` 对应 ``sys.argv[1]``\ ，\ ``'are'`` 对应 ``sys.argv[2]``\ ，\ ``'arguments'`` 对应
``sys.argv[3]``\ 。要注意到 Python 从 0 开始计数，而不是 1。

``sys.path`` 内包含了导入模块的字典名称列表。你能观察到 ``sys.path``
的第一段字符串是空的——这一空字符串代表当前目录也是 ``sys.path`` 的一部分，它与 ``PYTHONPATH``
环境变量等同。这意味着你可以直接导入位于当前目录的模块。否则，你必须将你的模块放置在 ``sys.path`` 内所列出的目录中。

有关sys的其他函数可以在help()中或者文档中查找.

另外要注意的是在sys中提到的当前目录指的是程序启动的目录。这就需要在操作系统上进行操作。我们需要os模块为操作系统服务提供了可移植的（portable）的接口。你可以通过运行
``print(os.getcwd())`` 来查看你的程序目前所处在的目录。 



按字节码编译的 .pyc 文件
------------------------

导入一个模块是一件代价高昂的事情，因此 Python
引入了一些技巧使其能够更快速的完成。其中一种方式便是创建 *按字节码编译的（Byte-Compiled）* 文件，这一文件以
``.pyc`` 为其扩展名，是将 Python
转换成中间形式的文件（还记得\ :doc:`基础：背景介绍  <./about_python>` \ 一章中介绍的
Python 是如何工作的吗？）。这一 ``.pyc``
文件在你下一次从其它不同的程序导入模块时非常有用——它将更加快速，因为导入模块时所需要的一部分处理工作已经完成了。同时，这些按字节码编译的文件是独立于运行平台的。

注意：这些 ``.pyc`` 文件通常会创建在与对应的 ``.py``
文件所处的目录中。如果 Python
没有相应的权限对这一目录进行写入文件的操作，那么 ``.pyc``
文件将 *不会* 被创建。

.. _from-import-statement:

``from..import`` 语句
---------------------

如果你希望直接将 ``argv`` 变量导入你的程序（为了避免每次都要输入
``sys.``\ ），那么你可以通过使用 ``from sys import argv``
语句来实现这一点。

   **警告：**\ 一般来说，你应该尽量_避免_使用 ``from...import``
   语句，而去使用 ``import``
   语句。这是为了避免在你的程序中出现名称冲突，同时也为了使程序更加易读。

案例：

.. code:: python

   from math import sqrt
   print("Square root of 16 is", sqrt(16))

.. _module-name:

模块的 ``__name__``
-------------------

每个模块都有一个名称，而模块中的语句可以找到它们所处的模块的名称。这对于确定模块是独立运行的还是被导入进来运行的这一特定目的来说大为有用。正如先前所提到的，当模块第一次被导入时，它所包含的代码将被执行。我们可以通过这一特性来使模块以不同的方式运行，这取决于它是为自己所用还是从其它从的模块中导入而来。这可以通过使用模块的
``__name__`` 属性来实现。

案例（保存为 ``module_using_name.py``\ ）：

.. code:: python

   if __name__ == '__main__':
       print('This program is being run by itself')
   else:
       print('I am being imported from another module')

输出：

.. code:: console

   $ python module_using_name.py
   This program is being run by itself

   $ python
   >>> import module_using_name
   I am being imported from another module
   >>>

**它是如何工作的**

每一个 Python 模块都定义了它的 ``__name__`` 属性。如果它与 ``__main__``
属性相同则代表这一模块是由用户独立运行的，因此我们便可以采取适当的行动。

编写你自己的模块
----------------

除了标准库模块外，我们可以自己编写与导入模块。每个py文件都可以是一个模块，。接下来我们自己定义一个模块，

编写你自己的模块很简单，这其实就是你一直在做的事情！这是因为每一个
Python 程序同时也是一个模块,被其他模块导入。你只需要保证它以 ``.py``
为扩展名即可。下面的案例会作出清晰的解释。

创建如下文件 ``bnupy.py`` 并运行:

.. code:: python

   def hello():
      print("hello,world!")
   def bye():
      print("bye,world!")
      
   if __name__=="__main__":
      hello()
      bye()
      print("Test pass!")

.. code:: console

   D:>python bnupy.py 
   hello,world!
   bye,world!
   Test pass!
   
   D:>python
   >>> import bnupy
   >>> dir()
   >>> bnupy.hello()
   hello,world!
   >>> bnupy.bye()
   bye,world!

案例（保存为 ``mymodule.py``\ ）：

.. code:: python

   def say_hi():
       print('Hi, this is mymodule speaking.')

   __version__ = '0.1'

上方所呈现的就是简单的模块。正如你所看见的，与我们一般所使用的
Python 的程序相比其实并没有什么特殊的区别。我们接下来将看到如何在其它
Python 程序中使用这一模块。

要记住该模块应该放置于与其它我们即将导入这一模块的程序相同的目录下，或者是放置在
``sys.path`` 所列出的其中一个目录下。

另一个模块（保存为 ``mymodule_demo.py``\ ）：

.. code:: python

   import mymodule

   mymodule.say_hi()
   print('Version', mymodule.__version__)

输出：

.. code:: console

   $ python mymodule_demo.py
   Hi, this is mymodule speaking.
   Version 0.1

**它是如何工作的**

你会注意到我们使用相同的点符来访问模块中的成员。Python
很好地重用了其中的符号，这充满了“Pythonic”式的气息，这使得我们可以不必学习新的方式来完成同样的事情。

下面是一个使用 ``from...import`` 语法的范本（保存为
``mymodule_demo2.py``\ ）：

.. code:: python

   from mymodule import say_hi, __version__

   say_hi()
   print('Version', __version__)

``mymodule_demo2.py`` 所输出的内容与 ``mymodule_demo.py``
所输出的内容是一样的。

在这里需要注意的是，如果导入到 mymodule 中的模块里已经存在了
``__version__``
这一名称，那将产生冲突。这可能是因为每个模块通常都会使用这一名称来声明它们各自的版本号。因此，我们大都推荐最好去使用
``import`` 语句，尽管这会使你的程序变得稍微长一些。

你也可以使用：

.. code:: python

   import math as m  # 导入同时给缩写
   import sys, os # 同时导入多个模块

你还可以使用：

.. code:: python

   
   from mymodule import *

这将导入诸如 ``say_hi`` 等所有公共名称，但不会导入 ``__version__``
名称，因为后者以双下划线开头。如果只需要某一个或某几个函数，把 ``*`` 替换为想要的函数，并用逗号分开即可。

   **警告：**
   
   要记住你应该避免使用 ``import *`` 这种形式，即  ``from mymodule import *``  。因为不同的模块中可能包含相同的函数名。

   **Python 之禅**

   Python 的一大指导原则是“明了胜过晦涩”(Explicit is better than implicit)。你可以通过在 Python 中运行
   ``import this`` 来了解更多内容。

.. _dir-function:

``dir`` 函数
------------

内置的 ``dir()`` 函数能够返回由对象所定义的名称列表。
如果这一对象是一个模块，则该列表会包括函数内所定义的函数、类与变量。

该函数接受参数。 如果参数是模块名称，函数将返回这一指定模块的名称列表。
如果没有提供参数，函数将返回当前模块的名称列表。

案例：

.. code:: python

   >>> import sys

   
   >>> dir(sys)             # 列出 sys 模块中的属性名称
   ['__displayhook__', '__doc__', 'argv', 
   'builtin_module_names', 'version', 'version_info']
   # 此处只展示部分条目

   >>> dir()                # 给出当前模块的属性名称
   ['__builtins__', '__doc__',
   '__name__', '__package__','sys']

   # 创建一个新的变量 'a'
   >>> a = 5

   >>> dir()
   ['__builtins__', '__doc__', '__name__', '__package__', 'a']

   >>> del a              # 删除或移除一个名称

   >>> dir()
   ['__builtins__', '__doc__', '__name__', '__package__']

**它是如何工作的**

首先我们看到的是 ``dir`` 在被导入的 ``sys``
模块上的用法。我们能够看见它所包含的一个巨大的属性列表。

随后，我们以不传递参数的形式使用 ``dir``
函数。在默认情况下，它将返回当前模块的属性列表。要注意到被导入模块的列表也会是这一列表的一部分。

给了观察 ``dir`` 函数的操作，我们定义了一个新的变量 ``a``
并为其赋予了一个值，然后在检查 ``dir``
返回的结果，我们就能发现，同名列表中出现了一个新的值。我们通过 ``del``
语句移除了一个变量或是属性，这一变化再次反映在 ``dir``
函数所处的内容中。

关于 ``del``
的一个小小提示——这一语句用于 *删除* 一个变量或名称，当这一语句运行后，在本例中即
``del a``\ ，你便不再能访问变量 ``a``——它将如同从未存在过一般。

要注意到 ``dir()`` 函数能对 *任何* 对象工作。例如运行 ``dir(str)``
可以访问 ``str``\ （String，字符串）类的属性。

同时，还有一个
`vars() <http://docs.python.org/3/library/functions.html#vars>`__
函数也可以返回给你这些值的属性，但只是可能，它并不能针对所有类都能正常工作。

**如果你对这个函数有疑问，可以通过help(包名.函数名)的方式查看帮助**


模块管理工具pip
----------------------


•pip功能：提供非核心模块的管理功能

•Python3 下名称为pip3，但是可以设置默认pip对应的版本为python3，需要调节优先级

以下是几个常用的命令：

.. code:: console

   where pip 查看pip路径
   pip -V 查看当前版本
   pip -h 查看命令
   pip install / uninstall / list


但是pip只处理python代码，程序包(whl)等，对于一些复杂的模块调用，需要借助包管理软件。在此之前我们先介绍什么是包：

包
--

现在，你必须开始遵守用以组织你的程序的层次结构。变量通常位于函数内部，函数与全局变量通常位于模块内部。
如果你希望组织起这些模块的话，应该怎么办？这便是包（Packages）应当登场的时刻。


什么是包(package)

•模块：py文件，包含若干函数

•包：py文件夹，包含多个py文件（模块）

•创建一个文件夹bnupy，其中包含两个文件 

``bnupy1.py`` : 

.. code:: python

   def hello1():
      print("hello,world!")
   if __name__=="__main__":
      hello1()    
      print("Test pass!")
      
``bnupy2.py`` :

.. code:: python

   def bye2 ():
      print("bye-bye,world!")
   if __name__=="__main__":
      bye2 ()
      print("Test pass!")
      
.. code:: python

   >>> import bnupy.bnupy1
   >>> bnupy.bnupy1.hello1

包是指一个包含模块与一个特殊的 ``__init__.py`` 文件的文件夹，后者向
Python 表明这一文件夹是特别的，因为其包含了 Python 模块。每当import的时候，就会自动执行里面的函数。

.. code:: python

   __all__ = ["bnupy1", "bnupy2"]  

.. code:: python

   >>> from bnupyimport *
   >>> dir()
   >>> bnupy2.bye()

.. code:: python

   from . import bnupy1
   from . import bnupy2

.. code:: python

   >>> import bnupy
   >>> dir()
   >>> bnupy.bnupy2.bye()

让我们这样设想：你想创建一个名为“world”的包，其中还包含着
“asia”、“africa”等其它子包，同时这些子包都包含了诸如“india”、
“madagascar”等模块。

下面是你会构建出的文件夹的结构：

.. code:: text

   - <some folder present in the sys.path>/
       - world/
           - __init__.py
           - asia/
               - __init__.py
               - india/
                   - __init__.py
                   - foo.py
           - africa/
               - __init__.py
               - madagascar/
                   - __init__.py
                   - bar.py

包是一种能够方便地分层组织模块的方式.你将在
\ :doc:`标准库 <./standrad_library>`\
中看到许多有关于此的实例。

包管理工具conda
----------------

conda是包及其依赖项和环境的管理工具。它是为Python项目而创造，但可适用于多种语言。
conda包和环境管理器包含于Anaconda的所有版本当中。


* 适用语言：Python, R, Ruby, Lua, Scala, Java, JavaScript, C/C++, FORTRAN。

* 适用平台：Windows, macOS, Linux

* 用途：

# 快速安装、运行和升级包及其依赖项。

# 在计算机中便捷地创建、保存、加载和切换环境。

    如果你需要的包要求不同版本的Python，你无需切换到不同的环境，因为conda同样是一个环境管理器。仅需要几条命令，你可以创建一个完全独立的环境来运行不同的Python版本，同时继续在你常规的环境中使用你常用的Python版本。——Conda官方网站

以下是conda常用的命令。

.. code:: python

   conda--help           # 查询命令
   conda--version        # 查看版本
   conda install numpy   # 安装numpy库
   conda uninstall numpy # 卸载numpy库
   conda list            # 列出已有库

模块应用
------------

以下我们举一些使用额外功能包的例子。

下载 `特定页面  <http://202.112.85.96/python/ref>`__ 中的所有文件,
在Python中可使用html代码分析模块BeautifulSoup提取超链接。

.. code:: python

   import urllib.request
   url= 'http://202.112.85.96/python/ref'
   response = urllib.request.urlopen(url)
   html = response.read()
   #print(html)
   from bs4 import BeautifulSoup as bs
   soup = bs(html, 'html.parser')
   links = soup.findAll('a')
   # print(links)
   for a in links:
      print(url+a['href'])
      if  "jpg" in a[ 'href']:
         urllib.request.urlretrieve(url+a['href'], a['href'])

使用python进行图像处理

在数字图像处理中，使用若干个 ``m*n`` 矩阵代表 ``m*n`` 像素的图像。如果是灰度图像则是2维矩阵，
彩色图像则是3个二维矩阵合成的三维矩阵，每个矩阵分别由图像的RGB值之一或者色调、饱和度和明度三个值组成，
矩阵元素范围为0-255的整数。

图像处理

.. code:: python

   import matplotlib.pyplot as plt
   img = plt.imread('BNULogo.jpg')
   img_s= img[::2,::2]
   plt.imsave("logo_sml.png", img_s)

|image01|

.. code:: python

   img_c= img[400:600 ,400:600]
   plt.imsave("logo_crop.png", img_c)
   plt.show()
   
|image02|

.. code:: python

   import numpy as np
   import matplotlib.pyplot as plt
   comb = np.zeros([992,1280,3])
   fname= "opo0907h"
   for i in range(3):
      img = plt.imread(fname+"_" +str(i)+'.jpg')
      print(i,img.shape)
      comb[:,:,i] = img/255.
   plt.imshow(comb)
   plt.show()
   plt.imsave(fname+"_comb.jpg", comb)

.. code:: text

   0 (992, 1280)
   1 (992, 1280)
   2 (992, 1280)

|image03|

滤镜特效（Lomo）:

.. code:: python

   import numpy as np
   import matplotlib.pyplot as plt
   #读取原始图像
   img=plt.imread('IMG_1556_c.jpg')#returnRGB
   #获取图像行和列
   rows,cols=img.shape[:2]
   B=np.sqrt(img[:,:,2])*12
   B[B>255]=255
   G=img[:,:,1];R=img[:,:,0]
   img_f=np.stack((R,G,B),axis=2)
   img_f=img_f.astype('uint8')
   #~print(img_f[:3,:3,:])
   plt.imsave('IMG_1556_r.png', img_f)

|image04|

.. code:: python

   import os
   import numpy as np
   import matplotlib.pyplot as plt
   from PIL import Image
   im=Image.open(os.path.join(str(os.getcwd())+'/'+"BNULogo.jpg"))
   print(im.size)
   im2=im.resize((300,300),Image.ANTIALIAS)
   im2.save("bnu_logo_scaled.jpg",quality=90)
   
|image05|

.. code:: python

   im3=im.crop((400,400,600,600))
   im3.save("bnu_logo_croped.jpg")
   im4=im.rotate(45)
   plt.imshow(im4)
   plt.show()

|image06|

验证码生成

.. code:: python

   from PIL import Image, ImageDraw
   import pylab as plt
   nimg= Image.new('RGB',(60,30),'red')
   draw = ImageDraw.Draw(nimg)
   draw.text((10,10 ),'1 2 3')
   plt.imshow(nimg)
   plt.show()

|image07|



总结
----

如同函数是程序中的可重用部分那般，模块是一种可重用的程序。包是用以组织模块的另一种层次结构。Python
所附带的标准库就是这样一组有关包与模块的例子。

.. |image01| image:: ../pic/09.modules/logo_sml.png
.. |image02| image:: ../pic/09.modules/logo_crop.png
.. |image03| image:: ../pic/09.modules/opo0907h_comb.jpg
.. |image04| image:: ../pic/09.modules/IMG_1556_r.png
.. |image05| image:: ../pic/09.modules/bnu_logo_scaled.jpg
.. |image06| image:: ../pic/09.modules/bnu_logo_croped.jpg
.. |image07| image:: ../pic/09.modules/captcha.png