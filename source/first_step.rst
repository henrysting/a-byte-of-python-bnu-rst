基础：初次运行
================

接下来我们通过尝试在 Python 中运行一个经典的“Hello
World”程序，来了解如何编写、保存与运行 Python 程序。

Python程序有两种运行方式——使用交互式解释器或使用脚本。下面我们将学习这两种方法。

终端（Terminal）
----------------

终端是一台机器和外界（人）进行交互（命令输入输出）的窗口。在图形化界面（Windows系统）出现之前，它是计算机唯一的交互方式。
虽然现在用户的大部分操作被软件中的窗口和按钮取代，但其背后的输入仍是通过具体的命令和接口来完成的。很多时候，终端仍然是更快捷高效的交互方式。

Windows 操作系统
~~~~~~~~~~~~~~~~ 

在Windows系统中启动终端可以通过\ ``开始``\ ，\ ``运行``\ （快捷键为\ ``Win``\ +
R），输入 cmd 并回车来实现。 然后在命令行中输入 ``python3`` 或 ``python3``
并按下回车键来打开 Python 提示符（Python Prompt）。
也可以输入``ipython`` 进入增强版的命令行环境。
Windows用户只要正确地设置了PATH变量，应该可以从命令行启动解释器。
如果安装时没有将Python加到系统路径当中，则需要修改环境变量，或者使用程序菜单中的 Anaconda
Prompt 程序启动。

Linux 或 MacOS系统
~~~~~~~~~~~~~~~~~~

在你的操作系统中开启终端程序然后通过输入 ``python`` 或 ``python3`` 并按下回车键来打开
Python 提示符（Python Prompt）。

解释器环境
----------

解释器启动之后，其中的输入就和平台无关了。

在 Python 启动后，你会看见在你能开始输入内容的地方出现了 ``>>>``
。它是 *Python 解释器提示符（Python Interpreter Prompt）* ，
是区别终端环境和Python解释器环境的标志。

在提示符后输入：

.. code:: python

   print("Hello World")

输入完成后按下回车键。你将会看到屏幕上打印出 ``Hello World`` 字样。

下面是一个在 Linux 电脑上你能够看见结果的示例。有关 Python
软件的细节将会因为你使用的电脑而有所不同，但是从提示符（如 ``>>>``
）开始部分应该是相同的，不会受到操作系统的影响。

.. code:: console

   nightwing@halo:~$ python
   Python 3.7.3 (default, Mar 27 2019, 22:11:17) 
   [GCC 7.3.0] :: Anaconda, Inc. on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> print("Hello World")
   Hello World
   >>>

你自然会注意到，Python
会立即给你输出了一行结果！你刚才所输入的便是一句独立的 Python *语句*
。我们使用 ``print`` 命令来打印你所提供的信息。在这里，我们提供了文本 
``Hello World`` ，然后它便被迅速地打印到了屏幕上。

获取帮助
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果你需要获得 Python 中有关任何函数或语句的快速信息，你可以使用其内置的
``help`` 功能。这在使用解释器环境时十分有用。例如，运行 ``help('len')``
命令——这将显示出有关 ``len`` 函数的帮助，了解其是用来计算项目数量的。

小贴士：按下 ``q`` 键可以退出帮助。

类似地，你可以通过此方式获得几乎所有有关 Python 的信息。使用 ``help()``
命令来了解有关 ``help`` 它本身的更多信息吧！

如果你需要获得有关 ``return``
这类运算符的帮助，你需要做的就是将它们放在引号中，就像
``help('return')`` 这般，这样 Python 就不会混淆我们正在试图做的事情。


退出解释器环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

退出解释器环境,在LINUX下可以通过按下 ``ctrl + d`` 组合键,在WINDOWS下可以通过按下 ``ctrl + z`` 组合键,或是在任意操作系统的命令行界面输入 ``exit()``
（注意：务必包含括号 ``()``\ ）并敲下回车实现。


编辑器
------

要创建 Python
源代码文件，我们需要一款能够输入并保存代码的编辑器软件。一款优秀的面向程序员的编辑器能够帮助你的编写源代码文件工作变得轻松得多。故而选择一款编辑器确实至关重要。你要像挑选你想要购买的汽车一样挑选你的编辑器。一款优秀的编辑器能够帮助你更轻松地编写Python程序，使你的编程之旅更加舒适，并助你找到一条更加安全且快速的道路到达你的目的地（实现你的目标）。一个好的代码编辑器应该具有下面几个功能：

- 支持语法高亮（syntax highlighting）。能够以不同的颜色和字体显示编程内容，从而提高代码可读性。
- 显示空白字符（whitespace character）。能够显示空格、制表符、换行符等空白字符，方便检查Python所要求的缩进。
- 设定等宽字体（monospaced font）。采用 *Consolas* 、 *DejaVu Sans Mono* 、 *Menlo* 、 *Source Code Pro* 之类的等宽字体，以便能够明确地区分0和O，l和I，\’和\'等相近字形。如果代码中包含中文，最好使用 *Microsoft Yahei Mono* （微软雅黑）、 *Noto Sans CJK* (思源黑体) 等支持中文的等宽字体。
- 支持代码快捷执行。能够通过菜单栏或快捷键直接执行代码并检查结果。

|image25| |image26|

如果你对于从哪开始还没有概念，我推荐你使用\ `SciTE <https://www.scintilla.org/SciTE.html>`__\， \ `VSCode <https://code.visualstudio.com/>`__\
或者\ `PyCharm 教育版 <https://www.jetbrains.com/pycharm-edu/>`__\ 软件，它们在 Windows、Mac OS X、GNU/Linux 上都可以运行。

如果你正在使用 Windows 系统，\ **千万不要用记事本**——这是一个很糟糕的选择，因为它没有语法加亮功能，不支持文本缩进功能。我们后面会了解这些功能究竟有多重要。一款好的编辑器能够极大地提高工作效率。

如果你已是一名经验丰富的程序员，那你一定在用\ `Vim <http://www.vim.org>`__\ 或\ `Emacs <http://www.gnu.org/software/emacs/>`__\
了。无需多言，它们都是强大的编辑器，用它们来编写 Python 程序绰绰有余。

如果你有时间学习 Vim 或Emacs，那么我自是强烈推荐你学习它们二者中的一款，它们将在长远意义上对你裨益颇深。当然，正如我先前所推荐的，初学者可以以
PyCharm 开始，从而在此刻专注于学习 Python 而不是编辑器。

再次重申，请选择一款合适的编辑器——它能够让编写 Python程序变得更加有趣且容易。

Visual Studio Code（VSCode）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这是微软公司开发的一款跨平台开源编辑器，具有丰富的功能和扩展。官方网站为
`http://code.visualstudio.com/ <https://code.visualstudio.com/>`__\ 。
装好之后还需要在扩展(Extension)中安装\ ``中文语言包``\ 和\ ``python``\ 支持。

1.在官网下选择对应版本并下载

|image21|
 
2.同意协议，点击确定

|image22|

3.勾选需要的选项，在WIN10下可以勾选添加到PATH选项，可以省去配置环境变量，

|image23|

4.点击并等待安装完成

5.配置插件。VS CODE需要对应的插件才能运行不同编程语言的程序，这里我们在左上角的搜索框搜索 ``python`` ,选择微软官方发布的插件（更新比较及时），点击安装即可。

|image24|


Vim
~~~

1. 安装 `Vim <http://www.vim.org>`__\ 。

   -  Mac OS X 应该通过 `HomeBrew <http://brew.sh/>`__ 来安装 ``macvim``
      包。
   -  Windows 用户应该通过 `Vim
      官方网站 <http://www.vim.org/download.php>`__
      下载“自安装可执行文件”。
   -  GNU/Linux 用户应该通过他们使用的发行版的软件仓库获取 Vim。例如
      Debian 与 Ubuntu 用户可以安装 ``vim`` 包。

2. 安装 `jedi-vim <https://github.com/davidhalter/jedi-vim>`__ 插件为
   Vim 增添自动完成功能。
3. 安装与之相应的 ``jedi`` Python 包：\ ``pip install -U jedi``

Emacs
~~~~~

1. 安装 `Emacs 24+ <http://www.gnu.org/software/emacs/>`__\ 。

   -  Mac OS X 用户应该从 http://emacsformacosx.com 获取 Emacs。
   -  Windows 用户应该从 http://ftp.gnu.org/gnu/emacs/windows/ 获取
      Emacs。
   -  GNU/Linux 用户应该从他们使用的发行版的软件仓库获取 Emacs。如
      Debian 和 Ubuntu 用户可以安装 ``emacs24`` 包。

2. 安装 `ELPY <https://github.com/jorgenschaefer/elpy/wiki>`__\ 。

集成开发环境(IDE)
-----------------
如果需要更丰富的功能，如检查并提示错误，进行代码跟踪和调试，则需要使用集成开发环境(IDE)。常见的IDE包括：
\ `Jupyter Notebook <https://jupyter.org/install>`__\ , \ `Spyder <https://www.spyder-ide.org/>`__\, \ `Pycharm <https://www.jetbrains.com/pycharm/>`__\等。其中Jupyter Notebook和Spyder都是Anaconda默认包含的工具，无需额外安装。
\ **注意：在进行编辑之前，请确认你的代码使用英文字体。部分中文字体的字符"O"与"0"无法区分，而且中文字符很容易导致解释器出错。**

IDLE
~~~~

IDLE
是Python语言核心包自带的跨平台开发工具。IDLE旨在为初学者提供一个简单的开发调试环境。IDLE使用Python和Tkinter
GUI工具包编写。但是这个软件功能有限，兼容性不好，甚至无法与终端输出保持一致。不建议使用。
|IDLE界面|

Jupyter
~~~~~~~~~~~~~~~~

`Jupyter Notebook <https://jupyter.org/>`__
是一个基于网页的开源多语言开发环境。它将文档、代码、注释、结果、甚至图像整合到一个文档文件之中。
代码的撰写、调试以及分享都因此变得十分方便，非常适合教学和演示。
Anaconda中已经包含了Jupyter，不需要安装额外的软件。

.. figure:: ../pic/03.IDE/jupyter.png                                                                                                                                       
   :alt: jupyter

使用时，仅需要在终端下用 cd 命令切换到工作目录，
键入 jupyter notebook  命令后，便会自动打开默认网页浏览器进入环境
 
.. figure:: ../pic/03.IDE/Jupyter01.png

在页面右上角可以找到新建笔记的按钮

.. figure:: ../pic/03.IDE/Jupyter02.png

在新建的笔记页面就可以直接撰写并执行代码

.. figure:: ../pic/03.IDE/Jupyter03.png

注意：显示图像需要在使文件第一句话为JUPYTER内置魔法函数 ``%matplotlib inline``

Jupyter笔记是以单元格的形式来组织的，

.. figure:: ../pic/03.IDE/Jupyter04.png

单元格还有幻灯片、附件、标签等属性，可以根据需要灵活定制。

.. figure:: ../pic/03.IDE/Jupyter05.png

在jupyter中，我们可以使用Markdown语法来做笔记。Markdown是一种轻量级的、易于学习的标记语言，
用于格式化纯文本。

.. code:: text

   # 第一个Jupyter笔记
   ## 这是二级标题
   这是本*Python课程*学习的第一个*Jupyter*笔记示例。
   ~~你可以通过本笔记迅速掌握python语言。~~

   >Life is short,you need Python.
   
   下面演示了对html代码的支持：
   <ahref="http://202.112.85.96/python">Python课程资源</a>

.. figure:: ../pic/03.IDE/markdown.png

如果我们想在单元格内插入公式，那么需要使用LATEX语法。

在JUPYTER中新建标签类单元格并键入以下内容：

.. code:: text

   当前行内插入公式：$\int_0^{+\infty}x^2dx$

   下一行插入公式,黑体辐射公式：
   $$I_{\lambda}(\lambda,T)=\frac{2hc^2}{\lambda^{5}}\frac{1}{e^{\frac{hc}{\lambda kT}}-1}$$

点击单元格下方的运行按钮，得到结果：

.. figure:: ../pic/03.IDE/latex.png

Spyder
~~~~~~

`Spyder <https://www.spyder-ide.org/>`__\ 是Anaconda打包的集成开发环境。

.. figure:: ../pic/03.IDE/spyder.png
   :alt: Spyder

   Spyder

Eclipse
~~~~~~~

`Eclipse <https://www.eclipse.org/>`__\ 是基于 Java
的通用开发平台，功能强大，应用广泛。可以与PyDev结合搭建python开发平台。
|Eclipse|

###pycharm
 
PyCharm
~~~~~~~

`PyCharm <https://www.jetbrains.com/pycharm/>`__\ 是JetBrains公司开发的一款编辑器，其教育版免费。

当你打开 PyCharm 时，你会看见如下界面，点击 ``Create New Project`` ：

.. figure:: ../pic/03.IDE/pycharm_open.png
   :alt: 当你打开 PyCharm 时

   当你打开 PyCharm 时

选择 ``Pure Python`` ：

.. figure:: ../pic/03.IDE/pycharm_create_new_project.png
   :alt: PyCharm 新项目

   PyCharm 新项目

将你的项目路径位置中的 ``untitled`` 更改为 ``helloworld``
，你所看到的界面细节应该类似于下方这番：

.. figure:: ../pic/03.IDE/pycharm_create_new_project_pure_python.png
   :alt: PyCharm 项目细节

   PyCharm 项目细节

点击 ``Create`` 按钮。

对侧边栏中的 ``helloworld`` 右击选中，并选择 ``New`` -> ``Python File``
：

.. figure:: ../pic/03.IDE/pycharm_new_python_file.png
   :alt: PyCharm -&gt; New -&gt; Python File

   PyCharm -&gt; New -&gt; Python File

你会被要求输入名字，现在输入 ``hello`` ：

.. figure:: ../pic/03.IDE/pycharm_new_file_input.png
   :alt: PyCharm 新文件对话框

   PyCharm 新文件对话框

现在你便可以看见一个新的文件已为你开启：

.. figure:: ../pic/03.IDE/pycharm_hello_open.png
   :alt: PyCharm hello.py 文件

   PyCharm hello.py 文件

删除那些已存在的内容，现在由你自己输入以下代码：

.. code:: python

   print("hello world")

现在右击你所输入的内容（无需选中文本），然后点击 ``Run 'hello'`` 。

.. figure:: ../pic/03.IDE/pycharm_run.png
   :alt: PyCharm Run ‘hello’

   PyCharm Run ‘hello’

此刻你将会看到你的程序所输出的内容（它所打印出来的内容）：

.. figure:: ../pic/03.IDE/pycharm_output.png
   :alt: PyCharm 输出内容

   PyCharm 输出内容

嚯！虽然只是刚开始的几个步骤，但从今以后，每当我们要求你创建一个新的文件时，记住只需在
``helloworld`` 上右击并选择 -> ``New`` -> ``Python File``
并继续如上所述步骤一般输入内容并运行即可。

你可以在\ `PyCharm
Quickstart <https://www.jetbrains.com/pycharm-educational/quickstart/>`__\
页面找到有关 PyCharm 的更多信息。


VSCode
~~~~~~~

使用方法同以上类似。先打开一个你想保存.py文件的文件夹，然后编辑并允许即可。需要注意的是你需要编辑.vscode文件夹下的launch.json才能使vscode按照一个python的文件运行它。

.. figure:: ../pic/03.IDE/vscode_edit1.png
   :alt: VSCODE打开文件

.. figure:: ../pic/03.IDE/vscode_edit2.png
   :alt: VSCODE打开文件夹（工作区域）
 
我们需要为launch.json配置如下内容

.. code:: javascript

    {
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    
    "version": "0.2.0",
    "configurations": [
        
        {
            "name": "Python: Current File (Integrated Terminal)",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Current File (External Terminal)",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "externalTerminal"
        }
    ]
    }

其中的关键部分在于 "type" ， "request" ， "program" 。

type，必填项，调试类型，也就是以什么语言读这个文件；

request，必填项，有两种类型，分别是 launch 和 attach，launch 就是以 debug 模式启动调试（开启一个新终端），attach 就是附加到已经启动的进程开启 debug 模式并调试（使用已有的终端）；

program，程序的启动入口（在哪里找到文件），$file代表当前的.py路径;

.. figure:: ../pic/03.IDE/vscode_edit3.png
   :alt: VSCODE编辑launch.json
   
使用脚本
----------

启动你所选择的编辑器，输入如下程序并将它保存为 ``hello.py`` 。

如果你正在使用
PyCharm，我们已经讨论过\ :doc:`如何从脚本中运行它 <./first_step>` \ 。

对于其它编辑器，打开一个新文件名将其命名为 ``hello.py``
，然后输入如下内容：

.. code:: python

   print("hello world")

你应当将文件保存到哪里？保存到任何你知道其位置与路径的文件夹。如果你不了解这句话是什么意思，那就创建一个新文件夹并用这一路径来保存并运行你所有的
Python 程序：

-  Mac OS X 上的 ``/tmp/py`` 。
-  GNU/Linux 上的 ``/tmp/py`` 。
-  Windows 上的 ``D:\\py`` 。

要想创建上述文件夹（在你正在使用的操作系统上），你可以在终端上使用
``mkdir`` 命令，如 ``mkdir /tmp/py`` 。

重要提示：你需要经常确认并确保你为文件赋予了 ``.py`` 扩展名，例如
``foo.py`` 。

要想运行你的 Python 程序：

1. 打开终端窗口（你可查阅先前的
   \ :doc:`安装 :<./installation>` \ 章节来了解应该怎么做）。
2. 使用 ``cd`` 命令来\ **改**\ 变\ **目**\ 录到你保存文件的地方，例如
   ``cd /tmp/py`` 。
3. 通过输入命令 ``python hello.py``
   来运行程序。程序的输出结果应如下方所示：

.. code:: text

   $ python hello.py
   hello world

.. figure:: ../pic/03.IDE/terminal_screenshot.png
   :alt: 在终端中运行程序的截图

   在终端中运行程序的截图

如果你得到了与上图类似的输出结果，那么恭喜你！你已经成功运行了你的第一个
Python程序。你也已经成功穿过了学习编程最困难的部分，即编写你的第一个程序！

如果你遇到错误，请确认是否已经 **正确地** 输入了上面所列出的内容，并尝试重新运行程序。要注意
Python 是区分大小写的，如 ``print`` 和 ``Print`` 是不同的——注意前者的 p
是小写的，而后者的 P
是大写的。此外，你需要确保每一行的第一个字符前面都没有任何空格或制表格——我们会在后面了解
\ :doc:`为什么这件事如此重要 <./basis>` \ 。

**它是如何工作的**

Python 程序是由 *语句*
所构成的。我们的第一个程序中只有一条语句。在这条语句中，我们调用
``print`` 命令来输出文本”hello world“。



总结
----

现在，你应该可以轻松地编写、保存并运行 Python 程序了。

从此你便成为一名 Python 用户了，现在让我们来学习更多有关 Python 的概念。

.. |IDLE界面| image:: ../pic/03.IDE/IDLE.png
.. |Eclipse| image:: ../pic/03.IDE/Eclipse.jpg
.. |image21| image:: ../pic/03.IDE/vscode1.png
.. |image22| image:: ../pic/03.IDE/vscode2.png
.. |image23| image:: ../pic/03.IDE/vscode3.png
.. |image24| image:: ../pic/03.IDE/vscode4.png
.. |image25| image:: ../pic/03.IDE/editor0.png
    :width: 49 %
.. |image26| image:: ../pic/03.IDE/editor1.png
    :width: 49 %

