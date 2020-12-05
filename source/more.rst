更多
====

到现在，我们的介绍已经涵盖了你将使用到的 Python
的大部分方面。在本章中，我们将介绍一些其它的方面，来让我们对 Python
的认识更加全面。

传递元组
--------

你可曾希望从一个函数中返回两个不同的值？你能做到的。只需要使用一个元组。

.. code:: python

   >>> def get_error_details():
   ...     return (2, 'details')
   ...
   >>> errnum, errstr = get_error_details()
   >>> errnum
   2
   >>> errstr
   'details'

要注意到 ``a, b = <some expression>``
的用法会将表达式的结果解释为具有两个值的一个元组。

这也意味着在 Python 中交换两个变量的最快方法是：

.. code:: python

   >>> a = 5; b = 8
   >>> a, b
   (5, 8)
   >>> a, b = b, a
   >>> a, b
   (8, 5)

特殊方法
--------

诸如 ``__init__`` 和 ``__del__`` 等一些方法对于类来说有特殊意义。

特殊方法用来模拟内置类型的某些行为。举个例子，如果你希望为你的类使用
``x[key]``
索引操作（就像你在列表与元组中使用的那样），那么你所需要做的只不过是实现
``__getitem__()`` 方法，然后你的工作就完成了。如果你试图理解它，就想想
Python 就是对 ``list`` 类这样做的！

下面的表格列出了一些有用的特殊方法。如果你想了解所有的特殊方法，请\ `参阅手册 <http://docs.python.org/3/reference/datamodel.html#special-method-names>`__\ 。

-  ``__init__(self, ...)``

   -  这一方法在新创建的对象被返回准备使用时被调用。

-  ``__del__(self)``

   -  这一方法在对象被删除之前调用（它的使用时机不可预测，所以避免使用它）

-  ``__str__(self)``

   -  当我们使用 ``print`` 函数时，或 ``str()`` 被使用时就会被调用。

-  ``__lt__(self, other)``

   -  当_小于_运算符（<）被使用时被调用。类似地，使用其它所有运算符（+、>
      等等）时都会有特殊方法被调用。

-  ``__getitem__(self, key)``

   -  使用 ``x[key]`` 索引操作时会被调用。

-  ``__len__(self)``

   -  当针对序列对象使用内置 ``len()`` 函数时会被调用

单语句块
--------

我们已经见识过每一个语句块都由其自身的缩进级别与其它部分相区分。
是这样没错，不过有一个小小的警告。如果你的语句块只包括单独的一句语句，那么你可以在同一行指定它，例如条件语句与循环语句。下面这个例子应该能比较清楚地解释：

.. code:: python

   >>> flag = True
   >>> if flag: print('Yes')
   ...
   Yes

注意，单个语句是在原地立即使用的，它不会被看作一个单独的块。尽管，你可以通过这种方式来使你的程序更加小巧，但除非是为了检查错误，我强烈建议你避免使用这种快捷方法，这主要是因为如果你不小心使用了一个“恰到好处”的缩进，它就很容易添加进额外的语句。

Lambda 表格
-----------

``lambda`` 语句可以创建一个新的函数对象。从本质上说，\ ``lambda``
需要一个参数，后跟一个表达式作为函数体，这一表达式执行的值将作为这个新函数的返回值。

案例（保存为 ``more_lambda.py``\ ）：

.. raw:: html

   <pre><code class="lang-python">{% include "./programs/more_lambda.py" %}</code></pre>

输出：

.. raw:: html

   <pre><code>{% include "./programs/more_lambda.txt" %}</code></pre>

**它是如何工作的**

要注意到一个 ``list`` 的 ``sort`` 方法可以获得一个 ``key``
参数，用以决定列表的排序方式（通常我们只知道升序与降序）。在我们的案例中，我们希望进行一次自定义排序，为此我们需要编写一个函数，但是又不是为函数编写一个独立的
``def`` 块，只在这一个地方使用，因此我们使用 Lambda
表达式来创建一个新函数。

列表推导
--------

列表推导（List
Comprehension）用于从一份现有的列表中得到一份新列表。想象一下，现在你已经有了一份数字列表，你想得到一个相应的列表，其中的数字在大于
2 的情况下将乘以 2。列表推导就是这类情况的理想选择。

案例（保存为 ``more_list_comprehension.py``\ ）：

.. raw:: html

   <pre><code class="lang-python">{% include "./programs/more_list_comprehension.py" %}</code></pre>

输出：

.. raw:: html

   <pre><code>{% include "./programs/more_list_comprehension.txt" %}</code></pre>

**它是如何工作的**

在本案例中，当满足了某些条件时（\ ``if i > 2``\ ），我们进行指定的操作（\ ``2*i``\ ），以此来获得一份新的列表。要注意到原始列表依旧保持不变。

使用列表推导的优点在于，当我们使用循环来处理列表中的每个元素并将其存储到新的列表中时时，它能减少样板（Boilerplate）代码的数量。

在函数中接收元组与字典
----------------------

有一种特殊方法，即分别使用 ``*`` 或 ``**``
作为元组或字典的前缀，来使它们作为一个参数为函数所接收。当函数需要一个可变数量的实参时，这将颇为有用。

.. code:: python

   >>> def powersum(power, *args):
   ...     '''Return the sum of each argument raised to the specified power.'''
   ...     total = 0
   ...     for i in args:
   ...         total += pow(i, power)
   ...     return total
   ...
   >>> powersum(2, 3, 4)
   25
   >>> powersum(2, 10)
   100

因为我们在 ``args`` 变量前添加了一个 ``*``
前缀，函数的所有其它的额外参数都将传递到 ``args``
中，并作为一个元组予以储存。如果采用的是 ``**``
前缀，则额外的参数将被视为字典的键值—值配对。

``assert`` 语句
---------------

``assert``
语句用以断言（Assert）某事是真的。例如说你非常确定你正在使用的列表中至少包含一个元素，并想确认这一点，如果其不是真的，就抛出一个错误，\ ``assert``
语句就是这种情况下的理想选择。当语句断言失败时，将会抛出
``AssertionError``\ 。

.. code:: python

   >>> mylist = ['item']
   >>> assert len(mylist) >= 1
   >>> mylist.pop()
   'item'
   >>> assert len(mylist) >= 1
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   AssertionError

你应该明智地选用 ``assert``
语句。在大多数情况下，它好过捕获异常，也好过定位问题或向用户显示错误信息然后退出。

装饰器
------

装饰器（Decorators）是应用包装函数的快捷方式。这有助于将某一功能与一些代码一遍又一遍地“包装”。举个例子，我为自己创建了一个
``retry``
装饰器，这样我可以将其运用到任何函数之中，如果在一次运行中抛出了任何错误，它就会尝试重新运行，直到最大次数
5
次，并且每次运行期间都会有一定的延迟。这对于你在对一台远程计算机进行网络调用的情况十分有用：

.. raw:: html

   <pre><code class="lang-python">{% include "./programs/more_decorator.py" %}</code></pre>

输出：

.. raw:: html

   <pre><code>{% include "./programs/more_decorator.txt" %}</code></pre>

**它是如何工作的**

请参阅：

-  http://www.ibm.com/developerworks/linux/library/l-cpdecor.html
-  http://toumorokoshi.github.io/dry-principles-through-python-decorators.html

Python 2 与 Python 3 的不同
---------------------------

请参阅：

-  `“Six” library <http://pythonhosted.org/six/>`__
-  `Porting to Python 3 Redux by
   Armin <http://lucumr.pocoo.org/2013/5/21/porting-to-python-3-redux/>`__
-  `Python 3 experience by
   PyDanny <http://pydanny.com/experiences-with-django-python3.html>`__
-  `Official Django Guide to Porting to Python
   3 <https://docs.djangoproject.com/en/dev/topics/python3/>`__
-  `Discussion on What are the advantages to python
   3.x? <http://www.reddit.com/r/Python/comments/22ovb3/what_are_the_advantages_to_python_3x/>`__

总结
----

我们在本章中介绍了有关 Python 的更多功能。虽然我们还未涵盖到 Python
的所有功能，不过在这一阶段，我们已经涉猎了大多数你将在实践中遇到的内容。这足以让你开始编写任何你所期望的程序。
