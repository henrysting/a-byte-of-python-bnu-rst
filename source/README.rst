前言
====

本文档是北京师范大学天文系的“Python科学计算”课程讲义，用于为低年级本科生介绍脚本编程语言Python 3 的基本知识，适合没有任何编程基础的读者。本讲义是在 **Swaroop C H**\ 所创作的《简明 Python 教程》（Byte of
Python）一书的基础上修订、补而来。

《简明 Python 教程》是由 **Swaroop C H** 编写，旨在介绍 Python 语言的开源图书。2005 年，\ **沈洁元** 将本书的 1.20 版（基于python
2.3）译为中文并公开发布，将本书的译名定为《简明 Python 教程》。2017年，\ **漠伦** 根据原书 4.0 版（基于python 3.5）重新翻译，并沿用同样的译名。
2018年，北师大天文系师生在本书4.08c 版的基础上进行了调整和修改，用作Python课程的参考资料。2020年9月将其由Gitbook转换为Sphinx格式，
所有代码在3.7下测试通过。最近更新于2020年10月09日。

版本
----

-  英文原版： https://python.swaroopch.com/
   （\ `英文版源文件 <https://github.com/swaroopch/byte-of-python>`__\ ）
-  沈洁元译本： http://woodpecker.org.cn/abyteofpython_cn/chinese/
   （\ `沈洁元译本源文件 <https://github.com/onion7878/A-Byte-of-Python-CN>`__\ ）
-  漠伦译本：
   https://legacy.gitbook.com/book/lenkimo/byte-of-python-chinese-edition/
   （\ `漠伦译本源文件 <https://github.com/LenKiMo/byte-of-python>`__\ ）
-  北师大天文系修改版：\ https://a-byte-of-python-bnu.gitbook.io/book/\ （\ `天文系修改版源文件 <https://github.com/WuShichao/a-byte-of-python-bnu>`__\ ）

版权
----

本讲义采用与原书相同的 `知识共享 署名-相同方式共享 国际 4.0 协议（CC BY-SA Intl. 4.0） <https://creativecommons.org/licenses/by-sa/4.0/deed.zh>`__
进行授权。也就是说，只要你给出署名和原始出处，并用_完全相同_的协议进行授权，就可以自由地\ **分享**\ 或\ **改编**\ 本作品，无论出于何种目的，甚至包括商业性用途。

另请注意：

-  请\ **不要**\ 销售本书的电子或印刷拷贝，除非你明确声明这些拷贝副本\ **并非**\ 来自本书的创作者。
-  在分发时\ **务必**\ 在文档的介绍性描述或封面、序言中提供回溯至本书原书以及本译本的链接，并明确指出本书之原文与译本可在上述链接处获取。
-  除非另有声明，本书所提供的所有代码与脚本均采用 `3-clause BSD License <http://www.opensource.org/licenses/bsd-license.php>`__  进行授权。

扩展阅读
--------

-  《Python入门》(\ `The Python Tutorial <https://docs.python.org/3/tutorial/index.html>`__) 官方指南
-  《笨办法学Python 3》[美] 泽德 A. 肖, 人民邮电出版社 (2018)
-  《Python物理学高效计算》[美]安东尼·斯科普斯 凯瑟琳·赫夫, 人民邮电出版社 (2018)
-  《Python编程快速上手》[美]Albert Sweigart，人民邮电出版社, (2016)
-  《Python编程：从入门到实践》 [美]埃里克·马瑟斯，人民邮电出版社, (2016)
-  `Python 3 官方文档 <https://docs.python.org/zh-cn/3/index.html>`__
-  《\ `深入 Python 3 <https://woodpecker.org.cn/diveintopython3/>`__\ 》(Dive Into Python 3), Mark Pilgrim, (2009)
