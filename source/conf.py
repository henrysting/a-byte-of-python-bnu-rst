# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'A-Byte-of-Python-BNU'
copyright = '2022, BNU-Astro'
author = 'BNU-Astro'

# The full version, including alpha/beta/rc tags
release = '1.2'

master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#~ extensions = ['rst2pdf.pdfbuilder']
#~ pdf_documents = [('index', u'rst2pdf', u'Sample rst2pdf doc', u'BNUAstro'),]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['/home/user/ywc/github/a-byte-of-python-bnu-rst/build']


latex_engine = 'xelatex'
latex_use_xindy = False
latex_elements = {
    'preamble': '''
\\usepackage{xeCJK}
\\usepackage{indentfirst}
\\setlength{\\parindent}{2em}
\\setCJKmainfont{Noto Serif CJK SC}
\\setCJKmonofont[Scale=0.9]{Noto Sans Mono CJK SC}
\\setCJKfamilyfont{song}{Noto Sans CJK SC}
\\setCJKfamilyfont{sf}{Noto Sans CJK SC}
'''

}