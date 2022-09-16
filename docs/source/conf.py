# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import datetime
import sphinx_rtd_theme
import doctest
import icenet

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

autosummary_generate = True
templates_path = ['_templates']

source_suffix  = '.rst'
master_doc     = 'index'

add_module_names = False


### Author

author    = 'Mikael Mieskolainen, Blackett Laboratory, Imperial College London'
project   = 'icenet'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = icenet.__version__
release = icenet.__release__


### HTML setup

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}


### https://sphinx-themes.org/

## "Alabaster" scheme
#html_theme       = 'alabaster'

## "RTD" scheme 
html_theme      = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
    'navigation_depth': 3,
}

#html_logo = '_static/img/logo.svg'
html_static_path = ['_static']
#html_static_path = []

#html_context = {'css_files': ['_static/css/custom.css']}


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
        ]
        return True if name in members else skip

    app.connect('autodoc-skip-member', skip)
