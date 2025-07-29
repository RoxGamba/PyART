# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GWForge'
copyright = '2025, Simone Albanesi, Danilo Chiaramello, Rossella Gamba, Koustav Chandra'
author = 'Simone Albanesi, Danilo Chiaramello, Rossella Gamba, Koustav Chandra'
release = '0.0.1dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_nb',
              'sphinx.ext.duration',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.mathjax',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary', 
              'sphinx.ext.viewcode',             
              'sphinx.ext.napoleon',
              'sphinx_math_dollar'
              ]
extensions.append('autoapi.extension')
autoapi_dirs = ['../../PyART']
templates_path = ['_templates']
exclude_patterns = []
jupyter_execute_notebooks = "force"  # or "auto" / "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']