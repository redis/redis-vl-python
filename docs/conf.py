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
sys.path.insert(0, os.path.abspath('.'))

print(f"Sphinx is using Python executable at: {sys.executable}", flush=True)
print(f"Python version: {sys.version}", flush=True)

# -- Project information -----------------------------------------------------

project = 'RedisVL'
copyright = '2024, Redis Inc.'
author = 'RedisVentures'
version = "0.2.0"

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    "sphinx_design",
    "sphinx_copybutton",
    "_extension.gallery_directive",
    'nbsphinx',
    "myst_nb",
    "sphinx_favicon"
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files=["css/custom.css", "css/sidebar.css"]
html_js_files=["js/sidebar.js"]
html_title = "RedisVL"
html_context = {
   "default_mode": "dark"
}
html_logo = "_static/Redis_Favicon_32x32_Red.png"
html_favicon = "_static/Redis_Favicon_32x32_Red.png"
html_context = {
    "github_user": "RedisVentures",
    "github_repo": "RedisVL",
    "github_version": "main",
    "doc_path": "docs",
}

import redisvl
version = redisvl.__version__
# This allows us to use ::: to denote directives, useful for admonitions
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

html_theme_options = {
    "logo": {
        "text": "RedisVL",
        "image_dark": "_static/redis-cube-red-white-rgb.svg",
        "alt_text": "RedisVL",
    },
    "use_edit_page_button": True,
    "show_toc_level": 4,
    "show_nav_level": 3,
    "navigation_depth": 5,
    "navbar_align": "content",  # [left, content, right] For testing that the navbar items align properly
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "switcher": {
        "json_url": "_static/version_names.json",
        "version_match": version,
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/RedisVentures/RedisVL",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ]
}

html_sidebars = {
    "**": ["custom_sidebar.html"]
}

autoclass_content = 'both'
add_module_names = False

nbsphinx_execute = 'never'
jupyter_execute_notebooks = "off"

# -- Options for autosummary/autodoc output ------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"

# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../src/redisvl"]
autoapi_keep_files = True
autoapi_root = "api"
autoapi_member_order = "groupwise"


# -- favicon options ---------------------------------------------------------

# see https://sphinx-favicon.readthedocs.io for more information about the
# sphinx-favicon extension

favicons = [
    # generic icons compatible with most browsers
    "Redis_Favicon_32x32_Red.png",
    "Redis_Favicon_16x16_Red.png",
    "Redis_Favicon_144x144_Red.png",
]
