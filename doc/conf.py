# docs/conf.py

import os
import sys

sys.path.insert(0, os.path.abspath('.'))

project = 'G.R.A.C.E.'
author = 'Carlo Musolino'
copyright = "Carlo Musolino 2023"

extensions = [
    "breathe",
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    "sphinx.ext.extlinks",
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinxcontrib.jquery",
    "sphinx_tabs.tabs",
    "notfound.extension",
    "sphinx_copybutton",
    "sphinx_togglebutton"
]

templates_path = [os.path.join('/mnt/rafast/musolino/grace/doc', '_templates')]

breathe_projects = {'G.R.A.C.E.': "doxygen/xml"}
breathe_default_project = 'G.R.A.C.E.'

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "prev_next_buttons_location": None
}
html_static_path = [os.path.join('/mnt/rafast/musolino/grace/doc', '_static')]

html_title = "G.R.A.C.E. Project Documentation"
html_logo  = os.path.join('/mnt/rafast/musolino/grace/doc', '_static','images',"GRACE_logo_padded.png")
html_favicon = html_logo 
reference_prefix = ""
docs_title="Docs"
is_release = False
version = '0.5'
release = '0.5'
html_context = {
    "show_license": True,
    "docs_title": docs_title,
    "is_release": is_release,
    "current_version": version,
    "versions": (
        ("latest", "/"),
    ),
    "reference_links": {
        "API": f"{reference_prefix}/doxygen/html/index.html"
    }
}

source_suffix = ['.rst', '.md']
master_doc = 'index'

def setup(app):
    # theme customizations
    app.add_css_file("css/custom.css")
    app.add_js_file("js/custom.js")
    app.add_js_file("js/dark-mode-toggle.min.mjs", type="module")
