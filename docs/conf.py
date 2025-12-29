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
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath(os.path.join('..', 'desilike')))
from _version import __version__

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme'
]

# -- Project information -----------------------------------------------------

project = 'desilike'
copyright = '2022, cosmodesi'
author = 'cosmodesi'

# The full version, including alpha/beta/rc tags
release = __version__

html_theme = 'furo'

autodoc_mock_imports = ['cosmoprimo', 'mpi4py']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

git_repo = 'https://github.com/cosmodesi/desilike.git'
git_root = 'https://github.com/cosmodesi/desilike/blob/main/'

extlinks = {'root': (git_root + '%s', '%s')}

intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy/', None)
}

# Thanks to: https://github.com/sphinx-doc/sphinx/issues/4054#issuecomment-329097229
def _replace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result


ultimate_replacements = {
    '{gitrepo}': git_repo
}


def skip_some_classes(app, what, name, obj, skip, options):
    # Skip those objects that are not imported at top-level
    import inspect
    try:
        fn = sys.modules[getattr(obj, '__module__')].__file__
    except (KeyError, AttributeError):
        return skip
    init_fn = os.path.join(os.path.dirname(fn), '__init__.py')
    try:
        with open(init_fn, 'r') as file:
            if name not in file.read():
                # If a 'BaseClass' subclass and not imported in '__init__.py', skip
                try:
                    return any('BaseClass' in cls.__name__ for cls in inspect.getmro(obj))
                except AttributeError:
                    return skip
    except FileNotFoundError:
        return skip
    return skip


def setup(app):
    app.add_config_value('ultimate_replacements', {}, True)
    app.connect('source-read', _replace)
    app.connect('autodoc-skip-member', skip_some_classes)


autoclass_content = 'both'

