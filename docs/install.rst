============
Installation
============

The package can be installed directly from the Python Package Index or GitHub.
The version on GitHub might be more recent, as only stable versions are
uploaded to the Python Package Index.

PyPI
----

The version from PyPI can directly be installed using pip

.. code-block:: bash

    pip install diag-eff

GitHub
------

The most recent version from GitHub can be installed like:

.. code-block:: bash

    git clone https://github.com/schwemro/diag-eff.git
    cd diag-eff
    python setup.py install

Note
----

Depending on you OS, you might run into problems installing all requirements
in a clean Python environment. These problems are usually caused by the scipy
and numpy package, which might require to be compiled. Instead, We recommend to
use an environment manager like anaconda.
Then, the requirements can be installed like:

.. code-block:: bash

    conda install numpy, scipy
