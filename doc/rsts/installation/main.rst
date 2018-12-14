Installation
==========

Documents are available on the ReNom.jp web site.

URL: http://renom.jp/index.html

**Requirements**

- python:>=3.4

For required python packages, please refer to the `requirements.txt <https://github.com/ReNom-dev-team/ReNomTDA/blob/release/2.1.4/requirements.txt>`_.

**Install ReNomTDA package**

Linux user can install ReNomIMG from Wheel package.

Other os user can't install from Wheel package but can install from source.

The Wheel package is provided at:

https://grid-devs.gitlab.io/ReNomTDA/bin/renom_tda-VERSION-py3-none-any.whl

(VERSION is stands for actual version number e.g. 2.1.0)

You can install the wheel package with pip command:

.. code-block:: sh

    pip install https://grid-devs.gitlab.io/ReNomTDA/bin/renom_tda-2.1.4-py3-none-any.whl


**Install from source**

For install ReNomTDA, download the repository from following url.

.. code-block:: sh

    git clone https://github.com/ReNom-dev-team/ReNomTDA.git


Then move to the ReNomTDA directory, and install the modules using pip.

.. code-block:: sh

    cd ReNomTDA
    pip install -r requirements.txt
    pip install -e .


Next, build javascript application files.

.. code-block:: sh

    cd ReNomTDA/js
    npm install
    npm run build
