# ReNomTDA v2.1.5 beta

Documents are available on the ReNom.jp web site.

- http://renom.jp/index.html


## Recommended Environment
- OS: Ubuntu 16.04
- Browser: Google Chrome(version 63.0.3239.132)
- Python: >=3.5


## Requirements

- bottle 0.12.16
- networkx 1.11


## Install ReNomTDA package

Linux user can install ReNomIMG from Wheel package.

Other os user can't install from Wheel package but can install from source.

The Wheel package is provided at:

https://grid-devs.gitlab.io/ReNomTDA/bin/renom_tda-VERSION-py3-none-any.whl

(VERSION is stands for actual version number e.g. 2.1.5)

You can install the wheel package with pip3 command::

```
pip install https://grid-devs.gitlab.io/ReNomTDA/bin/renom_tda-2.1.5-py3-none-any.whl
```

# Install from source

For install ReNomTDA, download the repository from following url.

```
git clone https://github.com/ReNom-dev-team/ReNomTDA.git
```

Then move to the ReNomTDA directory, and install the modules using pip.

```
cd ReNomTDA
pip install -r requirements.txt
python setup.py build
pip install -e .
```


## License

“ReNomTDA” is provided by GRID inc., as subscribed software.  By downloading ReNom, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNom for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNom for academic, educational and/or product evaluation purposes, you may use ReNom royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.
