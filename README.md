# ReNomTDA

Documents are available on the ReNom.jp web site.

- http://renom.jp/index.html

## Requirements

- python2.7, 3.4
- numpy 1.13.0 1.12.1
- bottle 0.12.13
- matplotlib 2.0.2
- networkx 1.11
- pandas 0.20.3
- scikit-learn 0.18.2
- scipy 0.19.0
- pytest 3.0.7
- cython 0.24
- cuda-toolkit 8.0
- cudnn 5.1
- renom


## Installation

First clone the ReNomTDA repository.

	git clone https://github.com/ReNom-dev-team/ReNomTDA.git

Then move to the ReNomTDA folder, install the module using pip.

	cd ReNomTDA
	pip install -e .

## Precision

If you set an environment variable RENOM_PRECISION=64, 
calculations are performed with float64.

Default case, the precision is float32.


## License

“ReNom” is provided by GRID inc., as subscribed software.  By downloading ReNom, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNom for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNom for academic, educational and/or product evaluation purposes, you may use ReNom royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.
