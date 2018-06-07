import os
import sys
import re
import shutil
import pathlib
import numpy
from setuptools import find_packages, setup
import distutils.command.build


if sys.version_info < (3, 4):
    raise RuntimeError('renom_img requires Python3')

DIR = str(pathlib.Path(__file__).resolve().parent)

requires = [
    "bottle", "matplotlib", "networkx", "numpy", "pandas", "scikit-learn", "scipy", "future"
]


entry_points = {
    'console_scripts': [
        'renom_tda = renom_tda.server.server:main',
    ]
}


class BuildNPM(distutils.command.build.build):
    """Custom build command."""

    def run(self):
        shutil.rmtree(os.path.join(DIR, 'renom_tda/server/.build'), ignore_errors=True)
        curdir = os.getcwd()
        try:
            jsdir = os.path.join(DIR, 'js')

            # skip if js directory not exists.
            if os.path.isdir(jsdir):
                os.chdir(jsdir)
                ret = os.system('npm install')
                if ret:
                    raise RuntimeError('Failed to install npm modules')

                ret = os.system('npm run build')
                if ret:
                    raise RuntimeError('Failed to build npm modules')

        finally:
            os.chdir(curdir)

        super().run()


setup(
    name="renom_tda",
    version="2.1.0",
    entry_points=entry_points,
    packages=['renom_tda'],
    install_requires=requires,
    include_package_data=True,
    zip_safe=True,
    cmdclass={
        'build': BuildNPM,
    }
)
