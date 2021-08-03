import os

from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup


base_path = os.path.dirname(__file__) + "/.."
nlp_losses_path = base_path + "/NLP_Losses/"
nlp_metrics_path = base_path + "/NLP_Metrics/"

setup(
    name="NLP_ModelTrainers",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/NLP_ModelTrainers/*.py")] + \
            [splitext(basename(path))[0] for path in glob("src/NLP_ModelTrainers/SentenceClassification/*.py")] + \
            [splitext(basename(path))[0] for path in glob("src/NLP_ModelTrainers/WordEmbedding/*.py")],
    install_requires=[
        "NLP_Losses @ file://localhost/%s" % nlp_losses_path,
        "NLP_Metrics @ file://localhost/%s" % nlp_metrics_path,
        ],
)
