# Update
# python3 setup.py sdist
# twine upload dist/*

from setuptools import setup, find_packages

setup(
  name="mhlb", 
  version="0.15",
  author="Austin Brown",
  author_email="austin.d.brown@warwick.ac.uk",
  description="A Python implementation to estimate lower bounds on the geometric convergence rate for RWM Metropolis-Hastings from the pre-print arxiv.org/abs/2212.05955.",
  url = "https://github.com/austindavidbrown/lower-bounds-for-Metropolis-Hastings",
  packages=["mhlb"],
  install_requires=["torch >= 1.9.1"],
  keywords=["Metropolis-Hastings", "MCMC diagnostics"]
)