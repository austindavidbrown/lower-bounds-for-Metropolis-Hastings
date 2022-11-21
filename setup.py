# Update
# python3 setup.py sdist
# twine upload dist/*

from setuptools import setup, find_packages

setup(
  name="mhlb", 
  version="0.1",
  author="Austin Brown",
  author_email="brow5079@umn.edu",
  description="Estimate Lower bounds for Metropolis-Hastings",
  url = "https://github.com/austindavidbrown/lower-bounds-for-Metropolis-Hastings",
  packages=["cmhi"],
  install_requires=["torch >= 1.9.1"],
  keywords=["Metropolis-Hastings", "MCMC"]
)