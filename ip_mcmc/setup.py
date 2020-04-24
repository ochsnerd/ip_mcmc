from setuptools import setup

setup(
   name='ip_mcmc',
   version='0.1',
   description='Markov Chain Monte Carlo methods for Inverse Problems',
   author='David Ochsner',
   author_email='ochsnerd@student.ethz.ch',
   packages=['ip_mcmc'],
   install_requires=['numpy', 'scipy', 'matplotlib'],  # can I get this directly from requirements.txt?
)
