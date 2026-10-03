#  _Causal Inference: What If_ [Book](https://static1.squarespace.com/static/675db8b0dd37046447128f5f/t/6a86cf7ecfcb6b4ca8eea5cb/1787219838424/hernanrobins_WhatIf_19aug26.pdf) 

# Python Code for _Causal Inference: What If_

This repo contains Python code for Part II of the book _Causal Inference: What If_, by Miguel Hernán and James Robins ([book site](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)):

> Hernán MA, Robins JM (2020). _Causal Inference: What If_. Boca Raton: Chapman & Hall/CRC.

This Python version roughly corresponds to the Stata, R, or SAS programs found at the book site, and was also translated into Julia, [here](http://www.github.com/jrfiedler/causal_inference_julia_code).

The code in this repo has been checked against the 30 March 2021 version of the book.


## Python dependencies

Required Python packages:

- numpy
- pandas
- statsmodels
- scipy
- matplotlib
- linearmodels
- tqdm

If you use the [Anaconda](https://www.anaconda.com/distribution/) distribution of Python, you'll have most of those packages already, and you'll only need to install

- linearmodels
- tqdm


## Data

The data can be obtained from the [book site](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/).

The notebooks all assume that the Excel version of the data has been saved in the same directory as the notebooks.


## Author

James Fiedler, with contributions from [Petty PY Chen](https://github.com/pettypychen) and [Piyush Madan](https://github.com/piyushmadan)
