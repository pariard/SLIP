# SLIP: Learning to predict in unknown dynamical systems with long-term memory

This repository includes an implementation of the spectral LDS improper predictor (SLIP) from our paper

[Paria Rashidinejad, Jiantao Jiao, and Stuart Russell. "SLIP: Learning to Predict in Unknown Dynamical Systems with Long-Term Memory."](https://arxiv.org/pdf/2010.05899.pdf)  NeurIPS 2020 (oral). 



## Prerequisites
+ Python (>= 3.5)
+ NumPy (>= 1.6)
+ SciPy (>= 0.17.0)
+ Matplotlib (>= 1.5.1)
+ Scikit-Learn 

## How to run the code
To run a simulation, in the LDS settings cell specify
+ the LDS parameters or let the function `generate_random_LDS` sample them randomly based on dimensions;
+ the time horizon;
+ the number of iterations;
+ the number of filters (based on horizon) and regularization parameter.

``` 
python SLIP.py
```

## Cite
```
@article{rashidinejad2020slip,
  title={SLIP: Learning to Predict in Unknown Dynamical Systems with Long-Term Memory},
  author={Rashidinejad, Paria and Jiao, Jiantao and Russell, Stuart},
  journal={arXiv preprint arXiv:2010.05899},
  year={2020}
}
```
