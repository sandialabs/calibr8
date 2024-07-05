CALIBR8
========

<img src="doc/xspec.png" alt="xspec" width="200"/>

## What is This?

CALIBR8 is a state-of-the-art material model calibration code that is
capable of executing on high performance machines in a massively parallel
context. CALIBR8 can perform material model calibration using either
adjoint-based or forward mode sensitivities using automatic differentiation
(AD).

## Getting Started

The [documentation](https://sandialabs.github.io/calibr8/) is the best
place to begin learning about CALIBR8, how to install it, its capabilities,
and how to use it.

## Automatic Differentiation

The use of AD to compute the required gradients for
material model calibration is discussed in depth in
[this](https://arxiv.org/abs/2010.03649) article. The code makes use of
[Sacado](https://github.com/trilinos/Trilinos/tree/master/packages/sacado)
for the purposes of AD.

If you've found CALIBR8 useful in your research, please cite the paper

```tex
@article{seidl2022calibration,
  title={Calibration of elastoplastic constitutive model parameters from full-field data with automatic differentiation-based sensitivities},
  author={Seidl, D Thomas and Granzow, Brian N},
  journal={International Journal for Numerical Methods in Engineering},
  volume={123},
  number={1},
  pages={69--100},
  year={2022},
  publisher={Wiley Online Library}
}

```

##

At Sandia, Calibr8 is SCR# 2690.0
