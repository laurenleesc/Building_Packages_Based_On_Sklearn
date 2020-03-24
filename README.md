# Building Packages Based On Sklearn

This package was originally built to deploy a bias correction in the coefficients (betas) for high dimensional logistic regression, based on the following paper: https://www.pnas.org/content/116/29/14516.short. So far, we have not been able to figure out the math, and so currently this project has an example "dummy" bias correction factor: if the beta estimate is negative, it adds a 1, if it is positive, it subtracts a 1.  This is a place-holder for real bias correction factors.

This is a work in progress. Use at your own risk. Distributed under the 3-Clause BSD license.

