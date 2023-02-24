# Streaming Quantile Estimator

Implements ideas found in Effective Computation of Biased Quantiles over Data
Streams (Cormode, Korn, Muthukrishnan, Srivastava) to provide a space and time
efficient estimator for streaming quantile estimation.

[![Build Status](https://travis-ci.org/streadway/quantile.svg?branch=master)](https://travis-ci.org/streadway/quantile)


## Improved API

This fork improves the currecntly existing API in allowing developers to customize **buffer** and **pool** size.

Original developers kept ratio between the two at 1/2. 
