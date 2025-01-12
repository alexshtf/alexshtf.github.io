---
layout: post
title:  "Your 'simple' and 'practical' ad-hoc solution may be neither simple nor practical."
tags: [pytorch,machine-learning,regression,moving-average,poisson]
description: Your exponential moving average trick is gradient descent in disguise. 
comments: true
image: /assets/polyedral_cone_layer.png
---

# Intro

Simplicity. Don't we all love simplicity? The so-called "applied science" groups in the industry love it as well. When working on advertising at Yahoo we had plenty of those 'simple and practical' solutions for various prediction and control tasks. Consider, for example, an ad campaign that produces hourly revenue . We would like to predict how much revenue would it produce in the next hour given its history. A well-known idea is the [exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing): given a factor $$\alpha \in [0, 1]$$, we compute the sequence 
$$
\begin{align*}
y_1 &= x_0\\
y_{i+1} &= (1 - \alpha) y_i + \alpha x_i 
\end{align*}
$$

Each $$y_i$$ is a weighted average of the history $$x_0, \dots, x_{i-1}$$, where most recent historical values have higher weights, and it may serve as some form of reasonable prediction for $$x_i$$. We will use the hourly ad campaign revenue as or running exammple, but the ideas, of course, apply to many other domains.

It is one of those 'simple and practical' things applied science groups do, before trying something more advanced. It's extremely easy to deploy in production: the "model" is just a number per ad campaign, which we can put in a simple file that is updated hourly. It also intuitively appears to do what we want - track the changing trends of how the campaign produces revenue, since this kind of weighted average gives more weight to more recent observations. It also appears as something we can reason about - we can compute the 'half-life' of the observations we track: observations of $$k \approx \ln(0.5) / \ln(1 - \alpha)$$ days ago have a weight less than a half.

Nice and easy! Right? Right? There are many cases when this apparently simple idea may not be that simple, and may not even be that practical. Why? Well, one of the reasons is because it is just _gradient descent_ in disguise! 

# Exponential smoothing as gradient descent

Consider the sequence of loss functions $$\ell_i(y) = \frac{1}{2}(y - x_i)^2$$. The derivative is:
$$
\ell'_i(y) = (y - x_i).
$$
Now, we can see that our exponential smoothing algorithm is nothing more than an online gradient descent method applied to the sequence of losses $$\ell_1, \ell_2, \dots$$, with a _constant_ step-size $$\alpha$$:
$$
\begin{aligned}
y_{i+1} 
  &= y_i - \alpha \ell_i'(y_i) \\
  & = y_i - \alpha (y_i - x_i) \\
  &= (1 - \alpha) y_i + \alpha x_i
\end{aligned}
$$
Whoa! Now we have several questions to ask here:

1. Is a constant step-size for our gradient descent the best strategy in this setting?
2. The loss functions aim to minimize the mean-squared error. Is this the right thing to do for money? Do we care about absolute errors, or relative (percentage) errors?
3. Don't we have other simple alternatives to gradient descent that perform better in this setting?

# Simulating ad campaigns 

So before trying different alternatives to exponential smoothing, let's first write some utility functions to simulate the revenue of ad campaigns. This is, of course, just a simulation, and we are going to make some assumptions.

Our ad campaign hourly revenue is going to be composed of two components - a periodic component, repeating every day, and a global trend. This will model the logarithm of the _mean_ revenue in each hour. So let's start from the periodic part:

```python
import numpy as np

def periodic_trend(hours):
    angles = 2 * np.pi * hours / 24 # arg for periodic trends: day = 2pi
    angles = angles.reshape(1, -1)

    # Fourier sequence of 3 frequencies - periodic trend
    coefficients = np.random.standard_t(df=4, size=(3, 1))
    phases = np.random.uniform(low=0, high=2*np.pi, size=(3, 1))
    frequencies = np.arange(1, 4).reshape(3, 1)
    return np.sum(
        coefficients * np.cos(frequencies * angles + phases), 
        axis=0
    )
```

We use the Student-T distribution for the coefficients mainly due to its heavier tails. It allows the curves to be more diverse. This is what some randomly generated periodic trends look like:

```python
import matplotlib.pyplot as plt

hours = np.arange(72)
np.random.seed(42)
plt.plot(periodic_trend(hours), label='one')
plt.plot(periodic_trend(hours), label='two')
plt.plot(periodic_trend(hours), label='three')
plt.xlabel('Hour')
plt.ylabel('Log-revenue')
plt.legend()
plt.show()
```

![periodic](/home/alex/git/alexshtf.github.io/assets/smooth_ogd/periodic.png)

Seems indeed like a bunch of random waveforms that repeat every 24 hours. Now let's generate some 'global' trends that are not trivial. For that, I found it easy to use the well-known [Bezier curves](https://en.wikipedia.org/wiki/B%C3%A9zier_curve):

```python
def global_trend(hours, total_hours=72):
    normalized_hours = hours / total_hours

    # Bezier curve with 3 coefficients - global trend
    #    see: https://en.wikipedia.org/wiki/B%C3%A9zier_curve
    global_trend_coef = np.random.standard_t(df=3, size=4)
    global_trend_basis = scipy.stats.binom.pmf(
        np.arange(4), 3, normalized_hours.reshape(-1, 1))
    return 3 * global_trend_basis @ global_trend_coef

```

Again, the coefficients come from the Student-T distribution to create interesting and diverse curves. Here are three such curves:

```python
plt.plot(global_trend(hours), label='one')
plt.plot(global_trend(hours), label='two')
plt.plot(global_trend(hours), label='three')
plt.xlabel('Hour')
plt.ylabel('Log-revenue')
plt.legend()
plt.show()
```



Moreover, there may be hours where an ad campaign does not produce revenue at all because its ads haven't received traffic. We shall assume that this happens during the more 'quiet' periods of the day more often than the more crowded parts, so the probability 
