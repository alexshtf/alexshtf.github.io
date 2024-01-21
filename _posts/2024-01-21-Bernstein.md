---
≠layout: post
title:  “Are polynomial features the root of all evil?"
tags: [machine-learning, feature-engineering, polynomials]
description: There is a well-known myth in the machine learning community - high degree polynomials are bad for modeling. In this post we
comments: true
---

# A myth

When fitting a non-linear model using linear regression, we typically generate new features using non-linear functions. We also know that any function, in theory, can be approximated by a sufficiently high degree polynomial. This result is known as [Weierstrass approximation theorem](https://en.wikipedia.org/wiki/Stone%E2%80%93Weierstrass_theorem). But many blogs, papers, and even books tell us that high polynomials should be avoided. They tend to oscilate and overfit! They even scare us with images, such as the one below, when the polynomial fit using the data points (in red) is far away from the true function (in blue):
![Polynomial overfitting]({{ "/assets/poly_overfit.png" | absolute_url }})

It turns out that it's just a MYTH. There's nothing inherently wrong with high degree polynomials, and the source of the myth stems mainly from two misconceptions about polynomials that we will explore here. In fact, not only they are great non-linear features, certain representations also provide us with powerful control over the shape of the function we wish to learn.

A colab notebook with the code for reproducing the above results is available [here](https://drive.google.com/file/d/10IxrJUKPfrYkUraC4WVncFozcqi0O5wL/view?usp=sharing).

# Approximation vs estimation

Vladimir Vapnik, in his famous book "The Nature of Statistical Learning Theory" which is cited more than 100,000 times as of today, coined the approximation vs. estimation balance. The approximation power of a model is its ability to represent the "reality" we would like to learn. Typically, approximation power increases with the complexity of the model - more parameters mean more power to represent any function to arbitrary precision. Polynomials are no different - higher degree polynomials can represent functions to higher accuracy. However, more parameters make it difficult to _estimate these parameters from the data_.

Indeed, higher degree polynomials have a higher capacity to approximate arbitrary functions. And since they have more coefficients, these coefficients are harder to estimate from data. But how does it differ from other non-linear features, such as the well-known [radial basis functions](https://en.wikipedia.org/wiki/Radial_basis_function)? Why do polynomials have such a bad reputation? Are they truly hard to estimate from data?

It turns out that the primary source is the standard polynomial basis for n-degree polynomials $$\mathbb{E}_n = {1, x, x^2, ..., x^n}$$. Indeed, any degree $$n$$  polynomial can be written as a linear combination of these functions:


$$
\alpha_0 \cdot 1 + \alpha_1 \cdot x + \alpha_2 \cdot x^2 + \cdots + \alpha_n x^n
$$


But the standard basis $$\mathbb{B}_n$$ is _awful_ for estimating polynomials from data. In this post we will explore other ways to represent polynomials that are appropriate for machine learning, and are readily available in standard Python packages. We note, that one advantage of polynomials over other non-linear feature bases is that the only hyperparameter is their _degree_. There is no "kernel width", like in radial basis functions[^1].

The second source of their bad reputation is misunderstanding of Weierstrass' approximation theorem. It's usually cited as "polynomials can approximate arbitrary continuous functions". But that's not entrely true. They can approximate arbitrary continuous functions **in an interval**. This means that when using polynomial features, the data must be normalized to lie in an interval. It can be done using min-max scaling, computing empirical quantiles, or passing the feature through a sigmoid. But we should avoid the use of polynomials on raw un-normalized features.

# Building the basics

In this post we will demonstrate fitting the function


$$
f(x)=\sin(8 \pi x) / \exp(x)+x
$$


on the interval $$[0, 1]$$ by fitting to $$m=30$$ samples corrupted by Gaussian noise. The following code implements the function and generates samples:

```python
import numpy as np

def true_func(x):
  return np.sin(8 * np.pi * x) / np.exp(x) + x

m = 30
sigma = 0.1

# generate features
np.random.seed(42)
X = np.random.rand(m)
y = true_func(X) + sigma * np.random.randn(m)
```

For function plotting, we will use uniformly-spaced points in $$[0, 1]$$. The following code plots the true function and the sample points:

```python
import matplotlib.pyplot as plt

plt_xs = np.linspace(0, 1, 1000)
plt.scatter(X.ravel(), y.ravel())
plt.plot(plt_xs, true_func(plt_xs), 'blue')
plt.show()
```

![polyfit_func]({{ "/assets/polyfit_func.png" | absolute_url }})

Now let's fit a polynomial to the sampled points using the standard basis. Namely, we're given the set of noisy points $$\{ (x_i, y_i) \}_{i=1}^m$$, and we need to find the coefficients $$\alpha_0, \dots, \alpha_n$$ that minimize:

$$
\sum_{i=1}^m (\alpha_0 + \alpha_1 x_i + \dots + \alpha_n x_i^n - y_i)^2
$$

As expected, this is readily accomplished by transforming each sample $$x_i$$ to a vector of features $$1, x_i, \dots, x_i^n$$, and fitting a linear regression model to the resulting features. Fortunately, NumPy has the `numpy.polynomial.polynomial.polyvander`function. It takes a vector containing $$x_1, \dots, x_m$$ and produces the matrix

$$
\begin{pmatrix}
1 & x_1 & x_1^2 & \dots & x_1^n \\
1 & x_2 & x_2^2 & \dots & x_2^n \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_m & x_m^2 & \dots & x_m^n \\
\end{pmatrix}
$$

The name of the function comes from the name of the matrix - the Vandermonde matrix. Let's use it to fit a polynomial of degree $$n=50$$.

```python
from sklearn.linear_model import LinearRegression
import numpy.polynomial.polynomial as poly

n = 50
model = LinearRegression(fit_intercept=False)
model.fit(poly.polyvander(X, deg=n), y)
```

The reason we use `fit_intercept=False` is because the 'intercept' is provided by the first column of the Vandermonde matrix. Now we can plot the function we just fit:

```python
plt.scatter(X.ravel(), y.ravel())                                    # plot the samples
plt.plot(plt_xs, true_func(plt_xs), 'blue')                          # plot the true function
plt.plot(plt_xs, model.predict(poly.polyvander(plt_xs, deg=n)), 'r') # plot the fit model
plt.ylim([-5, 5])
plt.show()
```

As expected, we got the "scary" image from the beginning of this post. Indeed, the standard basis is awful for model fitting! We hope that regularization provides a remedy, but it does not. Maybe adding some L2 regularization helps? Let's use the `Ridge` class from the `sklearn.linear_model`  package to fit an L2 regularized model:

```python
from sklearn.linear_model import Ridge

reg_coef = 1e-7
model = Ridge(fit_intercept=False, alpha=reg_coef)
model.fit(poly.polyvander(X, deg=n), y)

plt.scatter(X.ravel(), y.ravel())                                    # plot the samples
plt.plot(plt_xs, true_func(plt_xs), 'blue')                          # plot the true function
plt.plot(plt_xs, model.predict(poly.polyvander(plt_xs, deg=n)), 'r') # plot the fit model
plt.ylim([-5, 5])
plt.show()
```

We get the following result:

![polyfit_standard_ridge]({{ "/assets/polyfit_standard_ridge.png" | absolute_url }})

The regularization coefficient coefficient of $$\alpha=10^{-7}$$ is large enough to break the model in $$[0,0.8]$$ but not large enough to avoid over-fitting in $$[0.8, 1]$$. Increasing the coefficient clearly won't help - the model will be broken even further in $$[0, 0.8]$$.

Since we will be trying several polynomial bases, it makes sense to write a more generic function for our experiments that will accept various "Vandermonde" matrix functions of the basis of our choice, fit the polynomial using the `Ridge` class, and plot it with the original function and the sample points.

```python
def fit_and_plot(vander, n, alpha):
  model = Ridge(fit_intercept=False, alpha=alpha)
  model.fit(vander(X, deg=n), y)

  plt.scatter(X.ravel(), y.ravel())                           # plot the samples
  plt.plot(plt_xs, true_func(plt_xs), 'blue')                 # plot the true function
  plt.plot(plt_xs, model.predict(vander(plt_xs, deg=n)), 'r') # plot the fit model
  plt.ylim([-5, 5])
  plt.show()  
```

Now we can reproduce our latest experiment by invoking:

```python
fit_and_plot(poly.polyvander, n=50, alpha=1e-7)
```



# Polynomial bases

It turns out that in our sister discipline, approximation theory, reseachers also encountered similar difficulties with the standard basis $$\mathbb{E}_n$$, and developed a thoery for approximating functions by polynomials from different bases. Two prominent examples of bases of $$n$$-degree polynomials include, and their:

1. The [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) $$\mathbb{T}_n = \{ T_0, T_1, \dots, T_n \}$$, implemented in the `numpy.polynomial.chebyshev` module.
2. The [Legendre polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials) $$\mathbb{P}_n = \{ P_0, P_1, \dots, P_n \}$$, implemented in the `numpy.polynomial.legendre` module.

They are the computational workhorse of a large variety of numerical algorithms that are enabled by approximating a function using a polynomial, and are well-known for their advantages in approximating functions in the $$[-1, 1]$$ interval[^3]. In particular, the corresponding "Vandermonde" matrices are provided by the `chebvander` and `legvander` functions in corresponding modules above.

I will not elaborate their formulas and properties here for a reason that will immediately be revealed. However, I highly recomment Prof. Nick Trefethen's "Approximation theory and approximation practice" [online video course](https://people.maths.ox.ac.uk/trefethen/atapvideos.html) to get familiar with their advantages. His book with the same name is an excellent introduction to the subject. 

It might be tempting to try fitting a Chebyshev polynomial using our `fit_and_plot` method above directly:

```python
import numpy.polynomial.chebyshev as cheb

fit_and_plot(cheb.chebvander, n=50, alpha=1e-7)
```

However, that's not the best thing to do. We aim to fit a function sampled from $$[0, 1]$$, but the Chebyshev basis "lives" in $$[-1, 1]$$. Therefore, we will add the transformation $$x \to 2x-1$$ before invoking the `chebvander` function:

```python
def scaled_chebvander(x, deg):
  return cheb.chebvander(2 * x - 1, deg=deg)

fit_and_plot(scaled_chebvander, n=50, alpha=1)
```

Note that a different basis requires a different regularization coefficient. We get the following result:

![polyfit_cheb_reg1]({{ "/assets/polyfit_cheb_reg1.png" | absolute_url }})

Whoa! Seems even worse than the standard basis!. Maybe more regularization helps?

```python
fit_and_plot(scaled_chebvander, n=50, alpha=10)
```

![polyfit_cheb_reg10]({{ "/assets/polyfit_cheb_reg10.png" | absolute_url }})

Appears that our polynomial is both a bad fit for the function, and extremely oscilatory. Even worse when the standard basis! Interested readers can repeat the experiment with Legendre polynomials and see a slightly better, but similar result. So what's wrong? Is everything that approximation theory tries to teach us about polynomials wrong?

The answer stems from the fundamental difference between two tasks: 

- **Interpolation** - finding a polynomial that agrees with the approximated function $$f(x)$$ _exactly_ at a set of _carefully chosen_ points
- **Fitting** - finding a polynomial that agrees _approximately_ with a given _noisy_ set of points, which are _out of our control_.

The Chebyshev and Legendre bases are very well suited for the interpolation task, but not for the fitting task. It turns out that the polynomial $$T_k$$ in the Chebyshev basis, and the polynomial $$P_k$$ in the Legendre basis, are both $$k$$-degree polynomials. Thus, the coefficient of $$T_1$$ and $$T_{50}$$ have "different units". This property is shared with the standard basis as well. Thus, we have two issues:

1. A small change of the coefficient of a high degree basis function, say the coefficient $$\alpha_{50}$$, has a huge effect on the shape of the polynomial. A small change can drastically change the shape. A small change in the input data has a *huge* effect of the fit model. It can be either a small change in the feature $$x_i$$, or a change in the noised sample $$y_i = f(x_i) + \varepsilon$$.
2. L2 regularization makes no sense! For reasonable functions, the coefficient $$\alpha_{50}$$ should be much smaller than the coefficient $$\alpha_1$$. This is regardless of the choice of the basis!

Both properties show that for the fitting, rather the interpolation tasks we need something else.

# The Bernstein basis

A remedy is provided by the Bernstein basis $$\mathbb{B}_n = \{  B_{0,n}, \dots, B_{n, n} \}$$. These are $$n$$-degree polynomials defined by on $$[0, 1]$$ by:

$$
B_{i,n}(x) = \binom{n}{i} x^i (1-x)^{n-i}
$$

These polynomials are widely used in computer graphics to approximate curves and surfaces, but it appears that they're less known in the machine learning community. In fact, all the text you see on the screen when reading this post is rendered using Bernstein polynomials[^2].

First, note that each $$B_{i,n}$$ is an $$n$$-degree polynomial. Thus, when representing a polynomial using

$$
p_n(x) = \alpha_0 B_{0,n}(x) + \alpha_1 B_{1,n}(x) + \dots + \alpha_n B_{n,n}(x),
$$

all the coefficients have the same "units". 

Moreover, note that $$B_{i,n}(x)$$ is exactly the probability mass function of the binomial distribution for obtaining $$i$$ successes in a sequence of trials whose success probability is $$x$$. Therefore, $$p_n(x) \geq 0$$,  and $$\sum_{i=0}^n p_i(x) = 1$$ for any $$x \in [0, 1]$$. This means that $$p_n(x)$$ is just a weighted average of the coefficients $$\alpha_0, \dots, \alpha_n$$. So not only the coefficients have the same "units", their units are also the same as the targets $$y_i \approx p_n(x_i)$$.

Finally, due to the equivalence with the binomial distribution p.m.f, we can implement a "Vandermonde" matrix in Python using the `scipy.stats.binom.pmf` function.

``` python
from scipy.stats import binom

def bernvander(x, deg):
	return binom.pmf(np.arange(1 + deg), deg, x.reshape(-1, 1))
```

Let's try and fit without regularization at all

```python
fit_and_plot(bernvander, n=50, alpha=0)
```

![polyfit_bern_reg0]({{ "/assets/polyfit_bern_reg0.png" | absolute_url }})

We see our regular over-fitting. After playing a bit with the regularization coefficient, we obtain the following

```python
fit_and_plot(bernvander, n=50, alpha=5e-7)
```

![polyfit_bern_reg5em4]({{ "/assets/polyfit_bern_reg5em4.png" | absolute_url }})

Beautiful! This is a polynomial of degree 50! The fit is great, no oscillations, and the misfit near the right endpoint stems from the noise - I don't believe there's enough information in the data to convey the fact that it should "curve up" rather than "curve down".

Let's see what happens when we crank-up the degree. Can we produce a nice non-oscilating polynomial?

```python
fit_and_plot(bernvander, n=100, alpha=5e-4)
```

![polyfit_bern_100_reg5em4]({{ "/assets/polyfit_bern_100_reg5em4.png" | absolute_url }})

This is a polynomial of degree 100, that does not overfit!

# Summary

The notorious reputation of high-degree polynomials in the machine learning community is primarily a myth. Despite it, papers, books, and blog posts are based on this premise as an axiom. Bernstein polynomials are little known in the machine learning community, but there are a few papers using them to represent polynomial features. See the paper [^4] for example.  

In the following posts we will explore the Bernstein basis in more detail. We will use it to create polynomial features for real-world datasets and test it versus the standard basis, and see how we can control the shape of the curve by controlling the coefficients. For example, what if we know that the function we're aiming to fit is increasing? Stay tuned!

[^1]: There are also kernel methods, and polynomial kernels. But polynomial kernels suffer from problems similar to the standard basis.
[^2]: See [Bézier curves](https://en.wikipedia.org/wiki/B%C3%A9zier_curve) and TrueType [font outlines](https://en.wikipedia.org/wiki/TrueType#Outlines).
[^3]: The standard basis is not that awful. It's a great basis for representing polynomials on the complex unit circle. In fact, the Fourier transform is based exactly on this observation.
[^4]: Marco, Ana, and José-Javier Martı. "Polynomial least squares fitting in the Bernstein basis." Linear Algebra and its Applications 433.7 (2010): 1254-1264.
