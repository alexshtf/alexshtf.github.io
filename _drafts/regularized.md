---
≠layout: post
title:  “Proximal Point - regularized convex on linear II"
tags: [machine-learning, optimization, proximal-point, online-optimization, online-learning, logistic-regression, regularization]
comments: true
---

Last time we discussed applying the stochastic proximal point method to losses of the form:



$$
f(x)=\phi(a^T x + b) + r(x),
$$



where $$\phi$$ and $$r$$ are convex functions, and devised an efficient implementation for L2 regularized losses: $$r(x) = (\lambda/2) \|x\|_2^2$$.  We now aim for a more general approach, which will allow us to deal with many other regularizers.

Recall that implementing the method amounts to solving the problem dual to the optimizer’s step:


$$
q(s)=\color{blue}{\inf_x \left \{ \mathcal{Q}(x) = r(x)+\frac{1}{2\eta} \|x-x_t\|_2^2 + s a^T x \right \}} - \phi^*(s)+sb.
$$



The part highlighted in blue is a major barrier between a practitioner and the method - the practitioner has to mathematically re-derive the minimization of $$\mathcal{Q}(x)$$ and re-implement the results for every regularizer.  Seems quite ‘impractical’. Furthermore, since $$r(x)$$ can be arbitrarily complex, it might even be impossible. 

Unfortunately, we cannot remove this obstacle entirely, but we can express it in terms of a “textbook concept” - something a practitioner can pick from a catalog in a textbook and use. In fact, that’s how all optimization methods work. For example, SGD is based on the textbook concept of ‘gradient’,  the stochastic proximal point method in the last post is based on the textbook concept of ‘convex conjugate’, and in this post we will introduce and use yet another textbook concept.

# High-school tricks

One trick which is taught in high-school algebra classes is known as “completing a square” - we re-arrange the formula for a square of a sum $$(a+b)^2=a^2+2ab+b^2$$  to the form


$$
a^2+2ab=(a+b)^2-b^2.
$$


Such a trick is occasionally useful to express things in terms of squares only. We do a similar trick on the formula



$$
\frac{1}{2}\|a + b\|_2^2= \frac{1}{2}\|a\|_2^2+a^T b+\frac{1}{2}\|b\|_2^2. \tag{a}
$$



Re-arranging, we obtain



$$
\frac{1}{2}\|a\|_2^2+a^T b = \frac{1}{2}\|a + b\|_2^2 - \frac{1}{2}\|b\|_2^2. \tag{b}
$$



Now, let’s apply the trick to the term $$\mathcal{Q}(x)$$ inside the infimum. It is a bit technical, but the end-result leads us to our desired texbook recipe.



$$
\begin{align}
\mathcal{Q}(x) &= r(x)+\frac{1}{2\eta} \|x-x_t\|_2^2 + s a^T x \\
 &= \frac{1}{\eta} \left[ \eta r(x) +  \frac{1}{2} \|x - x_t\|_2^2 + \eta s a^T x \right] & \leftarrow \text{Factoring out  } \frac{1}{\eta} \\
 &= \frac{1}{\eta} \left[ \eta r(x) + \color{orange}{\frac{1}{2} \|x\|_2^2 - (x_t - \eta s a )^T x} + \frac{1}{2} \|x_t\|_2^2 \right] & \leftarrow \text{opening } \frac{1}{2}\|x-x_t\|_2^2 \text{with (a)} \\
 &= \frac{1}{\eta} \left[ \eta  r(x) + \color{orange}{\frac{1}{2} \|x - (x_t - \eta s a)\|_2^2 - \frac{1}{2} \|x_t - \eta s a\|_2^2 } + \frac{1}{2} \|x_t\|_2^2 \right] & \leftarrow \text{square completion with (b)}\\
 &= \left[ r(x)+\frac{1}{2\eta} \|x - (x_t - \eta s a)\|_2^2  \right] - \frac{1}{2\eta} \|x_t - \eta s a\|_2^2 + \frac{1}{2\eta} \|x_t\|_2^2  &\leftarrow{\text{applying (a) and canceling}}\\
 &= \left[ r(x)+\frac{1}{2\eta} \|x - (x_t - \eta s a)\|_2^2  \right] + (a^T x_t) s - \frac{\eta \|a\|_2^2}{2} s^2
\end{align}
$$



With the above in mind, the dual problem aims to maximize:



$$
q(s)= \color{magenta}{ \inf_x \left \{ r(x)+\frac{1}{2\eta} \| x - (x_t - \eta s a)\|_2^2 \right \}} + (a^T x_t + b) s - \frac{\eta \|a\|_2^2}{2} s^2  - \phi^*(s)
$$



It may seem unfamiliar, but the magenta term is a well-known concept in optimization: the Moreau envelope[^menv] of the function $$r(x)$$. Let’s get introduced to the concept properly.

Formally, the  Moreau envelope of a convex function $$r$$ with parameter $$\eta$$ is denoted by $$M_\eta r$$ and defined by



$$
M_\eta r(u) = \inf_x \left\{ r(x) + \frac{1}{2\eta} \|x - u\|_2^2 \right\}. \tag{c}
$$



Consequently, we can write  the function $$q(s)$$ of the dual problem as:



$$
q(s) = \color{magenta}{M_{\eta} r (x_t - \eta s a)} + (a^T x_t + b) s - \frac{\eta \|a\|_2^2}{2} s^2 - \phi^*(s).
$$



Now $$q(s)$$ is composed of two textbook concepts - the convex conjugate of $$\phi$$, and the Moreau envelope of the regularizer $$r$$. A related concept is the minimizer of (c) above - Moreau’s [proximal operator](https://en.wikipedia.org/wiki/Proximal_operator) of $$r$$ with parameter $$\eta$$:



$$
\operatorname{prox}_{\eta r}(u) = \operatorname*{argmin}_x \left\{ r(x)+\frac{1}{2\eta} \|x - u\|_2^2 \right\}.
$$



Moreau envelopes and proximal operators, which were introduced in 1965 by the French mathematician [Jean Jacques Moreau](https://en.wikipedia.org/wiki/Jean-Jacques_Moreau), are used to create a  “smoothed” version of arbitrary convex functions. Since then, the concepts have been used for many other purposes - just [Google Scholar](https://scholar.google.com/scholar?q=Moreau+Envelope) them :)

Before going deeper, let’s explicitly write our “meta-algorithm" for computing $$x_{t+1}$$ using the above concepts:

1. Compute  $$q(s)$$.
2. Solve the dual problem: find a maximizer $$s^*$$ of $$q(s)$$.
3. Compute $$x_{t+1} = \operatorname{prox}_{\eta r}(x_t - \eta s^* a)$$.



# Moreau envelope - tangible example

To make things less abstract, look at a one-dimensional example to gain some more intuition:  the absolute value function $$r(x) = |x|$$. Doing some calculus which is out of scope of this post, we can compute:



$$
M_{\eta}r (u) = \inf_x \left\{ |x| + \frac{1}{2\eta}(x - u)^2\right\} = \begin{cases}
\frac{u^2}{2\eta} & |u| \leq \eta \\
|u| - \frac{\eta}{2} & |u|>\eta
\end{cases}
$$



That is, the envelope is a function which looks like a prabola when $$u$$ is close enough to the 0, and ’switches’ to behaving like the absolute value when we get far away. Some readers may recognize this function - this is the well-known Huber function, which is commonly used in statistics as a loss function. Indeed, it is used as a smooth approximation of the absolute value. Let’s plot it:

```python
import numpy as np
import matplotlib.pyplot as plt
import math

def huber_1d(eta, u):
    if math.fabs(u) <= eta:
        return (u ** 2) / (2 * eta)
    else:
        return math.fabs(u) - eta / 2
    
def huber(eta, u):
    return np.array([huber_1d(eta, x) for x in u])

x = np.linspace(-2, 2, 1000)
plt.plot(x, np.abs(x), label='|x|')
plt.plot(x, huber(1, x), label='eta=1')
plt.plot(x, huber(0.5, x), label='eta=0.5')
plt.plot(x, huber(0.1, x), label='eta=0.1')
plt.legend(loc='best')
plt.show()
```

Here is the resulting plot:

![huber](../assets/huber.png)

Viola! A smoothed version of the absolute value function. Smaller values of $$\eta$$ lead to a better, but less smooth approximation. As we said before, this behavior is not unique to the absolute value’s envelope - Moreau envelopes of convex functions are _always_ smooth and differentiable. 

Another interesting thing we can see in the plot is that the envelopes approach the approximating function from below. It is not a coincidence as well, since:


$$
M_\eta r(u) = \inf_x \left\{ r(x) + \frac{1}{2\eta} \|x - u\|_2^2 \right\} \underbrace{\leq}_{\text{taking }x=u} r(u)+\frac{1}{2\eta}\|u-u\|_2^2=r(u),
$$


that is, the envelope always lies below the function itself.

# Maximizing $$q(s)$$

Now let’s get back to our meta-algorithm. We need to solve the dual problem by maximizing $$q(s)$$, and we typically do it by equating its derivative $$q’(s)$$ with zero. Hence, in practice, we are interested in the _derivative_ of $$q$$ rather than its value (assuming $$q$$ is indeed differentiable). Using the chain rule, we obtain:



$$
q'(s) = -\eta a^T ~\nabla M_\eta r(x_t - \eta s a) + (a^T x_t + b) - \eta \|a\|_2^2 s - {\phi^*}'(s). \tag{c}
$$



Well, if only we had some magic formula for computing the **gradient** of $$M_\eta r$$.  Moreau’s exceptional work does provide a formula:



$$
\nabla M_\eta r(u) = \frac{1}{\eta} \left(u - \operatorname{prox}_{\eta r}(u) \right).
$$



It requires some advanced math to derive, so its proof out of scope of this blog post. Substituting the formula for into (c), the derivative $$q’(s)$$ can be written as


$$
\begin{align}
q'(s)
 &=-a^T(x_t - \eta s a - \operatorname{prox}_{\eta r}(x_t - \eta s a)) + (a^T x_t+b) - \eta \|a\|_2^2 s -{\phi^*}'(s) \\
 &= a^T \operatorname{prox}_{\eta r}(x_t - \eta s a) - {\phi^*}'(s) + b
\end{align}
$$


To conclude, our ingredients for $$q’(s)$$ are: a formula for the proximal operator of $$r$$, and a formula for the derivative of $$\phi^*$$. Like convex conjuages, proximal operators are ubiquitous in optimization theory and practice. Consequently, entire book chapters about proximal operators were written, i.e. see [here](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)[^proxalgs] and [here](https://archive.siam.org/books/mo25/mo25_ch6.pdf)[^fom6]. The second reference contains, at the end of the chapter, a catalog of explicit formulas for $$\operatorname{prox}_{\eta r}$$ for various functions $$r$$  summarized in a table. Here are a two important examples:

| $$r(x)$$                                        | $$\operatorname{prox}_{\eta r}(u)$$                          | Remarks                                                |
| ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| $$(\lambda/2) \|x\|_2^2$$                       | $$\frac{1}{1+\eta \lambda} u$$                               |                                                        |
| $$\lambda \|x\|_1 = \lambda\sum_{i=1}^n |x_i|$$ | $$[|u|-\lambda \eta \mathbf{1}] \cdot \operatorname{sign}(u)$$ | $$\mathbf{1}$$ is a vector whose components are all 1. |
| 0                                               | u                                                            | no regularizer                                         |

With the above in mind, the meta-algorithm for computing $$x_{t+1}$$ amounts to:

1. Obtain a solution $$s^*$$ of the equation $$q’(s)=a^T \operatorname{prox}_{\eta r}(x_t - \eta s a) - {\phi^*}'(s) + b = 0$$
2. Compute $$x_{t+1} = \operatorname{prox}_{\eta r}(x_t - \eta s^* a)$$

# L2 regularization - again

Last time we ‘manually’ derived  the computational steps for L2 regularized least squares, namely, losses of the form:


$$
f(x)=\phi(a^T x + b)+\underbrace{\frac{\lambda}{2} \|x\|_2^2}_{r(x)}.
$$


Let’s see what we obtain given what we learned in this post. According to the table of proximal operators, we have $$\operatorname{prox}_{\eta r}(u)=\frac{1}{1+\eta \lambda} u$$.  Thus, to compute $$s^*$$ we need to solve:


$$
\begin{align}
q'(s)
 &=\frac{1}{1+\eta\lambda}a^T(x_t - \eta s a) - {\phi^*}'(s) + b \\
 &= \frac{a^T x_t}{1+\eta\lambda} + b -\frac{\eta \|a\|_2^2}{1+\eta\lambda} s -  {\phi^*}'(s) = 0
\end{align} 
$$


Looking carefully, we see that it is exactly the derivative of $$q(s)$$ from the last post, but this time it was obtained by taking a formula from a textbook, instead of manually computing $$q(s)$$.

Having obtained the solution of  $$s^*$$ of the equation $$q’(s)=0$$, we can proceed and compute


$$
x_{t+1}=\frac{1}{1+\eta\lambda}(x_t - \eta s^* a),
$$


which is, again, the same formula we obtained in the last post, but without manually deriving anything.

The only thing a practitioner wishing to derive a formula for $$x_{t+1}$$ has to do ‘manually’ is to find a way to solve the one-dimensional equation $$q’(s)=0$$. The rest is provided by our textbook concepts - the proximal operator, and the convex conjugate.

# End to end implementation - L1 regularized logistic regression

TODO

# Teaser



[^menv]: J-J Moreau. (1965). Proximite et dualit´e dans un espace Hilbertien. _Bulletin de la Société Mathématique de France 93._  (pp. 273–299)

[^proxalgs]: N. Parikh, S. Boyd (2013). Proximal Algorithms. _Foundations and Trends in Optimization_ Vol.1 No. 3 (pp. 123-231)
[^fom6]: A. Beck (2017). First-Order Methods in Optimization. _SIAM Books_ Ch.6 (pp. 129-177)