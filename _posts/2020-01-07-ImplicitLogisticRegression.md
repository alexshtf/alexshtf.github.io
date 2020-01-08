---
layout: post
title:  "Implicit learning - warmup"
comments: true
---

# Intro
When training machine-learning models, we typically aim to minimize the training loss

$$
\frac{1}{n} \sum_{k=1}^n f_k(x), 
$$

where $$f_k$$ is the loss of the $$k^{\mathrm{th}}$$ training sample with respect to the model parameter vector $$x$$. We usually do that by variants of the stochastic gradient method: at iteration $$t$$ we select $$f \in \{ f_1, \dots, f_n \}$$, and perform the gradient step $$x_{t+1} = x_t - \eta \nabla f(x_t)$$. Many variants exist, i.e. AdaGrad and Adam, but they all share one property - they use $$f$$ as a 'black box', and assume nothing about $$f$$, except for being able to compute its gradient. In this series of posts we explore methods which can exploit more information about the losses $$f$$.

# Gradient step revisited
The gradient step is usually taught as 'take a small step in the direction of the negative gradient', but there is a different view - the well-known[^prox] _proximal view_:  
$$
x_{t+1} = \operatorname*{argmin}_{x} \left\{ H_t(x) \equiv
    \color{blue}{f(x_t) + \nabla f(x_t)^T (x - x_t)} + \frac{1}{2\eta} \color{red}{\| x - x_t\|_2^2} \tag{*}
\right\}.
$$
The blue part in the formula above is the tangent, or the first-order Taylor approximation at $$x_t$$, while the red part is a measure of proximity to $$x_t$$. In other words, the gradient step can be interpreted as
> descent along the tangent at $$x_t$$, but stay close to $$x_t$$.

How close we stay to $$x_t$$ is controlled by the step-size parameter $$\eta$$. 

To convince ourselves that (*) above is indeed the gradient step in disguise, we recall that by Fermat's principle we have $$\nabla H_t(x_{t+1}) = 0$$, or equivalently

$$
\nabla f(x_t) + \frac{1}{2\eta} (x_{t+1} - x_t) = 0.
$$

By re-arranging and extracting $$x_{t+1}$$ we recover the gradient step.

# Beyond the black box
A first order approximation is reasonable if we know nothing about the function $$f$$, except for the fact that it is differentiable. But what if we **do** know something about $$f$$? Let us consider an extreme case - we would like to exploit all the information we have, and use $$f$$ as the approximation of itself. This idea is known as the stochastic proximal point method[^ppm], or implicit learning[^impl].

Let us consider a simple example - linear regression. Our aim is to minimize 

$$
\frac{1}{2n} \sum_{k=1}^n (a_i^T x + b_i)^2
$$

Thus, every $$f$$ is of the form $$f(x)=\frac{1}{2}(a^T x + b)^2$$, where $$a$$ is a vector and $$b$$ is a real number. The corresponding optimization step becomes
$$
x_{t+1}=\operatorname*{argmin}_x \left\{
 \frac{1}{2}(a^T x + b)^2 + \frac{1}{2\eta} \|x - x_t\|^2
\right\} \tag{**}
$$
To derive an explicit formula for $$x_{t+1}$$ let's again, take the gradient w.r.t $$x$$ at $$x_{t+1}$$ and equate it with zero:
$$
a(a^T x_{t+1} + b) + \frac{1}{\eta}(x_{t+1} - x_t) = 0
$$
Now it becomes a bit technical, so bear with me - it leads to an important conclusion. Re-arranging, we obtain
$$
[\eta (a a^T) + I] x_{t+1} = x_t - (\eta b) a
$$
Solving for $$x_{t+1}$$ leads to
$$
x_{t+1} =[\eta (a a^T) + I]^{-1}[x_t - (\eta b) a].
$$
 It seems that we have defeated the whole point of using a first-order method - avoiding inverting matrices to solve least-squares problems. The remedy comes from the famous [Sherman-Morrison matrix inversion formula]([https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula](https://en.wikipedia.org/wiki/Sherman–Morrison_formula)), which leads us to
$$
x_{t+1}=\left[I - \frac{\eta a a^T}{1+\eta \|a\|_2^2} \right][x_t - (\eta b) a],
$$
which by careful mathematical manipulations can be further simplified into
$$
x_{t+1}=x_t - \frac{\eta (a^T x_t+b)}{1+\eta \|a\|_2^2} a.
$$
Ah! Finally! Now we have arrived at a formula which can be implemented in $$O(d)$$ operations, where $$d$$ is the dimension of $$x$$, just like the regualr gradient method. We just need to compute the coefficient before $$a$$, and take a step in the direction opposite to $$a$$.

Before moving forward and testing our algorithm in practice, let's look at our derivation again. It seems we employed some heavy-duty machinery - we had to find an efficient way to invert a matrix. Moreover, it seems that a different problem would require deriving an entirely new update step, which might be even more involved. Can we avoid this complexity, and construct a generic and simple method for a family of learning problems? It turns out to be possible, and we will address this challenge in a subsequent blog post.

# Show me some code

 TODO - implement code, and test it on a well-known dataset.

# References

[^ppm]: Bianchi, P. (2016). Ergodic convergence of a stochastic proximal point algorithm. _SIAM Journal on Optimization_, 26(4), 2235-2260.
[^impl]: Kulis, B., & Bartlett, P. L. (2010). Implicit online learning. _In Proceedings of the 27th International Conference on Machine Learning (ICML-10)_ (pp. 575-582).
[^prox]: Polyak B. (1987). Introduction to Optimization. _Optimization Software_

