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

It is well-known that the gredient step can be written as  
$$
x_{t+1} = \operatorname{argmin}_{x} \left\{ H_t(x) \equiv
    \color{blue}{f(x_t) + \nabla f(x_t)^T (x - x_t)} + \frac{1}{2\eta} \color{red}{\| x - x_t\|_2^2} \tag{*}
\right\}.
$$  
The blue part in the formula above is the tangent, or the first-order Taylor approximation at $$x_t$$, while the red part is a measure of proximity to $$x_t$$. In other words, the gradient step can be interpreted as "descent along the tangent at $$x_t$$, but stay close to $$x_t$$". How close we stay to $$x_t$$ is controlled by the step-size parameter $$\eta$$. 

To convince ourselves that (*) above is indeed the gradient step in disguise, we recall that by Fermat's principle we have $$\nabla H_t(x_{t+1}) = 0$$, or equivalently  
$$
\nabla f(x_t) + \frac{1}{2\eta} (x_{t+1} - x_t) = 0.
$$  
By re-arranging and extracting $$x_{t+1}$$ we recover the gradient step.

A first order approximation is reasonable if we know nothing about the function $$f$$, except for the fact that it is differentiable. But what if we **do** know something about $$f$$? Let us consider an extreme case - we would like to exploit all the information we have, and use $$f$$ as the approximation of itself. This idea is known as the stochastic proximal point method[^ppm], or implicit learning[^impl].

[^ppm]: Bianchi, P. (2016). Ergodic convergence of a stochastic proximal point algorithm. _SIAM Journal on Optimization_, 26(4), 2235-2260.
[^impl]: Kulis, B., & Bartlett, P. L. (2010). Implicit online learning. _In Proceedings of the 27th International Conference on Machine Learning (ICML-10)_ (pp. 575-582).