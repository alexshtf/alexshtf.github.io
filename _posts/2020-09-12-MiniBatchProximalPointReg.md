---
≠layout: post
title: “Proximal Point with Mini Batches - Regularized Convex On Linear"
tags: [machine-learning, optimization, proximal-point, stochastic-optimization, theory]
description: The mini-batch version of the proximal point method for regularized convex-on-linear losses requires solving a low-dimensional convex dual problem. We explore how the regularizer fits in the dual problem, so that our computational complexity will be linear in the dimension of the model.
comments: true
series: "Proximal point"
image: /assets/proxpt_reg_logreg_minibatch.png

---

# Recap and intro

We continue our endaevor of extending the reach of efficiently implementable stochastic proximal point method in the mini-batch setting:

$$
x_{t+1} = \operatorname*{argmin}_x \Biggl \{ \frac{1}{|B|} \sum_{i \in B} f_i(x) + \frac{1}{2\eta} \|x - x_t\|_2^2 \Biggr \}.
$$

[Last time]({{ page.previous.url }}) we discussed the implementation for convex-on-linear losses, which include linear least squares and linear logistic regression. Continuing the same journey we already went through before the mini-batch setting, this time we add regularization, and consider losses of the form:

$$
f_i(x)=\phi(a_i^T x + b_i) + r(x),
$$

where the regularizer $$r$$ is the same for all training samples, and $$\phi$$ is a scalar convex function. In that case, the method becomes:

$$
x_{t+1} = \operatorname*{argmin}_x \Biggl \{ \frac{1}{|B|} \sum_{i \in B} \phi(a_i^T x + b_i) + r(x) + \frac{1}{2\eta} \|x - x_t\|_2^2 \Biggr \}.
$$

Our aim in this post is to derive an efficient implementation, in Python, of the above computational step.

# Solving via duality

We again employ duality in an attempt to make the problem of computing $$x_{t+1}$$ tractable. We  replace that problem with its equivalent constrained variant:

$$
\operatorname*{minimize}_{x,z} \quad \frac{1}{|B|} \sum_{i\in B} \phi(z_i) + r(x) + \frac{1}{2\eta}\|x - x_t\|_2^2 \quad \operatorname{\text{subject to}} \quad z_i = a_i^T x + b_i.
$$

We will not get into the tedious details, but embedding the vectors $$a_i$$ into the rows of the batch matrix $$A_B$$, and after some mathematical manipulations, $$q(s)$$ is:

$$
q(s)=\color{magenta}{-\frac{1}{|B|}\sum_{i \in B} \phi^*(|B| s_i) + \underbrace{\min_x \left\{ r(x) + \frac{1}{2\eta} \|x - (x_t - \eta A_B^T s)\|_2^2\right\}}_{\text{Moreau envelope}}} + (A_B x_t + b_B)^T s - \frac{\eta}{2} \|A_B^Ts\|_2^2.
$$

And here we arrive at our problem, which appears in the magenta-colored part. The function $$-\phi^*$$ is always _concave_, while the Moreau envelope is always _convex_. The sum of such functions may, in general, be nor convex nor concave. So, although _we know_ from duality theory that the function $$q(s)$$ is concave, by separating it into two components we cannot convey its concavity to a generic convex optimization solver such as CVX - these solvers require that we write functions in a way which obeys an explicit set of rules which convey the function’s curvature as either convex or concave. One such rule is - we can add convex functions together, and concave functions together. But we cannot mix both, for reasons which go beyond the scope of this blog post.

The conclusion is the same duality trick which served us well before in transforming a high-dimensional problem on $$x$$ to a low-dimensional problem on the dual variables $$s$$, cannot serve us now. Can we do something else? It turns out we can, but first let’s explore another extension of duality - inequality constraints.

# Solving via duality - take 2

We aim to compute 

$$
x_{t+1} = \operatorname*{argmin}_x \Biggl \{  \frac{1}{|B|} \sum_{i \in B} \phi(a_i^T x + b_i) + r(x) + \frac{1}{2\eta} \|x - x_t\|_2^2 \Biggr \}.
$$

Let's rephraze the optimization problem in a slightly different manner, by including _two_ auxiliary variables - the vectors $$z$$ and $$w$$, and by embedding the vectors $$a_i$$ into the rows of the batch matrix $$A_B$$:

$$
\operatorname*{minimize}_{x,z,w} \quad \frac{1}{|B|} \sum_{i\in B} \phi(z_i) + r(w) + \frac{1}{2\eta} \|x - x_t\|_2^2 \quad \operatorname{\text{subject to}} \quad z = A_B x + b, \ \ x = w
$$

Let's construct a dual by assigning prices $$\mu$$ and $$\nu$$ to the violation of each set of constraints, and separating the minimization over each variable:

$$
\begin{aligned}
q(\mu, \nu) 
 &= \inf_{x,z,w} \left\{\frac{1}{|B|} \sum_{i\in B} \phi(z_i) + r(w) + \frac{1}{2\eta} \|x - x_t\|_2^2 + \mu^T(A_B x + b - z) + \nu^T (x - w) \right\} \\
 &= \color{blue}{\inf_x \left\{  \frac{1}{2\eta} \|x - x_t\|_2^2 + (A_B^T \mu + \nu)^T x \right\}} + \color{purple}{\inf_z \left\{ \frac{1}{|B|} \sum_{i\in B} \phi(z_i) - \mu^T z \right\}} + \color{green}{\inf_w \left\{ r(w)- \nu^T w \right\}} + \mu^T b.
\end{aligned}
$$

We've already encountered the purple part in a previous post - it can be written in terms of the convex conjugate $$\phi^*$$:

$$
\text{purple} = -\frac{1}{|B|}\sum_{i \in B} \phi^*(|B| \mu_i)
$$

 The green part is also straightforward, and is exactly $$-r^*(\nu)$$, where $$r^*$$ is the convex conjugate of the regularizer $$r$$. Finally, the blue part, despite being some cumbersome, is a simple quadratic minimiation problem over $$x$$, so let’s solve it by equating the gradient of the term inside the $$\inf$$ with zero:

$$
\frac{1}{\eta}(x - x_t) + A_B^T \mu - \nu = 0.
$$

By re-arranging, we obtain that the equation is solved at $$x = x_t - \eta(A_B^T \mu - \nu)$$. Recall, that if strong duality holds, that’s exactly the rule for computing the optimal $$x$$ from the optimal $$(\mu, \nu)$$ pair. Let’s substitute the above $$x$$ into the blue term, and after some algebraic manipulations obtain:

$$
\text{blue} = \frac{1}{2\eta} \left\| \color{brown}{x_t - \eta (A_B^T \mu - \nu)} - x_t \right\|_2^2 + (A_B^T \mu + \nu)^T [\color{brown}{x_t - \eta (A_B^T \mu - \nu)}] = - \frac{\eta}{2} \| A_B^T \mu - \nu \|_2^2 + (A_B x_t)^T \mu + x_t^T \nu
$$

Summarizing everything, we have:

$$
q(\mu, \nu) = \color{blue}{- \frac{\eta}{2} \| A_B^T \mu - \nu \|_2^2 + (A_B x_t)^T \mu + x_t^T \nu} \color{purple}{-\frac{1}{|B|}\sum_{i \in B} \phi^*(|B| \mu_i)} \color{green}{-r^*(\nu)} + b^T \mu.
$$

The blue part is a concave quadratic, and $$-\phi^*$$ and $$-r^*$$ are both concave. Well, seem that we’ve done it, haven’t we? Not quite! Recall that the dimension of $$\nu$$ is the same as the dimension of $$x$$, so we haven’t reduced the problem’s dimension at all! If we have a huge model parameter vector $$x$$, we’ll have a huge dual variable $$\nu$$.

# Have we reached the glass ceiling?

The above two failures to come up with an efficient algorithm for computing $$x_{t+1}$$ for mini-batches in the regularized convex-on-linear losses might make us wonder - is it even possible to implement the method in this setting? Well, it turns out that if we insist on using an *off-the-shelf* optimization solver, it might be very hard. But if we are willing to write our own, it is possible. 

Indeed, consider the dual problem we derived in take 1. We know that the dual function being maximized is concave. If we can compute its gradient, we can write our own fast gradient method, i.e. FISTA[^fista] or Nesterov’s accelerated gradient[^nag] method, to solve it. Note that I am not referring to a stochastic optimization algorithm for training a model, but a fully deterministic one for solving a simple optimization problem. Such methods can be quite fast. Furthermore, if we can compute the Hessian matrix of $$q$$ we can employ Newton’s method[^newton], and solve the dual problem even faster, in a matter of a few milliseconds. However, writing convex optimization solvers is beyond the scope of this post.

# What’s next?

The entire post series was devoted to deriving efficient implementations of the proximal point methods to various _generic_ problem classes. Contrary to the above, I would like to devote the next, and last blog post of this series to implementing the method on a specific, but interesting problem. Stay tuned!



[^fista]: Beck A. & Teboulle M. (2009) A fast iterative shrinkage-thresholding algorithm for linear inverse problems. _SIAM Journal on Imaging Science_, 2(11), 183–202.
[^nag]: Nesterov Y. (1983) A method for solving the convex programming problem with convergence rate O(1/k^2). _Dokl. Akad. Nauk SSSR 269_, 543-547
[^newton]: [https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)

