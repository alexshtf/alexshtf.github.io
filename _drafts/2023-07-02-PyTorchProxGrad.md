---
≠layout: post
title:  “To prune or not to prune? Adventures with L1 regularization and PyTorch."
tags: [machine-learning, optimization, proximal-gradient]
description: We remind our readers about an old technique to prune model weights - L1 regularization, and Lasso as a special case. Then, we give a teaser showing what we can achieve with PyTorch, and begin developing a module to achieve such results.
comments: true
---

# Intro

Overall, there are two main techniques to reduce the size of a neural network *pruning* and *quantization*. This and the next few posts will be devoted to pruning, namely, removing "unimportant" weights from the network to reduce its size to meet resource constraints. 

The idea of removing unimportant weights from a model is not new, and dates back to a well-known technique for linear regression, called [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)), which boils down to solving an L1-regularized linear regression model:
$$
\min_{\mathbf{w}} \quad \|\mathbf{X} \mathbf{w}-\mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_1
$$
where $$\mathbf{X}$$ is a matrix whose rows are the training samples, $$\mathbf{y}$$ is the vector of labels, and $$\|\mathbf{x}\|_1 =\sum_{i=1}^n |x_i|$$ is the L1-norm penalty term.

With $$\alpha=0$$, we obtain the ordinary least squares problem with features $$\mathbf{X}$$ and labels $$\mathbf{y}$$. When $$\alpha>0$$, the L1 regularization term takes effect. It is [well-known](https://en.wikipedia.org/wiki/Lasso_(statistics)) that the L1 norm "promotes sparsity", meaning that an optimal $$\mathbf{w}$$ tends to have more zeroes as $$\alpha$$ increases. L1 regularization, of course, does not have to be applied to a least squared problem. Indeed, we can train a model $$\hat{y}(\mathbf{w}, \mathbf{x})$$ on the set $$(\mathbf{x}_i, y_i)_{i=1}^N$$ by minimizing:
$$
\min_{\mathbf{w}} \quad \sum_{i=1}^N \ell(\hat{y}(\mathbf{w}, \mathbf{x}_i), y_i) + \alpha \|\mathbf{w}\|_1
$$


Going back to the linear regression case, the `scikit-learn` library has the [sklearn.linear_model.**Lasso**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) class to solve such problems, but if we try to do it by ourselves using a gradient-based method, such as the optimizers available in PyTorch, we are up to an un-pleasant surprise: the resulting vector $$\mathbf{w}$$ is *not* going to be sparse. The reason is simple - the derivatives of the $$\alpha \|\mathbf{w}\|_1$$ term are not sparse, so there is no reason to believe that whatever our optimizer converges to, is sparse. To produce sparse solutions, L1 regularization requires different techniques - we have to treat the L1 norm "directly" instead of computing its gradients. This is exacltly what the *proximal gradient* family of methods do, which drive many Lasso solvers. 

In this and several posts afterwards, we will discuss the method and create an easy to use generic implementation for PyTorch. Then, we try it out on neural networks instead of just linear regression models as an alternative for pruning. Namely, our aim is generating sparse weight matrices by balancing between minimizing the training loss and having a small "sparsity-promoting" norm. When we are done, the library's users will be able to write snippets like the one below (the interesting  parts marked by (A) and (B)) to employ L1 regularization and variants, without having to understand the math behind it:

```python
class Net(nn.Module):
    def __init__(self, input_size, output_size, reg_coef=0.1):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, ...)
        self.l2 = nn.Linear(...)
        self.l3 = nn.Linear(..., output_size)
        self.prox_dict = linear_l1reg(reg_coef, self.l1, self.l2, self.l3) # <-- (A)

    def forward(self, x):
        output = nn.functional.relu(self.l1(x))
        output = nn.functional.relu(self.l2(output))
        output = self.l3(output)
        return output
      
# create model and optimizer
model = Net(input_dim, output_dim, reg_coef=alpha)  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.register_step_post_hook(adam_prox_hook(model.prox_dict))  # <-- (B)

# train the model
for epoch in range(num_epochs):
    train_model(trainset, model, optimizer)
    val_loss = test_model(val_set, model, optimizer)
    print(f'epoch = {epoch}, loss = {val_loss}')
    
```

Before digging deeper, we would like to note a conceptual difference betwen the pruning and the regularization approach. Pruning comprises of two steps: **training** - find the best model we can, and **pruning** - modify the model remove entries which we believe have little effect. That's conceptually different from L1 regularization - we only have the **training** step, which attempts to find the best model it can given a regularized loss. The model *is not* modified in any way after training.

# The proximal-gradient method

We first do a short (re-)introduction of two mathematical concepts, and then use them to explain the method.

## Gradient descent via the proximal view

We begin the journey by recalling what we saw in the [first post]({% post_url 2020-01-31-ProximalPointWarmup %}) in the series about proximal-point methods. We usually see the gradient step on a function  $$f(\mathbf{w})$$ taught as 'take a small step in the direction of its negative gradient', and described by the celebrated formula
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t).
$$
But there is a different and equivalent view - the well-known[^prox] _proximal view_:  
$$
\mathbf{w}_{t+1} = \operatorname*{argmin}_{\mathbf{w}} \left\{ 
    {\color{blue}{f(\mathbf{w}_t) + \nabla f(\mathbf{w}_t)^T (\mathbf{w} - \mathbf{w}_t)}} +  {\color{red}{ \frac{1}{2\eta} \| \mathbf{w} - \mathbf{w}_t\|_2^2}} \tag{*}
\right\}.
$$

The blue part in the formula above is the tangent, or the first-order Taylor approximation at $$\mathbf{w}_t$$, while the red part is a measure of proximity to $$\mathbf{w}_t$$. In other words, the gradient step can be interpreted as:

> Find a point which balances between descending along the tangent at $$\mathbf{w}_t$$, and staying in close proximity to $$\mathbf{w}_t$$.

The balance is controlled by the step-size parameter $$\eta$$. Larger $$\eta$$ puts less emphasis on the red proximity term, and thus allows us to take a step farther away from the current iterate $$\mathbf{w}_t$$. Another intuitive interpretation comes from visualizing it in one-dimension - we are just minimizing a tangent parabola. The step-size controls the "width" of the parabola - larger width means larger steps.

![parabola]({{ "/assets/gradient_descent_parabola.webp" | absolute_url }})

So, to summarize, the well-known gradient step is equivalent to mimizing a quadratic model of the desired function.

## Moreau's proximal operator

For some constant $$\alpha > 0$$, the [proximal operator](https://en.wikipedia.org/wiki/Proximal_operator) is defined by
$$
\operatorname{prox}_{\alpha \phi}(\mathbf{x}) = \operatorname*{argmin}_\mathbf{u} \left\{ \alpha \phi(\mathbf{u}) + \frac{1}{2} \| \mathbf{x} - \mathbf{u} \|_2^2 \right\}
$$
We already discussed this concept in a [previous post]({% post_url 2020-04-04-ProximalConvexOnLinearCont %}) in a different context. Since we can multiply the term inside the $$\operatorname{argmin}$$ by any positive constant without changing the semantics, multiplying by $$\frac{1}{\alpha}$$ produces the following equivalent definition that is usually found in textbooks:
$$
\operatorname{prox}_{\alpha \phi}(\mathbf{x}) = \operatorname*{argmin}_\mathbf{u} \left\{  \phi(\mathbf{u}) + \frac{1}{2\alpha} \| \mathbf{x} - \mathbf{u} \|_2^2 \right\}
$$
Intuitively, the meaning is "find a point that balances between minimizing $$\eta \phi$$ and being close to $$\mathbf{x}$$". It turns out that there are many functions $$\phi$$ for which a closed-form formula exists. Optimization textbooks usually contain a table of those, just like high-school books contain tables of elementary function derivatives. See, for example, [this](https://www.tau.ac.il/~becka/Prox_Computations_Tables.pdf) table from the book _First-Order Methods in Optimization_[^fom]. In particular, for $$\phi(\mathbf{x}) = \| \mathbf{x} \|_1$$ , we have
$$
\operatorname{prox}_{\alpha \phi}(\mathbf{x}) = \max(\mathbf{0}, |\mathbf{x}| - \alpha \mathbf{1}) \cdot \operatorname{sgn}(\mathbf{x}), \tag{PROX-L1}
$$
where $$\mathbf{0}$$ is a vector of zeroes, $$\mathbf{1}$$ is a vector of ones, and all operations are component-wise. With PyTorch it's even simpler - it is implemented in the `torch.nn.functional.softshrink` function.

## The proximal gradient method

Having the ingredients in place, let's bake the cake. In general, we aim to minimize a function of the form

$$
f(\mathbf{w}) = h(\mathbf{w}) + R(\mathbf{w}), \tag{A}
$$
where $$h(\mathbf{w})$$ is the data-fitting term - the average loss over the data-set, and $$R(\mathbf{w})$$ is the regularizer.  The gradient method approximates the entire function $$f$$. But we do not have to - we can choose which parts of $$f$$ we approximate, and which we do not. Approximating _only_ the data-fitting term $$h$$ using its tangent at $$\mathbf{w}_t$$, leads to:

$$
\mathbf{w}_{t+1} = \operatorname*{argmin}_{\mathbf{w}} \left\{ 
    {\color{blue}{h(\mathbf{w}_t) + \nabla h(\mathbf{w}_t)^T (\mathbf{w} - \mathbf{w}_t)}} + R(\mathbf{w})+  {\color{red}{ \frac{1}{2\eta} \| \mathbf{w} - \mathbf{w}_t\|_2^2}} 
\right\}\tag{*}
$$
Note, that $$R$$ remains intact. No gradients of $$R$$ appear. With some tedious algebra, and discarding constants that do not depend on $$\mathbf{w}$$, such as $$h(\mathbf{w}_t)$$, we can re-arrange the terms inside the $$\operatorname{argmin}$$ and obtain:
$$
\mathbf{w}_{t+1} = \operatorname*{argmin}_{\mathbf{w}} \left\{ R(\mathbf{w}) + \frac{1}{2\eta} \| \mathbf{w} - (\mathbf{w}_t - \eta \nabla h(\mathbf{w}_t)) \|.
\right\}
$$
Whoa! That's exactly the proximal operator of $$R$$:
$$
\mathbf{w}_{t+1} = \operatorname{prox}_{\eta R} (\mathbf{w}_t - \eta \nabla h(\mathbf{w}_t)).
$$
The method, of course, is not new. It is **very** old. Just typing "proximal gradient" in Google Scholar results in an enormous number of papers with thousands of citations each. We can decompose the formula of $$\mathbf{w}_{t+1}$$ into two steps, by first computing the term inside the $$\operatorname{prox}$$:
$$
\begin{aligned}
\mathbf{z}_{t+1} &= \mathbf{w}_t - \eta \nabla h(\mathbf{w}_t) \\
\mathbf{w}_{t+1} &= \operatorname{prox}_{\eta R}(\mathbf{z}_{t+1})
\end{aligned}
$$
We can now see some structure - we handle $$h$$ and $$R$$ separately - first we perform a gradient step using $$h$$, and then a prox step using $$R$$. 

Stochastic variants exist, where instead of the gradient $$\nabla h(\mathbf{w}_t)$$ we have a vector $$\mathbf{g}_t$$, that is an estimate of the gradient. Usually, the estimate is the gradient over a mini-batch of data when not using momentum, and a weighted sum of past mini-batch gradients when using momentum. In that case, in each step we compute:
$$
\begin{aligned}
\mathbf{z}_{t+1} &= \mathbf{w}_t - \eta \nabla \mathbf{g}_t \\
\mathbf{w}_{t+1} &= \operatorname{prox}_{\eta R}(\mathbf{z}_{t+1})
\end{aligned}
$$
In terms of PyTorch, it means that in our training loop, first we perform an `optimizer.step()` , and then do some additional logic for the prox step. Note, that the formula we derived has $$\eta$$ as a scalar step size, and it is the same step-size for all model parameters. So at this stage we work with SGD, but we will eventually cover coordinate-wise step-sizes, like in Adam. 

Now, let's take a closer look at the case of L1 regularization. Recalling the equation (PROX-L1) above, we see that the proximal is going to zero-out items of $$\mathbf{z}_{t+1}$$ that are "too small". To be precise, if $$R(\mathbf{w}) = \alpha \| \mathbf{w} \|_1$$, then all elements whose magnitude is below $$\eta \alpha$$ are going to be zeroed out! We just saw another perspective of why L1 regularization "promotes sparsity" - we can directly see it from its proximal operator!

## Naive implementation

 In this example we will work with the well-known california housing data-set, available in `scikit-learn`, so let's begin by loading it, and standardizing the features and the regression target:
```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

x, y = fetch_california_housing(return_X_y=True)
x = StandardScaler().fit_transform(x)
y = StandardScaler().fit_transform(y.reshape(-1, 1)).squeeze(1)

dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
```

Now, let's build a three-layer neural network as our predictive model:

```python
from torch import nn

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, 2 * input_size)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(2 * input_size, input_size // 2)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(input_size // 2, 1)

    def forward(self, x):
        output = nn.functional.relu(self.l1(x))
        output = nn.functional.relu(self.l2(output))
        output = self.l3(output)
        return output
```

And now let's implement the L1-norm proximal operator in Equation (PROX-L1):

```python
import torch

def prox_l1(param, coef):
    param_sign = torch.sign(param)
    prox = param_sign * torch.maximum(param.abs() - coef, torch.zeros_like(param))
		return prox
```

Well, now we can implement our training loop

```
from torch.utils.data import DataLoader

num_epochs = 10

```



# Implementing with optimizer hooks

# Let's test it!

# What if we have a budget of non-zeros?

# Conclusion and teaser

# References

[^prox]: Polyak B. (1987). Introduction to Optimization. _Optimization Software_
[^fom]: Back A. (2017) First Order Methods in Optimization
