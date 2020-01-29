---
≠layout: post
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
> find a point which balances between descending along the tangent at $$x_t$$, and staying in close proximity to $$x_t$$.

The balance is controlled by the step-size parameter $$\eta$$. Larger $$\eta$$ puts less emphasis on the proximity term, and thus allows us to take a step farther away from $$x_t$$. 

To convince ourselves that $$(\text{*})$$ above is indeed the gradient step in disguise, we recall that by Fermat's principle we have $$\nabla H_t(x_{t+1}) = 0$$, or equivalently

$$
\nabla f(x_t) + \frac{1}{2\eta} (x_{t+1} - x_t) = 0.
$$

By re-arranging and extracting $$x_{t+1}$$ we recover the gradient step.

# Beyond the black box
A first order approximation is reasonable if we know nothing about the function $$f$$, except for the fact that it is differentiable. But what if we **do** know something about $$f$$? Let us consider an extreme case - we would like to exploit as much as we can about $$f$$, and define


$$
x_{t+1} = \operatorname*{argmin}_x \left\{
    \color{blue}{f(x)} + \frac{1}{2\eta} \color{red}{\|x - x_t\|_2^2}
\right\}
$$


The  idea is known as the stochastic proximal point method[^ppm], or implicit learning[^impl]. Note, that when $$f$$ is “too complicated”, we might not have any efficient method to compute $$x_{t+1}$$, which makes this method impractical for many types of loss functions. However, it turns out to be useful for many losses.

Let us consider a simple example - linear regression. Our aim is to minimize 

$$
\frac{1}{2n} \sum_{k=1}^n (a_i^T x + b_i)^2
$$

Thus, every $$f$$ is of the form $$f(x)=\frac{1}{2}(a^T x + b)^2$$,  and our computational steps are of the form:
$$
x_{t+1}=\operatorname*{argmin}_x \left\{ P_t(x)\equiv
 \frac{1}{2}(a^T x + b)^2 + \frac{1}{2\eta} \|x - x_t\|^2
\right\} \tag{**}
$$
To derive an explicit formula for $$x_{t+1}$$ let’s solve the equation $$\nabla P_t(x_{t+1}) = 0$$:


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
x_{t+1}=x_t - \underbrace{\frac{\eta (a^T x_t+b)}{1+\eta \|a\|_2^2}}_{\alpha_t} a. \tag{S}
$$


Ah! Finally! Now we have arrived at a formula which can be implemented in $$O(d)$$ operations, where $$d$$ is the dimension of $$x$$. We just need to compute the coefficient $$\alpha_t$$, and take a step in the direction opposite to $$a$$. 

An interesting thing to observe here is that large step-sizes $$\eta$$ do not lead to an overly large coefficient $$\alpha_t$$, since $$\eta$$ appears both in the numerator and the denominator. Intuitively, this might lead to a more stable learning algorithm - it is less sensitive bad step-size choice. 

Now it’s time to reflect back on the derivation process and the final result, pose some interesting questions, and draw conclusions. The final result appears simple, yet it required trickery and advanced mathematical tools to obtain it. Is there some apparent simplicity hiding here, which we haven’t discovered yet? It turns out there is, and in the next post we will discuss a simple and generic method to derive computational steps for the proximal point method for an entire family of loss functions $$f$$,  including least squares and logistic regression, and develop a small Python framework for minimizing losses in this family.

It is also a good time to give a short preview of our future post subjects. 

- a simple and generic method to derive and implement the proximal point approach for a large family of losses
- The proximal point step is not always tractable. We consider approaches which only _partially_ approximate $$f$$ , thus preserving some information about the loss functions.
- The quadratic proximity term is not always the ‘best fit’ for any loss function $$f$$. We discuss additional proximity terms, and their advantages for some problem families.
- The idea suggested above treats each training sample separately. What about mini-batches? 

# Experiment

First, we will look at the performance of our method against several optimizers which are widely used in existing machine learning frameworks: AdaGrad, Adam, SGD, and  will test the stability of our algorithm w.r.t the step-size choice, since our intuition suggested that our method might be more stable than the ‘black box’ approaches. The python code we describe below, which reproduces our results, can be found in this [git repo](https://github.com/alexshtf/proxptls).

We use the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) to test our algorithms on a linear regression model attempting to predict housing prices $$y$$ based on the data vector $$a \in \mathbb{R}^3$$ comprising the number of rooms, population lower status percentage, and average pupil-teacher ratio by the linear model:


$$
y = p^T \beta +\alpha
$$


To that end, we will attempt to minimize the mean squared error over all our samples $$(p_j, y_j)$$, namely:


$$
\min_{\alpha, \beta} \quad \frac{1}{2n} \sum_{j=1}^n (p_j^T \beta +\alpha-y_j)^2
$$


Note, that the problem can be solved exactly using a direct method for linear least squares problems. Thus, we can obtain the optimal loss exactly and use it as our baseline. So let’s begin by writing code which loads and normalizes our data-set, and computes the exact least-squares solution:

```python
import numpy as np
import pandas as pd

def compute_opt_loss(P, b):
    # pad with a column of ones - the for each observation, the
    # coefficient of the `alpha' parameter is 1.
    Q = np.pad(P, pad_width=[(0, 0), (0, 1)], constant_values=1)
    
    # solve the  least-squares problems
    opt_w, opt_loss, residual, svd = np.linalg.lstsq(Q, b) 
    opt_loss /= P.shape[0]
    
    return opt_loss

boston = pd.read_csv('boston.csv')
P = boston[['RM', 'LSTAT', 'PTRATIO']].to_numpy()
P = minmax_scale(P)
b = boston['MEDV'].to_numpy()
b = minmax_scale(b)

optimal_loss = compute_opt_loss(P, b)

```

Next, let’s create a PyTorch dataset object which will be used to train our model:

```python
import torch.utils.data as td
...

train_set = td.TensorDataset(torch.tensor(X), torch.tensor(b))
```

We will also need the linear model itself:

```python
import torch
...

class LinReg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.empty((1,), dtype=torch.float64), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty((3,), dtype=torch.float64), requires_grad=True)

        torch.nn.init.normal_(self.alpha)
        torch.nn.init.normal_(self.beta)

    def forward(self, p):
        return self.alpha + torch.dot(self.beta, p)

```

Now comes the interesting part. Our experiments require three customization points - how do we construct our  optimizer, how do we train on each sample, and what do we perform at the end of each training epoch. Thus, we define a base-class which we call `ExperimentPlugin`:

```python 
from abc import ABC, abstractmethod


class ExperimentPlugin(ABC):
    @abstractmethod
    def construct(self, model, lr):
        pass

    @abstractmethod
    def process_sample(self, x, y, pred):
        pass

    @abstractmethod
    def end_epoch(self):
        pass
```

We can now use this class to write a method which performs an experiment with each optimizer. An experiment consists of running a training session comprising of $$E$$ epochs, several times for each learning rate, and produce a table with the best training loss achieved at each training session. We record the deviation from our baseline - the optimal training loss, since visualization in that case is easier. The code is a little longer, but quite straightforward:

```python
def run_experiment(name, plugin, train_set, epochs, attempts, lrs, opt_loss):
    losses = pd.DataFrame(columns=['lr', 'epoch', 'attempt', 'loss'])

    for lr in lrs:
        for attempt in attempts:
            model = LinReg()

            plugin.construct(model, lr)

            with tqdm(epochs, desc=f'plugin = {name}, lr = {lr}, attempt = {attempt}', unit='epochs', ncols=100) as tqdm_epochs:
                for epoch in tqdm_epochs:
                    train_loss = 0
                    for x, y in td.DataLoader(train_set, shuffle=True, batch_size=1):
                        xx = x.squeeze(0)

                        model.zero_grad()

                        pred = model.forward(xx)
                        loss = (pred - y) ** 2
                        train_loss += loss.item()
                        loss.backward()

                        plugin.process_sample(xx, y, pred)

                    train_loss /= len(train_set)
                    losses = losses.append(pd.DataFrame.from_dict(
                        {'loss': [train_loss - opt_loss.item()],
                         'epoch': [epoch],
                         'lr': [lr],
                         'attempt': attempt}), sort=True)

                    plugin.end_epoch()

    best_loss_df = losses[['lr', 'attempt', 'loss']].groupby(['lr', 'attempt'], as_index=False).min()
    return best_loss_df[['lr', 'loss']]

```

The `OptimizerPlugin` class uses a regular PyTorch optimizer to train the model. Its implementation is straightforward. It is constructed with two function pointers - one for creating the optimizer, and another one for creating a step-size scheduler. The reason for a scheduler is simple - we run SGD using a decaying step-size, which decreases on each epoch. But AdaGrad and Adam are executed with a constant step-size. The implementation of the optimizer-based experiment plugin is straightforward:

```python
class OptimizerPlugin(ExperimentPlugin):
    def __init__(self, make_optimizer, make_scheduler):
        self.make_optimizer = make_optimizer
        self.make_scheduler = make_scheduler
        self.optimizer = None
        self.scheduler = None

    def construct(self, model, lr):
        self.optimizer = self.make_optimizer(model, lr)
        self.scheduler = self.make_scheduler(self.optimizer, lr)

    def process_sample(self, x, y, pred):
        self.optimizer.step()

    def end_epoch(self):
        self.scheduler.step()
```

The second plugin implements our proximal point method, according to the formula (S) above:

```python
import torch

class ProxPointPlugin(ExperimentPlugin):
    def __init__(self):
        self.model = None
        self.lr = None

    def construct(self, model, lr):
        self.model = model
        self.lr = lr

    def process_sample(self, x, y, pred):
        with torch.no_grad():
            numerator = self.lr * (pred - y)
            denominator = (1 + self.lr * (1 + torch.dot(x, x)))
            coeff = numerator / denominator

            self.model.beta -= coeff * x
            self.model.alpha -= coeff

    def end_epoch(self):
        pass


```

Now, we can run experiments with all our optimization strategies and plot the results. We will use the excellent seaborn library for plotting:

```python
import seaborn as sns
import matplotlib.pyplot as plt
...

epochs = range(0, 100)
attempts = range(0, 20)
lrs = np.geomspace(0.001, 100, num=60)
experiments = [
    ('Proximal point', ProxPointPlugin()),
    ('Adagrad', OptimizerPlugin(make_optimizer=lambda model, lr: opt.Adagrad(model.parameters()),
                                make_scheduler=lambda optimizer, lr: LambdaLR(optimizer,
                                                                              lr_lambda=[lambda i: lr]))),
    ('Adam', OptimizerPlugin(make_optimizer=lambda model, lr: opt.Adam(model.parameters()),
                             make_scheduler=lambda optimizer, lr: LambdaLR(optimizer,
                                                                           lr_lambda=[lambda i: lr]))),
    ('SGD', OptimizerPlugin(make_optimizer=lambda model, lr: opt.SGD(model.parameters(), lr=lr),
                            make_scheduler=lambda optimizer, lr: LambdaLR(optimizer,
                                                                          lr_lambda=[lambda i: lr / math.sqrt(1 + i)])))
]

experiment_results = [run_experiment(name, plugin, train_set, epochs, attempts, lrs, optimal_loss).assign(name=name)
                      for name, plugin in experiments]
result_df = pd.concat(experiment_results)
result_df = result_df[result_df['loss'] < 10000]
result_df.head()

sns.set()
ax = sns.lineplot(x='lr', y='loss', hue='name', data=result_df, err_style='band')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

```

Here are the results:

<TODO - insert plot here>

What a pleasant surprise! The proximal point method indeed seems more stable then the others. We paid the price of dealing with our loss function directly instead of approximating it, but we gained stability. Is it a coincidence? It turns out that the question is no - and you are encouraged to read the excellent paper[^stbl] by John Duchi and Hilal Asi about stability in such methods.

# Next

Constructing a custom optimizer for each loss seems infeasible. In the next post we will construct a generic proximal-point optimizer for a large family of losses, which include linear least squares and logistic regression, and will demonstrate stability in training logistic regression models as well.

# References

[^ppm]: Bianchi, P. (2016). Ergodic convergence of a stochastic proximal point algorithm. _SIAM Journal on Optimization_, 26(4), 2235-2260.
[^impl]: Kulis, B., & Bartlett, P. L. (2010). Implicit online learning. _In Proceedings of the 27th International Conference on Machine Learning (ICML-10)_ (pp. 575-582).
[^prox]: Polyak B. (1987). Introduction to Optimization. _Optimization Software_

[^stbl]: Asi, H. & Duchi J. (2019). Stochastic (Approximate) Proximal Point Methods: Convergence, Optimality, and Adaptivity *SIAM Journal on Optimization 29(3)* (pp. 2257–2290)

