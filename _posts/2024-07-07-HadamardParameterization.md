---
layout: post
title:  "Fun with sparsity in PyTorch via Hadamard product parametrization"
tags: [pytorch,machine-learning,sparse,sgd,hadamard-parametrization,factorization-machine]
description: We demonstrate how we can reduce model size by pruning un-needed neurons.
comments: true
image: /assets/hadamard_linear_convergence_plot.png
---

# Intro

Models having a high-dimensional parameter space, such as large neural networks, often pose a challenge when deployed on edge devices, due to various constraints. Two remedies are often suggested: _pruning_ and _quantization_. In this post I'd like to concentrate on the idea of pruning, which amounts to removing neurons that we beleive have little or no contribution to the over-all model performance. PyTorch provides various heuristics for model pruning, that are explained in a [tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) in the official documentation.

I'd like to discuss a decades-old alternative idea to those heuristics - L1 regularization. It is "known" to promote sparsity, and there is an end-less amount of resources online explaining why. But there are very little resources explaining **how** this can be achieved in modern ML frameworks, such as PyTorch. I believe there are two major reasons for that.

The first reason is very direct - just adding an L1 regularization term to the cost we differentiate in each training loop, in conjunction with an optimizer such as SGD or Adam, will often _not_ produce sparse solution. You can find plenty of evidence online, such as [here](https://stackoverflow.com/questions/59941063/l1-regularization-neural-network-in-pytorch-does-not-yield-sparse-solution), [here](https://stackoverflow.com/questions/43146015/is-the-l1-regularization-in-keras-tensorflow-really-l1-regularization), or [here](https://stackoverflow.com/questions/50054049/lack-of-sparse-solution-with-l1-regularization-in-pytorch). I want to avoid discussing _why_, and will just say that the reason is in the optimizers - they were not designed to properly handle sparsity-inducing regularizers, and some trickery is required. 

The second reason stems from how software engineering is done. People want to re-use components or patterns. There is a very clear pattern of how PyTorch training is implemented, and we either implement it manually in simple cases, or rely on a helper library, such as [PyTorch Ignite](https://pytorch-ignite.ai/), or [PyTorch Ligntning](https://lightning.ai/docs/pytorch/stable/) to do the job for us.

So can we use sparsity-inducing regularization with PyTorch, that nicely and easily integrates with the existing ecosystem? It turns out that there is an interesting stream of research that facilitates exactly that - the idea of sparse regularization by Hadamard parametrization. I first encountered it in a paper by Peter Hoff[^1], and noticed that the idea has been further explored in several additional papers [^2][^3][^4]. I believe this stream of research hasn't received the attention (pun intended!) it deserves, since it allows an extremely easy way of achieving sparse L1 regularization that _seamlessly integrates_ into the existing PyTorch ecosystem of patterns and libraries. In fact, the code is so embarrasingly simple that I am surprised that such parametrizations haven't become popular.

The basic idea is very simple. Suppose we aim to find model weights $$\mathbf{w}$$ by minimizing the L1 regularized loss over our training set:

$$
\tag{P} \min_{\mathbf{w}} \quad \frac{1}{n} \sum_{i=1}^n \ell_i(\mathbf{w}) + \lambda \|w\|_1
$$

We reformulate the problem $$(P)$$ above by representing $$\mathbf{w}$$ as a component-wise product of two vectors. Formally, $$\mathbf{w} = \mathbf{u} \odot \mathbf{v}$$ where $$\odot$$ is the component-wise (or Hadamard) product. And instead of solving $$(P)$$ we solve the problem below:

$$
\tag{Q} \min_{\mathbf{u},\mathbf{v}} \quad \frac{1}{n} \sum_{i=1}^n \ell_i(\mathbf{u} \odot \mathbf{v}) + \lambda \left( \|\mathbf{u}\|_2^2 + \| \mathbf{v} \|_2^2 \right)
$$

Note, that $$(Q)$$ uses L2 regularization! As it turns out[^1], any local minimum $$(Q)$$ is also a local minimum of $$(P)$$.  L2 regularization is native to PyTorch in the form of the `weight_decay` parameter to its optimizers. But more importantly, [parametrizations](https://pytorch.org/tutorials/intermediate/parametrizations.html) are also a native beast in PyTorch!

We first begin with implementing this idea in PyTorch for a simple linear model, and then extend it to Neural Networks. This is, of course, _not_ the best method to achieve sparsity. But it's an extremmely simple one, easy to try out for your model, and fun!  As customary, the code is available in a [notebook](https://github.com/alexshtf/alexshtf.github.io/blob/master/assets/hadamard_parameterization.ipynb) that you can deploy on Google Colab.

# Parametrizing a linear model

In this section we will demonstrate how to implement Hadamard parametrization in PyTorch to train a linear model on a data-set, and verify that we indeed achieve sparsity similar to a truly optimal solution of $$(P)$$. We regard the solution achieved by [CVXPY](www.cvxpy.org), which is a well-known convex optimization package for Python, as an "exact" solution.

## Setting up the dataset

We begin from the data which we use throughout this section. We will use the Madelon dataset, which is a synthetic data-set that was used for the NeurIPS 2003 feature selection challenge. It's available from openml as data-set [1485](https://www.openml.org/search?type=data&status=active&id=1485), and therefore we can use the `fetch_openml` function from scikit-learn to fetch it:

```python
from sklearn.datasets import fetch_openml

madelon = fetch_openml(data_id=1485, parser='auto')
```

To get a feel of what this data-set looks like, let's print it:

```python
print(madelon.frame)
```

The output is:

```
       V1   V2   V3   V4   V5   V6   V7   V8   V9  V10  ...  V492  V493  V494  \
0     485  477  537  479  452  471  491  476  475  473  ...   481   477   485   
1     483  458  460  487  587  475  526  479  485  469  ...   478   487   338   
2     487  542  499  468  448  471  442  478  480  477  ...   481   492   650   
3     480  491  510  485  495  472  417  474  502  476  ...   480   474   572   
4     484  502  528  489  466  481  402  478  487  468  ...   479   452   435   
...   ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...   ...   ...   ...   
2595  493  458  503  478  517  479  472  478  444  477  ...   475   485   443   
2596  481  484  481  490  449  481  467  478  469  483  ...   485   508   599   
2597  485  485  530  480  444  487  462  475  509  494  ...   474   502   368   
2598  477  469  528  485  483  469  482  477  494  476  ...   476   453   638   
2599  482  453  515  481  500  493  503  477  501  475  ...   478   487   694   

      V495  V496  V497  V498  V499  V500  Class  
0      511   485   481   479   475   496      2  
1      513   486   483   492   510   517      2  
2      506   501   480   489   499   498      2  
3      454   469   475   482   494   461      1  
4      486   508   481   504   495   511      1  
...    ...   ...   ...   ...   ...   ...    ...  
2595   517   486   474   489   506   506      1  
2596   498   527   481   490   455   451      1  
2597   453   482   478   481   484   517      1  
2598   471   538   470   490   613   492      1  
2599   493   499   474   494   536   526      2
```

So it's a classification data-set with 500 numerical features, and two classes. Naturally, we will use the binary cross-entropy loss in our minimization problem.

At this stage our objective is just demonstrating properties of the model fitting procedure, rather than evaluating the performance of the model. Thus, for simplicity, we will not split into train / evaluation sets, and operate on the entire data-set. 

To make it more friendly for model training, let's first rescale it to zero mean and unit variance, and extract labels as values in $$\{0, 1\}$$:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

scaled_data = np.asarray(StandardScaler().fit_transform(madelon.data))
labels = np.asarray(madelon.target.cat.codes)
```

## Exact L1 regularization using CVXPY

Now let's find an optimal solution of the L1 regularized problem $$(P)$$ using [CVXPY](www.cvxpy.org), which is a Python framework for accurate solution of convex optimization problems. For Logistic Regression, the loss of the sample $$(\mathbf{x}_i, y_i)$$ is:

$$
\ell_i(\mathbf{w}) = \ln(1+\exp(\mathbf{w}^T \mathbf{x_i})) - y \cdot \mathbf{w}^T \mathbf{x}_i
$$

This is obviously convex, due to the convexity of $$\ln(1+\exp(x))$$, which is modeled by the `cvxpy.logistic` function. The corresponding CVXPY code for constructing an object representing $$(P)$$ is:

```python
import cvxpy as cp

# (coef, intercept) are the vector w.
coef = cp.Variable(scaled_data.shape[1])
intercept = cp.Variable()
reg_coef = cp.Parameter(nonneg=True)

pred = scaled_data @ coef + intercept  # <--- this is w^T x
loss = cp.logistic(pred) - cp.multiply(labels, pred)
mean_loss = cp.sum(loss) / len(scaled_data)
cost = loss + reg_coef * cp.norm(coef, 1)
problem = cp.Problem(cp.Minimize(cost))
```

Of course, we don't know at this stage which regularization coefficient to use to achieve sparsity, so let's begin with $$10^{-4}$$:

```python
reg_coef.value = 1e-4
problem.solve()
print(f'Loss at optimum = {loss.value:.4g}')
```

I got th efollowing output:

```
Loss at optimum = 0.5466
```

Let's also plot the coefficients. The `plot_coefficients` function below just contains boilerplace to make the plot nice, and ability to specify transparency and color for other parts of this post, where we want to make several plots on the same axes:

```python
import matplotlib.pyplot as plt

def plot_coefficients(coefs, ax_coefs=None, alpha=1., color='blue', **kwargs):
  if ax_coefs is None:
    ax_coefs = plt.gca()
  markerline, stemlines, baseline = ax_coefs.stem(coefs, markerfmt='o', **kwargs)
  ax_coefs.set_xlabel('Feature')
  ax_coefs.set_ylabel('Weight')
  ax_coefs.set_yscale('asinh', linear_width=1e-6)  # linear near zero, logarithmic further from zero

  stemlines.set_linewidth(0.25)
  markerline.set_markerfacecolor('none')
  markerline.set_linewidth(0.1)
  markerline.set_markersize(2.)
  baseline.set_linewidth(0.1)

  stemlines.set_color(color)
  markerline.set_color(color)
  baseline.set_color(color)

  stemlines.set_alpha(alpha)
  markerline.set_alpha(alpha)
  baseline.set_alpha(alpha)
 
plot_coefficients(coef.value)
```

I got the following plot:

![madelon_cvxpy_0.0001]({{"/assets/madelon_cvxpy_0.0001.png" | absolute_url}})

Note that the y-axis appears logarithmic to to the $$\operatorname{arcsinh}$$ scale. Doesn't look sparse at all! So let's try a larger coefficient:

```python
reg_coef.value = 1e-2
problem.solve()
print(f'Loss at optimum = {loss.value:.4g}')
plot_coefficients(coef.value)
```

The output is

```
Loss at optimum = 0.6188
```

The plot I obtained:

![madelon_cvxpy_0.01]({{"/assets/madelon_cvxpy_0.01.png" | absolute_url}})

Now it looks much sparser!  Let's store the coefficients vector, we will need it in the remainder of this section to compare it to the results we achieve with PyTorch:

```python
cvxpy_sparse_coefs = coef.value.copy()
```

I don't know if this is a 'good' feature selection strategy for this specific dataset, but it's not our objective. Our objective is showing how to implement Hadamard parametrization in PyTorch that recovers a similar sparsity pattern. So let's do it!

## Using PyTorch parametrization

Parametrizations in PyTorch allow representing any learnable parameter as a function of other learnable parameters. Typically, this is used to impose constraints. For example, we may represent a vector representing discrete event probabilities as the soft-max operation applied to a vector of arbitrary real values. A parametrization in PyTorch is just another module. Here is an example:

```python
class SimplexParametrization(torch.nn.Module):
  def forward(self, x):
    return torch.softmax(x)
```

Now, suppose our model has a parameter called `vec` which we'd like to constrain to lie in the probability simplex. It can be done in the following manner:

```python
torch.nn.utils.parametrize.register_parametrization(model, 'vec', SimplexParametrization())
```

Viola! 

Since a parametrization is just another module, it can have its own learnable weights! So we can use this fact to easily parametrize the weights of a `torch.nn.Linear` module: we will regard its original weights as $$\mathbf{u}$$, the parametrization module will have its own weigths $$\mathbf{v}$$, and will compute $$\mathbf{u} \odot \mathbf{v}$$. Here is the code:

```python
import torch

class HadamardParametrization(torch.nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.v = torch.nn.Parameter(torch.ones(out_features, in_features))

  def forward(self, u):
    return u * self.v
```

Note, that I initialized the $$\mathbf{v}$$ vector to a vector of ones. This is because the first time a parametrization is applied, the `forward` function is called to compute the parametrized value, and I want to use the deep mathematical fact that $$1$$ is neutral w.r.t the multiplication operator to keep the original weight unchanged. 

Let's apply it to a linear layer and inspect its trainable parameters to get a feeling:

```python
layer = torch.nn.Linear(8, 1)
torch.nn.utils.parametrize.register_parametrization(layer, 'weight', HadamardParametrization(8, 1))
```

That's it! Now if we train our linear model using PyTorch optimizers with `weight_decay`, we will in fact apply L1 regularization to the original weights. The weight decay is exactly equivalent to the L1 regularization coefficient. Under the hood, the `layer.weight` parameter is now represented as a Hadamard product of two tensors.

To get a feeling of what happens under the hood, let's inspect our linear layer after applying the parametrization:

```python
for name, param in layer.named_parameters():
  print(name, ': ', param)
```

The output I got is:

```python
bias :  Parameter containing:
tensor([-0.1233], requires_grad=True)
parametrizations.weight.original :  Parameter containing:
tensor([[-0.0035,  0.2683,  0.0183,  0.3384, -0.0326,  0.1316, -0.1950, -0.0953]],
       requires_grad=True)
parametrizations.weight.0.v :  Parameter containing:
tensor([[1., 1., 1., 1., 1., 1., 1., 1.]], requires_grad=True)
```

We can see there are three trainable parameters. The bias of the linear layer, the original weight of the linear layer, which we now treat as the $$\mathbf{u}$$ vector, and the weight of the `HadamardParametrization` module, which is initialized to ones, which we treat as the $$\mathbf{v}$$ vector. What happens if we try to access the `weight` of the linear layer? Let's see:
```python
print(layer.weight)
```

Here is the output:

```python
tensor([[-0.0035,  0.2683,  0.0183,  0.3384, -0.0326,  0.1316, -0.1950, -0.0953]],
       grad_fn=<MulBackward0>)
```

But it has a `MulBackward` gradient back-propagation function, because under the hood it is computed as a product of two tensors.

## Training a parametrized logistic regression model

To see our parametrization in action, we will need three components. First, a function that implements a pretty standard PyTorch training loop. Something that looks familiar, and without any trickery. Second, a function that plots its results. Third, a function that integrates the two above ingredients to train a Hadamard-parametrized logistic regression model.

Here is our pretty-standard PyTorch training loop. It returns the training loss achieved in each epoch in a list, so that we can plot it:

```python
from tqdm import trange

def train_model(dataset, model, criterion, optimizer, n_epochs=500, batch_size=8):
  epoch_losses = []
  for epoch in trange(n_epochs):
    epoch_loss = 0.
    for batch, batch_label in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
      # compute predictiopn and loss
      batch_pred = model(batch)
      loss = criterion(batch_pred, batch_label)
      epoch_loss += loss.item() * torch.numel(batch_label)
      
      # invoke the optimizer using the gradients.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
		
    epoch_losses.append(epoch_loss / len(dataset))
  return epoch_losses
```

Our second ingredient are plotting functions. Here is a function that plots the epoch losses:

```python
def plot_convergence(epoch_losses, ax=None):
  if ax is None:
    ax = plt.gca()

  ax.set_xlabel('Epoch')
  ax.set_ylabel('Cost')
  ax.plot(epoch_losses)
  ax.set_yscale('log')
  last_iterate_loss = epoch_losses[-1]
  ax.axhline(last_iterate_loss, color='r', linestyle='--')
  ax.text(len(costs) / 2, last_iterate_loss, f'{last_iterate_loss:.4g}',
          fontsize=12, va='center', ha='center', backgroundcolor='w')

```

To get a feeling of what the output looks like, let's plot a a dummy list simulatinga loss of $$\exp(-\sqrt{i})$$ in the $$i$$-th epoch:

```python
plot_convergence(np.exp(-np.sqrt(np.arange(100))))
```

![hadamard_convergence_plot_dummy]({{"/assets/hadamard_convergence_plot_dummy.png" | absolute_url}})

We can see a plot of the achieves loss on a logarithmic scale, and a horizontal line denoting the loss at the last epoch. We would also like to see the coefficients of our trained model, just like we did with the CVXPY models. So here is a function that plots the losses on the left, and the coefficients on the right. The coefficients are plotted together with 'reference' coefficients, so that we can visually compare our model to some reference. In our case, the reference coefficients are the ones we obtained from CVXPY.

```python
def plot_training_results(model, losses, ref_coefs):
  # create figure and decorate axis labels
  fig, (ax_conv, ax_coefs) = plt.subplots(1, 2, figsize=(12, 4))
  plot_coefficients(ref_coefs, ax_coefs, color='blue', label='Reference')
  plot_coefficients(model.weight.ravel().detach().numpy(), ax_coefs, color='orange', label='Hadamard')
  ax_coefs.legend()
  plot_convergence(losses, ax_conv)
  plt.tight_layout()
  plt.show()
```

Our third and last ingredient is the function that integrates it all. It trains a Hadamard parametrized model, and plots the coefficients and the epoch losses:

```python
import torch.nn.utils.parametrize

def train_parameterized_model(alpha, optimizer_fn, ref_coefs, **train_kwargs):
  model_shape = (scaled_data.shape[1], 1)
  model = torch.nn.Linear(*model_shape)
  torch.nn.utils.parametrize.register_parametrization(
      model, 'weight', HadamardParametrization(*model_shape)) # <-- this applies Hadamard parametrization

  dataset = torch.utils.data.TensorDataset(
      torch.as_tensor(scaled_data).float(),
      torch.as_tensor(labels).float().unsqueeze(1))  
  criterion = torch.nn.BCEWithLogitsLoss()  # <-- this is the loss for logistic regression
  optimizer = optimizer_fn(model.parameters(), weight_decay=alpha)
  epoch_losses = train_model(dataset, model, criterion, optimizer, **train_kwargs)
  
  plot_training_results(model, epoch_losses, ref_coefs)
```

Now let's try it out with a regularization coefficient of $$10^{-2}$$. That is exactly the same coefficient we used to obtain the sparse coefficients with CVXPY. However, this is not CVXPY, and we need to also chose an optimizer and its parameters. I used the Adam optimizer with a learning rate of $$10^{-4}$$. And yes, I know[^6] that Adam's weight decay is not exactly L2 regularization, but many use Adam as their go-to optimizer, and I want to demonstrate that the idea works with Adam as well:

```python
from functools import partial

train_parameterized_model(alpha=1e-4,
                          optimizer_fn=partial(torch.optim.Adam, lr=1e-4),
                          ref_coefs=cvxpy_sparse_coefs)
```

Here is the result I got:

![hadamard_linear_convergence_plot]({{"/assets/hadamard_linear_convergence_plot.png" | absolute_url}})

On the left, we can see the convergence plot. On the right, we can see coefficients from both CVXPY and the Hadamard parametrization. They almost coincide, with almost the same sparsity pattern. The training loss, 0.6194, is also pretty close to 6188, which is what we achieved with CVXPY.

Now, having seen that Hadamard parametrization indeed 'induces sparsity', just like its equivalent L1 regularization, we can do something more interesting, and apply it to neural networks.

# Parametrizing a neural network

The concept of sparsity doesn't necessarily fit neural networks in the best way, but a related concept of _group sparsity_ does. We introduce it here, show how it is seamlessly implemented using a Hadamard product parametrization in PyTorch, and conduct an experiment with the famous california housing prices dataset.

## Group sparsity

One caveat of the parametrization technique we saw is that it requires twice as many trainable parameters. For large neural networks this may be prohibitive in terms of time, space, or just the cost of training on the cloud. But with neural networks, it may be enough to produce a zero either at the neuron input level, or at the output level. For example, some of the neurons of a given layer produce a zero, whereas others do not. 

This can be achieved through regularization that induces _group sparsity_, meaning that we would like entire groups of weights to be zero whenever the effect of the group on the loss is small enough. If we define the groups to be the _columns_ of the weight matrices of our linear layers, we will achieve sparsity on neuron inputs. This is because in PyTorch the columns correspond to the of input features of a linear layer.

One way to achieve this, is using the sum of the column norms in the regularization coefficient. For example, suppose we have 3-layer neural network whose weight matrices are $$\mathbf{W}_1 \in \mathbb{R}^{8\times 3}, \mathbf{W}_2\in\mathbb{R}^{3\times 2}$$, and $$\mathbf{W}_3 \in \mathbb{R}^{2\times 1}$$, and we are training over a data-set with $$n$$ samples with cost functions $$\ell_1, \dots, \ell_n$$. Then to induce column sparsity, we should train by minimizing

$$
\begin{align*}
\min_{\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3} \quad \frac{1}{n} \sum_{i=1}^n \ell_i(\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3) + \lambda \Bigl(
	&\| \mathbf{W}_{1,1} \|_2 + \| \mathbf{W}_{1,2} \|_2 +  \| \mathbf{W}_{1,2} \|_3 + \\
  &\| \mathbf{W}_{2,1} \|_2 + \| \mathbf{W}_{2,2} \|_2 +  \\
  &\| \mathbf{W}_{3,1} \|_2
\Bigr),
\end{align*}
$$

where $$\mathbf{P}_i$$ denotes the $$i$$-th column of the matrix $$\mathbf{P}$$. Seems a bit clumsy, but the regularizer just sums up the Euclidean norms of the weight matrix columns. Note, that the norms are **not** squared, so this is not our friendly neighborhood L2 regularization.

It turns out[^2][^3][^4] that this is equivalent to a Hadamard product parametrization _with_ our friendly neighborhood L2 regularization. This means that we can again use the `weight_decay` feature of PyTorch optimizers to achieve column sparsity. As we would expect, the parametrization operates on matrix columns, rather than individual components. A weight matrix $$\mathbf{W} \in \mathbb{R}^{m \times n}$$ will parametrized by the matrix $$\mathbf{U} \in \mathbb{R}^{m \times n}$$ and the vector $$\mathbf{v} \in \mathbb{R}^n$$ as by multiplying each column of $$\mathbf{U}$$ by the corresponding component of $$\mathbf{v}$$:

$$
\mathbf{W} = \begin{bmatrix}
\mathbf{U}_1 \cdot v_1 & \mathbf{U}_2 \cdot v_2 & \cdots & \mathbf{U}_m \cdot v_m
\end{bmatrix}
$$

The implementation in PyTorch is embarasingly simple:

```python
class InputsHadamardParametrization(torch.nn.Module):
  def __init__(self, in_features):
    super().__init__()
    self.v = torch.nn.Parameter(torch.ones(1, in_features))

  def forward(self, u):
    return u * self.v
```

Note, that we use the _broadcasting_ ability of PyTorch to multiply each column of the argument `u` by the corresponding component of `v`.

## Group sparsity in action

To see our idea in action, we shall use the california housing dataset, mainly due to its availability on Google colab. It has 8 numerical features, and a continuous regression target.  Let's load it:

``` python
train_df = pd.read_csv('sample_data/california_housing_train.csv')
test_df = pd.read_csv('sample_data/california_housing_test.csv')
```

The first 5 rows of the train data-set are:

|   longitude |   latitude |   housing_median_age |   total_rooms |   total_bedrooms |   population |   households |   median_income |   median_house_value |
|------------:|-----------:|---------------------:|--------------:|-----------------:|-------------:|-------------:|----------------:|---------------------:|
|     -114.31 |      34.19 |                   15 |          5612 |             1283 |         1015 |          472 |          1.4936 |                66900 |
|     -114.47 |      34.4  |                   19 |          7650 |             1901 |         1129 |          463 |          1.82   |                80100 |
|     -114.56 |      33.69 |                   17 |           720 |              174 |          333 |          117 |          1.6509 |                85700 |
|     -114.57 |      33.64 |                   14 |          1501 |              337 |          515 |          226 |          3.1917 |                73400 |
|     -114.57 |      33.57 |                   20 |          1454 |              326 |          624 |          262 |          1.925  |                65500 |

The first 8 columns are the features, and the last column is the regression target. We make some preprocessing by splitting into training and evaluation set, and Scikit-Learn's `StandardScaler` to standardize the numerical features. Then, we convert everything to PyTorch datasets:

```python
# standardize features
scaler = StandardScaler().fit(train_df)
train_scaled = scaler.transform(train_df)
test_scaled = scaler.transform(test_df)

# conver to PyTorch objects
train_ds = torch.utils.data.TensorDataset(
    torch.as_tensor(train_scaled[:, :-1]).float(),
    torch.as_tensor(train_scaled[:, -1]).float().unsqueeze(1))
test_features = torch.as_tensor(test_scaled[:, :-1]).float()
test_labels = torch.as_tensor(test_scaled[:, -1]).float().unsqueeze(1)
```

Now we are ready. We will use the following simple four-layer neural network to fit the training set:

```python
class Network(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = torch.nn.Linear(8, 32)
    self.fc2 = torch.nn.Linear(32, 64)
    self.fc3 = torch.nn.Linear(64, 32)
    self.fc4 = torch.nn.Linear(32, 1)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.fc4(x)
    return x

  def linear_layers(self):
    return [self.fc1, self.fc2, self.fc3, self.fc4]
```

The first layer has 8 input features, since our data-set has 8 features. Later layers expand it to 64 hidden features, and then shrink back. We don't know if we will need all those dimensions, but that's what we have our sparsity inducing regularization for - so that we can find out. Note that I added a `linear_layers()` method to be able to operate on all the linear layers of the network. It could be done in a generic manner by inspecting all modules and checking which ones are `torch.nn.Linear`, but I want to make the subsequent code simpler. 

Let's inspect our network to see how many parameters it has. To that end, we shall use the torchinfo package:

```python
import torchinfo

network = Network()
torchinfo.summary(network, input_size=(1, 8))
```

Most of the output is not interesting, but one line is:

```
Trainable params: 4,513
```

So our network has 4513 trainable parameters. As we shall see, using sparsity inducing regularization we can let gradient descent (or Adam) discover how many dimensions we need! Let's proceed to training our network with column-parametrized weights:

```python
def parametrize_neuron_inputs(network):
  for layer in network.linear_layers():
    num_inputs = layer.weight.shape[1]
    torch.nn.utils.parametrize.register_parametrization(
        layer, 'weight', InputsHadamardParametrization(num_inputs))

parametrize_neuron_inputs(network)
epoch_costs = train_model(train_ds, network, torch.nn.MSELoss(),
                          n_epochs=200, batch_size=128,
                          optimizer=torch.optim.Adam(network.parameters(), lr=0.002, weight_decay=0.001))
```

Reusing our `plot_convergence` function from the previous section, we can see how the model trains:

```python
plot_convergence(epoch_costs)
```

![hadamard_nn_epoch_costs]({{"/assets/hadamard_nn_epoch_costs.png" | absolute_url}})

We can now also evaluate the performance on the test set:

```python
def eval_network(network):
  network.eval()
  criterion = torch.nn.MSELoss()
  print(f'Test loss = {criterion(network(test_features), test_labels):.4g}')

eval_network(network)
```

The output is:

```python
Test loss = 0.2997
```

So we did not over-fit. The test MSE is similar to the train MSE. Now let's inspect our sparsity. To that end, I implemeted a funciton that plots the matrix sparsity patterns of the four layers, where I regard any entry below some threshold as a zero.   Nonzeros are white, whereas zeros are black. Here is the code:

```python
def plot_network(network, zero_threshold=1e-5):
  fig, axs = plt.subplots(1, 4, figsize=(12, 3))
  layers = network.linear_layers()

  for i, (ax, layer) in enumerate(zip(axs.ravel(), layers), start=1):
    layer_weights = layer.weight.abs().detach().numpy()
    image = layer.weight.abs().detach().numpy() > zero_threshold
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Layer {i}')
  plt.tight_layout()
  plt.show()
 
plot_network(network)
```

![hadamard_nn_sparsity]({{"/assets/hadamard_nn_sparsity.png" | absolute_url}})

Now here is a surprise! It is most apparent in the first layer. Our parametrization is supposed to induce sparsity on the _columns_ of the matrices, but we see that it also induces sparsity on the _rows_. So what's going on? Well, it turns out that if the inputs of some inner layer are unused, because the column weights are zero, we can also zero-out the corresponding rows of the layer before. Since the outputs of the layer before are unused by the layer after, it has no effect on the training loss, but reduces the regularization term. Indeed, careful inspection will show that the rows of the first layer that were fully zeroed out exactly correspond to the columns of the second layer that were zeroed out. This is true to any consequent pair of layers. What's truly amazing is that we didn't have to do anything - gradient descent (or Adam, in this case) 'discovered' this pattern on its own!

Now that we know exactly which rows and columns we can remove, let's write a function that does it. It's a bit technical, and I don't want to go into the PyTorch details, but you can read the code and convince yourself that this is exactly what the function below does for a linear layer - it computes a mask of columns whose norm is negilgibly small, receives the mask from the previous layer, and removes the corresponding rows and columns. 

```python
@torch.no_grad()
def shrink_linear_layer(layer, input_mask, threshold=1e-6):
  # compute mask of nonzero output neurons
  output_norms = torch.linalg.vector_norm(layer.weight, ord=1, dim=1)
  if layer.bias is not None:
    output_norms += layer.bias.abs()
  output_mask = output_norms > threshold

  # compute shrunk sizes
  in_features = torch.sum(input_mask).item()
  out_features = torch.sum(output_mask).item()

  # create a new shrunk layer
  has_bias = layer.bias is not None
  shrunk_layer = torch.nn.Linear(in_features, out_features, bias=has_bias)
  shrunk_layer.weight.set_(layer.weight[output_mask][:, input_mask])
  if has_bias:
    shrunk_layer.bias.set_(layer.bias[output_mask])
  return shrunk_layer, output_mask
```

Now let's apply it to all four layers:

```python
mask = torch.ones(8, dtype=bool)
network.fc1, mask = shrink_linear_layer(network.fc1, mask)
network.fc2, mask = shrink_linear_layer(network.fc2, mask)
network.fc3, mask = shrink_linear_layer(network.fc3, mask)
network.fc4, mask = shrink_linear_layer(network.fc4, mask)
```

Note, that we _replace_ the linear layers of the network with new ones. These new layers do not have a Hadamard parametrization, so now applying weight decay will apply the regular L2 regularuzation we are used to. Let's see how many trainable weights does our network have now:

```python
torchinfo.summary(network, input_size=(1, 8))
```

Here is the output:

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Network                                  [1, 1]                    --
├─Linear: 1-1                            [1, 18]                   162
├─ReLU: 1-2                              [1, 18]                   --
├─Linear: 1-3                            [1, 15]                   285
├─ReLU: 1-4                              [1, 15]                   --
├─Linear: 1-5                            [1, 9]                    144
├─ReLU: 1-6                              [1, 9]                    --
├─Linear: 1-7                            [1, 1]                    10
==========================================================================================
Total params: 601
Trainable params: 601
Non-trainable params: 0
Total mult-adds (M): 0.00
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
==========================================================================================
```

Only 601 trainable parameters! So let's train its 601 remaining weights, now without any parametrizations:

```python
epoch_costs = train_model(train_ds, network, torch.nn.MSELoss(),
                          n_epochs=200, batch_size=128,
                          optimizer=torch.optim.Adam(network.parameters(), lr=0.002, weight_decay=1e-6))
plot_convergence(epoch_costs)
```

![sparse_nn_epoch_costs]({{"/assets/sparse_nn_epoch_costs.png" | absolute_url}})

We can also evaluate its performance:

```python
eval_network(network)
```

The output is:

```python
Test loss = 0.2385
```

## How should we work in practice?

You may have noticed that I manually chose the learning rate and the weight decay for the parametrized network, and a different learning rate and weight-decay for the shrunk network. In practice, we should do hyperparameter tuning, and select the best combination that optimizes for some metric on an evaluation set. Namely, each hyperparameter tuning experiment performs two phases, just like what we did with our neural network in this post. In the first phase, it trains a parametrized network and shrinks it. The parametrization helps 'discover' the correct sparsity pattern. Then, in the second phase, we train the shrunk network, and then evaluates its performance. This is because we have no way of knowing in advance which hyperparameters will induce the 'optimal' sparsity pattern. So a pesudo-code for hyperparameter tuning experiment may like this:

```python
def tuning_objective(phase_1_lr, phase_1_alpha, phase_2_lr, phase_2_alpha):
  network = create_network()
  
  apply_hadamard_parametrization(network)
  train(network, phase_1_lr, phase_1_alpha)
  
  network = shrink_network(network)
  train(network, phase_2_lr, phase_2_alpha)
  
  return evaluate_performance(network)
```

So I recommend relying on the hyperparameter tuner to discover good parameters for the above objective, just like we rely on gradient descent to discover the 'right' sparsity pattern.

The idea of training first with sparsity inducing regularization, and then again without it, is not new. In fact, many statisticians working with [Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)) do something similar:  we first use Lasso for feature selection, and then re-train the model on the selected features _wihtout_ Lasso.  This is because sparsity inducing regularization typically hurts performance by shrinking the remaining model weights too aggressively. This was a kind of "crasftman-knowledge", but recently some papers [^8][^9] formally analyzed this approach and made it more publicly known. This idea also has some resemblance to relaxed Lasso[^10].

Finally, if we have an inference "budget", we may choose to inform our hyperparameter tuner that the cost for exceeding the budget is very high. For example, in the above tuning objective, we can replace the return statement by:

```python
  return evaluate_performance(network) + 1000 * max(number_of_parameters(network) - budget, 0)
```

This way the tuner will try to avoid exceeding the budget, because of the high cost of each additional model parameter. Of course, the cost doesn't have to be that extreme,  and we can make it much less than 1000 units for each additional parameter, depending on our requirements.

# Conclusions

The beauty of sparsity inducing regularization is that we let our optimizer discover the sparsity patterns, instead of doing extremely expensive neural architecture search. And the beauty of Hadamard-product parametrization is that it lets us re-use existing optimizers of our ML frameworks to add sparsity-inducing regularizers, without having to write specialized custom optimizers. Maybe to some of you this may sound like [Klingon](https://en.wikipedia.org/wiki/Klingon_language), but for readers familiar with proximal minimization:  writing a proximal operator for group sparsity inducing norm with componentwise learning rates using PyTorch, so that it is also GPU friendly, is extremely hard. But with Hadamard parametrization we don't need to.

Beyond neural networks, the idea can be also applied to convolutional nets - we can make each filter a "group", and let gradient descent discover how many filters, or channels, we need in each convolutional layer. We can also apply it to factorization machines[^7], to discover the 'right' latent embedding dimension. The idea is extremely versatile! 

I hope you had fun reading it as much as I had fun writing it, and see you in the next post!

---

[^1]: Hoff, Peter D. "Lasso, fractional norm and structured sparse estimation using a Hadamard product parametrization." Computational Statistics & Data Analysis 115 (2017): 186-198.
[^2]: Ziyin, Liu, and Zihao Wang. "spred: Solving L1 Penalty with SGD." International Conference on Machine Learning. PMLR, 2023.
[^3]:Kolb, Chris, et al. "Smoothing the edges: a general framework for smooth optimization in sparse regularization using Hadamard overparametrization." arXiv preprint arXiv:2307.03571 (2023).
[^4]: Poon, Clarice, and Gabriel Peyré. "Smooth over-parameterized solvers for non-smooth structured optimization." Mathematical programming 201.1 (2023): 897-952.
[^5]: Ouyang, Wenqing, et al. "Kurdyka-Lojasiewicz exponent via Hadamard parametrization." arXiv preprint arXiv:2402.00377 (2024).
[^6]: Loshchilov, Ilya, and Frank Hutter. "Decoupled Weight Decay Regularization." International Conference on Learning Representations (2019).
[^7]: Rendle, Steffen. "Factorization machines." 2010 IEEE International conference on data mining. IEEE, 2010.
[^8]: Belloni, Alexandre, and Victor Chernozhukov. "ℓ1-penalized quantile regression in high-dimensional sparse models." (2011): 82-130.
[^9]: BELLONI, ALEXANDRE, and VICTOR CHERNOZHUKOV. "Least squares after model selection in high-dimensional sparse models." Bernoulli (2013): 521-547.
[^10]: Meinshausen, Nicolai. "Relaxed lasso." Computational Statistics & Data Analysis 52.1 (2007): 374-393.

