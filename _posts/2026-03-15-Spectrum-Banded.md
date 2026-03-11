---
layout: post
title:  "Cheaper eigenvalue training and inference"
tags: []
description: "TODO"
comments: true
# image: assets/pow_spec_recurrent.png
series: "Eigenvalues as models"
---

# Intro

In the last post we discussed the meaning of our model family

$$
f({\boldsymbol x};{\boldsymbol A}_{0..n}) = \lambda_k \Bigl({\boldsymbol A}_0 + \sum_{i=1}^n x_i {\boldsymbol A}_i\Bigr),
$$

where each $$\boldsymbol A_i$$ is a symmetric matrix. In the last post we discussed what do these models predict, and how can we explain them to ourselves and other stakeholders. Beforehand, we also discussed GPU acceleration to make training and inference faster. Speed is important, but so is _cost_, and fast GPUs may pe expensive. Here, our aim is not only making it faster, but also cheaper, by making the eigenvalue problem easier to solve even on weaker hardware. We begin our exploration from theory, which immediately yields practical applications.

# Simultaneous simplification

Recall, that for any orthogonal matrix $${\boldsymbol Q} \in \mathbb{R}^{d \times d}$$, we have

$$
\lambda_k(\boldsymbol A) = \lambda_k({\boldsymbol Q}^T \boldsymbol A {\boldsymbol Q}),
$$

So our model family is invariant under such orthogonal similarity transformations, meaning a model with matrices $$\boldsymbol A_i$$ is identical to a model with matrices $$\boldsymbol Q^T \boldsymbol A_i \boldsymbol Q$$ for any orthogonal $$\boldsymbol Q$$.

One of the interesting phenomena in linear algebra are _simultaneously diagonalizable_ matrices. A set of matrices $${\boldsymbol A}_i$$ is simultaneously diagonalizable if there exists an orthogonal matrix $${\boldsymbol Q}$$ such that $${\boldsymbol Q}^T {\boldsymbol A}_i {\boldsymbol Q}$$ is diagonal for all $$i$$. In other words, the same matric $$\boldsymbol Q$$ diagonalizes all matrices simultaneously. 

If we restrict ourselves to models where all of our learned matrices are simultaneously diagonalizable, we can equialently just assume all matrices are diagonal:

$$
f({\boldsymbol x};{\boldsymbol A}_{0:n}) = \lambda_k \Bigl(\operatorname{diag}({\boldsymbol a}_0) + \sum_{i=1}^n x_i \operatorname{diag}({\boldsymbol a}_i)\Bigr).
$$

So what is the $$k$$-th eigenvalue of this matrix? It's just the $$k$$-th smallest entry of the vector 

$$
{\boldsymbol a}_0 + \sum_{i=1}^n x_i {\boldsymbol a}_i.
$$

On the one hand, it's extremely easy eigenvalue problem. But we actually lost almost all of the expressive power, since it's just a convoluted way to describe a piecewise linear function of $${\boldsymbol x}$$.

But there is another family of matrices for which the eigenvalue problem is easy - _symmetric tridiagonal_ matrices, meaning, matrices of the form:

$$
\mathcal{T}(\boldsymbol a, \boldsymbol b) = 
\begin{pmatrix}
a_1    & b_1    & 0      & \dots  & 0      \\
b_1    & a_2    & b_2    & \dots  & 0      \\
0      & b_2    & a_3    & \ddots & \vdots \\
\vdots & \vdots & \ddots & \ddots & b_{n-1} \\
0      & 0      & \dots  & b_{n-1} & a_n
\end{pmatrix}.
$$

Such a matrix is defined by two vectors, the main diagonal $$\boldsymbol a \in \mathbb{R}^d$$, and the off-diagonal $$\boldsymbol b \in \mathbb{R}^{d-1}$$. We stand on the shoulders of giants, and use the impressive legacy of numerical analysis research, given to us in the form of `scipy.linalg.eigh_tridiagonal` on a silver platter.
