---
layout: post
title:  "Interpreting eigenvalue models"
tags: []
description: "TODO"
comments: true
image: assets/pow_spec_props_norms_45.png
series: "Eigenvalues as models"
---

# Intro

We continue our discussion of machine-learned models of the form
$$
f({\boldsymbol x};{\boldsymbol A}_{0:n}) = \lambda_k \Bigl({\boldsymbol A}_0 + \sum_{i=1}^n x_i {\boldsymbol A}_i\Bigr),
$$
where $${\boldsymbol A}_i$$ are learned symmetric matrices, and $$\lambda_k$$ is the $$k$$-th smallest eigenvalue. We touched one aspect of interpretability in a previous post - what is the order of importance of the features $$x_1, \dots, x_n$$. But there are other aspect, namely, what does this model actually compute? How can we reason about it, or explain it to our collegues or to a regulator?   We began this series from saying that this is a "neuron" that is solving an optimization problem. So in this post we shall focus on the different kinds of optimization problems that  $$f({\boldsymbol x}, {\boldsymbol A}_{0:n})$$ solves, and what interpretations we can give to them, and why we should care.

# Courant mini-max principle

Eigenvectors in linear algebra are typically presented as the directions along which a matrix _stretches_ vectors without changing their direction (flipping is permitted), and the amount of stretch are the corresponding eigenvalues. So here is another characterizations of the $$k$$-th smallest eigenvalue, known as the Courant mini-max principle, named after Richard Courant. It's a bit "hairy", so let's first present it, and then interpret it:
$$
\lambda_k({\boldsymbol A}) = \max_{{\boldsymbol C} \in \mathbb{R}^{(k-1)\times d}} \min_{{\boldsymbol u} \in \mathbb{R}^d} \left\{ {\boldsymbol u}^T {\boldsymbol A} {\boldsymbol u} : \| {\boldsymbol u} \|_2 = 1, \, {\boldsymbol C}{\boldsymbol u} = {\boldsymbol 0}\right\}
$$
First, we see that is a bi-level optimization problem, which we can think of as a _game_ with two turns. The first player is choosing matrices $${\boldsymbol C}$$ that have $$k-1$$ columns. Exactly one column less than the index of our eigenvalue. In response, the second player is allowed to choose unit vectors $${\boldsymbol u}$$ that are in the null-space of $${\boldsymbol C}$$, or _orthogonal_ to the columns of $${\boldsymbol C}$$. 

The objective of second player is to minimize the cost $${\boldsymbol u}^T {\boldsymbol A} {\boldsymbol u}$$ - so they are choosing the vector appropriately. But the objective of the first player is to make their opponent pay as much as possible, so they are choosing a "worst case" matrix  $${\boldsymbol C}$$. 



