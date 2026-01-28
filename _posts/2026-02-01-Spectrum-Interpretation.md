---
layout: post
title:  "Interpreting eigenvalue models"
tags: []
description: "TODO"
comments: true
image: TODO
series: "Eigenvalues as models"
---

# Intro

We continue our discussion of machine-learned models of the form

$$
f({\boldsymbol x};{\boldsymbol A}_{0:n}) = \lambda_k \Bigl({\boldsymbol A}_0 + \sum_{i=1}^n x_i {\boldsymbol A}_i\Bigr),
$$

where $${\boldsymbol A}_i$$ are learned symmetric matrices, and $$\lambda_k$$ is the $$k$$-th smallest eigenvalue. We touched one aspect of interpretability in a previous post - what is the order of importance of the features $$x_1, \dots, x_n$$. But there are other aspect, namely, what does this model actually compute? How can we reason about it, or explain it to our collegues or to a regulator?   We began this series from saying that this is a "neuron" that is solving an optimization problem. So in this post we shall focus on the different kinds of optimization problems that  $$f({\boldsymbol x}, {\boldsymbol A}_{0:n})$$ solves, and what interpretations we can give to them, and why we should care.

# A game between two players

Eigenvectors in linear algebra are typically presented as the directions along which a matrix _stretches_ vectors without changing their direction (flipping is permitted), and the amount of stretch are the corresponding eigenvalues. So here is another characterizations of the $$k$$-th smallest eigenvalue, known as the Courant mini-max principle, named after Richard Courant. It's a bit "hairy", so let's first present it, and then interpret it:

$$
\lambda_k({\boldsymbol A}) = \max_{{\boldsymbol C} \in \mathbb{R}^{(k-1)\times d}} \min_{{\boldsymbol u} \in \mathbb{R}^d} \left\{ {\boldsymbol u}^T {\boldsymbol A} {\boldsymbol u} : \| {\boldsymbol u} \|_2 = 1, \, {\boldsymbol C}{\boldsymbol u} = {\boldsymbol 0}\right\}
$$

First, we see that is a bi-level optimization problem, which we can think of as a _game_ with two turns. The first player is choosing matrices $${\boldsymbol C}$$ that have $$k-1$$ columns. Exactly one column less than the index of our eigenvalue. In response, the second player is allowed to choose unit vectors $${\boldsymbol u}$$ that are in the null-space of $${\boldsymbol C}$$, or _orthogonal_ to the columns of $${\boldsymbol C}$$. 

The objective of second player is to pay as little as possible, where $${\boldsymbol u}^T {\boldsymbol A} {\boldsymbol u}$$ is the cost. So they are choosing the vector appropriately. But the objective of the first player, of course, is to make their opponent pay as much as possible, so they are choosing a "worst case" matrix  $${\boldsymbol C}$$. You have probably guessed, but the optimal vector $${\boldsymbol u}$$ is a corresponding eigenvector. So what role does the matrix $${\boldsymbol A}$$ play here? Well, we can think of $${\boldsymbol u}$$ as a kind of _energy allocation_ vector among a set of resouces, due to its bounded norm, and $$A_{i,j}$$ is the cost associate with every pairwise interaction of resources, since:

$$
{\boldsymbol u}^T {\boldsymbol A} {\boldsymbol u} = \sum_{i=1}^n \sum_{j=1}^n A_{i,j} u_i u_j
$$

Depending on the context, you can give different interpretations. For example, $${\boldsymbol A}$$ represents a set of latent skills a student possesses, and $${\boldsymbol u}$$ is a _test vector_ for pairs of skills.

There is a mirror-image of the above game, if we want to sort eigenvalues from largest to smallest. So the $$k$$-th _largest_ eigenvalue, which is also the $$n-k$$-th smallest one, can be written as
$$
\lambda_{n-k}({\boldsymbol A}) = \min_{{\boldsymbol C} \in \mathbb{R}^{(k-1)\times d}} \max_{{\boldsymbol u} \in \mathbb{R}^d} \left\{ {\boldsymbol u}^T {\boldsymbol A} {\boldsymbol u} : \| {\boldsymbol u} \|_2 = 1, \, {\boldsymbol C}{\boldsymbol u} = {\boldsymbol 0}\right\}
$$

We can think of it in terms of utility rather than cost - each entry in the matrix is a utility associated with a pair of resources. The first player is choosing the matrix so that the second play will get as little utility as possible, whereas the second player, in response, aims to choose a vector $${\boldsymbol u}$$ that will maximize their utility. 

Adopting the $$\max-\min$$ convention, we can write our "neuron" as:
$$
f({\boldsymbol x};{\boldsymbol A}_{0:n}) = \max_{{\boldsymbol C} \in \mathbb{R}^{(k-1)\times d}} \min_{{\boldsymbol u} \in \mathbb{R}^d} \left\{ {\boldsymbol u}^T \left({\boldsymbol A}_0 + \sum_{i=1}^n x_i {\boldsymbol A}_i \right) {\boldsymbol u} : \| {\boldsymbol u} \|_2 = 1, \, {\boldsymbol C}{\boldsymbol u} = {\boldsymbol 0}\right\}
$$


Consequently, each feature is associated with a matrix of some latent "costs" and our features are just the weights of these costs. Suppose you're in the insurance business, and someone asks you what your model is doing. You can explain something like "oh, we're representing each feature of the insured using a table of latent skills to avoid claims, we sum them up, and simulate a game where we aim to elicit their worst-case ability to either avoid damage or absorb it without claiming". You can give an example of what those "latent skills" could be, just like in matrix factorization people explain what would be the latent features in a movie recommendation system.

# A difference of convex functions

