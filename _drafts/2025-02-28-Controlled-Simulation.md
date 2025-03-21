---
layout: post
title:  "Interpretable time-series simulation"
tags: [bernstein-polynomial,spline,random]
description: Controlling global trends of synthetically generated time series using splines, Bernstein polynomials, and other totally-positive bases.
comments: true
image: /assets/polyedral_cone_layer.png
---

# Intro

Suppose we would like to try out a new super-duper algorithm whose input are hourly metrics of an ad campaign, such as the number of ad views (impressions), clicks, and revenue in each hour. We want to make sure our algorithm is resilient to various scenarios, so we don't have to "babysit" while it is running for months, or even years, and in many cases simulation is a good solution.

But why should we simulate? We can just gather a few years-worth of historical data and try our algorithm! Or can we? First, history is not necessarily predictive of the future - businesses grow, advertisers join and leave, properties where we show ads change. Second, and more important - the amount of history we may be limited because of regulations. We may want our algorithm to be resilient to some extremities and abrupt changes, like Black Friday, but we may no longer have the data from last Black Friday!

So now, suppose one of the things we want to simulate is the number of hourly ad impressions some hypothetical ad campaign delivers. We know there is seasonality - periodic behavior within days, and within weeks. We also know there are global trends. Seasonal behavior may be easy to simulate - we have sines and cosines, and we decide the frequency and phase. But what about the global trends? Can we simulate them as well in a manner that is interpretable and controlled? This is something we intend to explore in this post.

# Simulation framework

Foo

# Totally positive bases

Bar

# Cubic splines

Baz

# Bernstein polynomials

Cuux

# Summary

