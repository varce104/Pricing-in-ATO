# Thesis title: Pricing decision in Assemble-to-Order Systems

The codes correspond to the different configurations of the multi component, multi product ATO model. Some key definitions before going to the model itself:

1. The main problem is the coordination of decisions like what component to order, what product to assemble and at what price, all that on a finite horizon while dealing with price-sensitive uncertain demand and a positive components lead times. The most important part of this problem is the sistem is an Assemble-to-Order one, this means that instead of having prucrement and inventory of products (for example a distribution center), the firm manages components, which are required to "aasemble" the products the firm sells (like car manufacturers or Apple, in the Apple store you can choose the storage, ram memory, screen size, or color of a macBook, all these components are in inventory, and the MacBook is assembled once the order is received).

2. The problem is modeled as a Two-Stage Stochastic Program. What this means is that, for every time period, decisions are split in two: before and after uncetainty is realized. Another way to see it is separating every period in two, in decisions made at the beggining and at the end. For this problem, two stage works as follows: all the first stage decisions are made simultaneously, and one uncertainty is revealed, all the second stage decisions are made, depending on the scenario of the uncertainty (i.e, you can make as much second stage decicions as the amount of scenarios of the model). In the near future, we will explore Multi-Stage programming, a more close to real life approach, but significantly more complex, computationally speaking.

## Base Model: Pricing as a parameter
The first two configurations considers first stage decision of the amount of components to order in every period, while the second stage is the quantity of products to assemble. Pricing for the moment isn't a decision variable, this allows us just to provide a base model to further make modifications. It also provide a computationally treatable model to study uncertainty impact, with metrics like Value of Stochastic Solutions (VSS) or Expected Value of Perfect Information (EVPI). With this considerations, we provide two models, with a minor but relevant modification:

1. *Model (i):* Stochastic Demand and Deterministic (and positive) Components Lead Times.
2. *Model (ii):* Stochastic Demand and Lead Times.

While the intention is just to focus on the codes, it is precise to note that in the mathematical formulation (as well as the code), the lead times are switched between "influenced" by the scenarios or not. 

## Two-Stage Non-Linear Model: Pricing as a Decision Variable
Again, we provide two configurations for this case, but this time we make substantial changes. First, the pricing is now a decision variable, incorporated as a binary variable, which chooses a price in every period in a discrete and finite set of prices (like a range of prices). The other important change is that the demand goes from a stochastic parameter to a non-increasing lienar function depending on the price, with additive and multiplicative stochastic components or "noise".

The consecuences of this new formulation makes the objective function a non-linear function, with the pricing variable multiplying the production decision. While this make the problem way more complex for a solver, small instances are still computationally tractables. The model, like previous, are as follows:

1. *Model (iii):* Pricing decision, price-sensitive stochastic demand and deterministic lead times.
2. *Model (iv):* Pricing decision, price-dependant demand and stochastic lead times.

This far we constructed this models. Is in the work to calculate at least VSS for the models *(iii)* and *(iv)*. It is also in testing the tractability and gap of the model for certain relaxed variables, like the price (as a discrete or real variable), or the relaxed problem itself.

We will see how this project goes on, but most certainly will shift to a Multi-Stage formulation. idk how am I gonna model the lead times but oh well, if we manage to do it, it will be huuuge research relevance :)
