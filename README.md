# GenerativeTopologyOptimization

This work is very much still ongoing, but the flow relies on the following figure:

![Alt text](rect8.png)

The script ```trainer.py``` trains all components of the above figure (minus the surrogate model) in three stages:

1. An embedding stage which trains the PointNet embedding of surface coordinates and auto-encoder and decoder together (a la [Diffusion-SDF](https://arxiv.org/abs/2211.13757))
2. A reconstruction stage which trains the SDF network (this part is very much ongoing)
3. A diffusion model which learns to generate latent representations of shapes

The ```latent_bayes.py``` script runs the whole flow as described in the above figure, but is still under construction. Ultimately, whether a surrogate physics model is needed or not, the ```latent_bayes.py``` script will run some kind of posterior sampling algorithm in the latent space of the autoencoder and perhaps use some kind of Bayesian optimization-inspired acquisition function to explore the space of possible designs.
