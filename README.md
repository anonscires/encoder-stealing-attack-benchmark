# From Queries to Clones: A Systematic Study of Encoder Stealing Attacks - Research Code

This repository contains the implementation code for our research paper on benchmarking encoder stealing attacks against self-supervised learning models.


The victim needs to be trained using the [ssl-attacks-defenses](https://github.com/cleverhans-lab/ssl-attacks-defenses) code.
We follow the ssl-attacks-defenses and train the victim for 200 epochs. The Direct Extraction surrogate can be trained through this code itself using InfoNCE paper.


For training the surrogate, individual [StolenEncoder](https://github.com/liu00222/StolenEncoder), [Con-Steal](https://github.com/zeyangsha/Cont-Steal), [RDA](https://github.com/ShuchiWu/RDA) be used.
These codes are also present in the respecitve directory in the repo. They all have been standardized to use the same model clas for ease of running experiments at scale.
