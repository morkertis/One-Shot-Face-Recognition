# One-Shot-Face-Recognition
One-Shot Face Recognition Using Siamese Neural Networks

In this work, we used convolutional neural networks (CNNs) to carry out the task of facial recognition. 
More specifically, we have implemented a one-shot classification solution. 
In order to achieve this goal, we implemented a Siamese Neural Network(SNN), according to the paper:
[Siamese Neural Networks for One-shot Image recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
Our main goal was to successfully execute a one-shot learning task for previously unseen objects.
Given two facial image of previously unseen persons, our SNN will try and determine whether they are the same person.

In addition, for import the photos we change the part of sklearn API.


references:
1. [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
2. [Contrastive Loss](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
