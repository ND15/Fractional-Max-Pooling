# Fractional-Max-Pooling

## This repository is a crude implementation of Fractional Max Pooling written by Benjamin Graham in 2014. 
Link to the paper : https://arxiv.org/abs/1412.6071
This Model is implemented on the basis of the Research paper on Fractional Max Pooling,<br/>
where the author trains the model without any training set augmentation.<br/>

### Main Aspects
The main motivation of fractional max pooling is to reduce the spatial size of the image by 
a factor of alpha where 1 < alpha < 2. The pooling regions can either be disjoint or overlapping.
The pooling regions can be defined in two ways.
* Disjoint : P = [a<sub>i−1</sub>, a<sub>i</sub> − 1] × [b<sub>j−1</sub>, b<sub>j</sub> − 1]
* Overlapping : P = [a<sub>i−1</sub>, a<sub>i</sub>] × [b<sub>j−1</sub>, b<sub>j</sub>1]<br/>
The pooling regions can be generated either randomly or in a pseudorandom order.
For generating the regions in a pseudorandom order, the sequence have to take the form of:
                      ai = ceiling(α(i + u)), α ∈ (1, 2), with some u ∈ (0, 1).      

### This Repository
* Dataset is MNIST
* Here the input layer size is 28 * 28
* Architecture : 6 layers of (32nC2 - FMP(1.25)) - C2- C1 - Output, n = 1,...6
    
* The model is trained for 1 epoch on the MNIST dataset, where it achieved an accuracy of <br/>
    0.9540 and a loss of 0.1846.
