# ChaosNet

Experimenting with the ChaosNet architecture from the paper [ChaosNet](https://doi.org/10.1063/1.5120831)


This method uses GLS neurons.

$$ T(x) = {x \over b} , 0<=b   $$
$$      = {(1-x) \over (1-b)} , b<=1 $$    


We get the firing time, i.e the time during which the iterates are in the neighborhood of the threshold.
Using this we get the probabilities and then averaging them gives a representation of the class values.
By comparing this encoding for each class using cosine distance we get the predicted class.

There is no method for hyperparameter tuning yet, so they are found by randomly trying values.
The values depend from dataset to dataset and have no  method for fixing it beforehand.

The values used here from [parameters](https://github.com/HarikrishnanNB/ChaosNet/blob/master/chaosnet/parameterfile.py)


### MNIST

--------------------------------------------

b = 0.3310 

q = 0.336

length = 20000

num_classes = 10

check = "Sk-B"

method = "TT-SS" 

epsilon = 0.01

------------------------------------
