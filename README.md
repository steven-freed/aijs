# aijs
An AI JavaScript library that makes it simple to; read delimited files, train algorithms with your own training data sets, and pick and choose from some of the most popular AI algorithms known to man.

## Algorithms Supported
### Classification
1. K-Nearest Neighbors
2. Naive Bayes
3. Perceptron

## Documentation
### KNN
1. train
```
Trains the algorithm
@param {[[]]} featMatrix feature matrix of training data
@param {[]} labelVector class labels vector
```

2. classify
```
Classifies a matrix of feature vectors
@param {[[]]} featMatrix features matrix you wish to be classified
@param {{ k: 3 }} options An optional param. k to specify how many neighbors to use
```

### NaiveBayes
1. train
```
Trains the algorithm
@param {[[]]} featMatrix feature matrix of training data
@param {[]} labelVector class labels vector
```

2. classify
```
Classifies a matrix of feature vectors
@param {[[]]} featMatrix features matrix you wish to be classified
```

### Perceptron
1. train
```
Trains the algorithm
@param {[[]]} featMatrix feature matrix of training data
@param {[]} labelVector class labels vector
@param {options={ epochs: 100 }} options optional epochs parameter, 100 by default
```

2. classify
```
Classifies a matrix of feature vectors
@param {[[]]} featMatrix features matrix you wish to be classified
```

## Util Functions
1. readFile
```
Reads a delimited file synchronously line by line seperating the data into
the feature matrix and label (classification) vector
 
@param {String} filePath path to a delimiter seperated spread sheet
@param {skipHeader: true, delimiter: ',', encoding: 'utf8'} options skipHeader skips the 
header of the file, delimiter can be specified if using anything other than a csv, encoding
can be specified
@returns {[featureMatrix, labelVector]} array of your result data 
```

2. getAccuracy
```
Calculates the accuracy given the algorithm of your choices output 
and the known results of the true classifications 

@param {[]} classifications classifications received by algorithm
@param {[]} classifiedSet already known classifications
@returns {Number} the percentage of accuracy
```
