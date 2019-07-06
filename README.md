# aijs
An AI JavaScript library that makes it simple to; read delimited files, train algorithms with your own training data sets, and pick and choose from some of the most popular AI algorithms known to man.

## Algorithms Supported (with their codes to train)
1. K-Nearest Neighbors 'knn'
2. Naive Bayes 'nb'

### Algorithm Functions
1. KNN
```
@param {[[]]} data data you wish to be classified
@param {{ k: 3 }} options An optional param. k to specify how many neighbors to use
```

2. naiveBayes
```
```

### Util Functions
1. readToMatrixAndVector
```
Reads a delimiter seperated file synchronously line by line seperating the data into
the feature matrix and label (classification) vector
 
@param {String} filePath path to a delimiter seperated spread sheet
@param {skipHeader: true, delimiter: ',', encoding: 'utf8'} options skipHeader skips the 
header of the file, delimiter can be specified if using anything other than a csv, encoding
can be specified
@returns {[featureMatrix, labelVector]} array of your result data 
```

2. train
```
Trains your algorithm of choice

@param {String} algo algorithm you want to train 
@param {[[]]} trainingSet the training set provided as set of vectors, aka 2D array
@param {[]} classifiedSet the classified set provided as a single vector of classifications 
corresponding directly with the same array index as the training set
```

3. getAccuracy
```
Calculates the accuracy given the algorithm of your choices output 
and the known results of the true classifications 

@param {[]} classifications classifications received by algorithm
@param {[]} classifiedSet already known classifications
@returns {Number} the percentage of accuracy
```
