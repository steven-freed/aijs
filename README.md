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
@param {{ epochs: 100 }} options optional epochs parameter, 100 by default
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
@param {{skipHeader: true, delimiter: ',', encoding: 'utf8'}} options optional param. skipHeader skips the 
header of the file, delimiter can be specified if using anything other than a csv, encoding
can be specified
@returns {Object} 'featMatrix', 'labelVector'
```

2. getAccuracy
```
Calculates the accuracy given the algorithm of your choices output 
and the known results of the true classifications 

@param {[]} classifications classifications received by algorithm
@param {[]} classifiedSet already known classifications
@returns {Number} the percentage of accuracy
```

## Examples

Perceptron Example
```
var aijs = require('../aijs');

// reads from a comma delimited file and returns a feature matrix and label vector
let {featMatrix, labelVector} = aijs.readFile('data.csv', options={ skipHeader: true, encoding: 'utf8', delimiter: ',' });

// data you want to classify                                          
var testData = [
      [4.8, 3.0, 1.4, 0.1], 
      [5.4, 3.0, 4.5, 1.5], 
      [4.8, 3.0, 1.4, 0.3] 
];

// create Perceptron instance
let perceptron = aijs.Perceptron();

// train your instance
perceptron.train(featMatrix, labelVector);

// classifies your data and returns a vector of classifications
let res = perceptron.classify(testData);

// print the results
console.log('classified as', res);
```
