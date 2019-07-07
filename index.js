module.exports = {
    readToMatrixAndVector,
    train,
    getAccuracy,
    KNN,
    naiveBayes
};

var fs = require('fs');

/**
 * Trained Set constants
 */
knnTrainedSet = {};
nbTrainedSet = undefined;

/**
 * Reads a delimiter seperated file synchronously line by line seperating the data into
 * the feature matrix and label (classification) vector
 * 
 * @param {String} filePath path to a delimiter seperated spread sheet
 * @param {skipHeader: true, delimiter: ',', encoding: 'utf8'} options skipHeader skips the 
 * header of the file, delimiter can be specified if using anything other than a csv, encoding
 * can be specified
 * @returns {[featureMatrix, labelVector]} array of your result data 
 */
function readToMatrixAndVector(filePath, options={ skipHeader: true, delimiter: ',', encoding: 'utf8' }) {
    let fileLines = fs.readFileSync(filePath, { encoding: options.encoding }).split('\n');

    let labelVector = [];
    let featureMatrix = [];
    for (let line = 0; line < fileLines.length; line++) {
        if (options.skipHeader && line === 0) 
            continue; 
        // loops features in a line to cast strings to numbers if possible
        let features = fileLines[line].split(options.delimiter);
        for (let feat = 0; feat < features.length; feat++) {
            if (!isNaN(features[feat])) // if number
                features[feat] = Number(features[feat]);
            else if (String(features[feat].toLowerCase()) == 'true') // true
                features[feat] = 1;
            else if (String(features[feat].toLowerCase()) == 'false') // false
                features[feat] = 0;
            else // if string
                features[feat] = features[feat].trim();
        }
        labelVector.push(features.slice(features.length - 1));
        featureMatrix.push(features.slice(0, features.length - 1));
    }

    return [featureMatrix, labelVector];
}

/**
 * Calculates the accuracy given the algorithm of your choices output 
 * and the known results of the true classifications
 * 
 * @param {[]} classifications classifications received by algorithm
 * @param {[]} classifiedSet already known classifications
 */
function getAccuracy(classifications, classifiedSet) {
    let correct = [0, 0];
    for (let _class = 0; _class < classifications.length; _class++) {
        if (classifications[_class] == classifiedSet[_class])
            correct[0] += 1;
        correct[1] += 1;
    }
   
    if (correct[1] === 0)
        return 0.0;
    else
        return (correct[0] / correct[1]) * 100.0;
}

/**
 * Trains the given algorithm using the provided training set which is equal in 
 * length to the classified set
 * 
 * @param {String} algo algorithm you want to train 
 * @param {[[]]} trainingSet the training set provided as set of vectors, aka 2D array
 * @param {[]} classifiedSet the classified set provided as a single vector of classifications
 * corresponding directly with the same array index as the training set
 */
function train(algo, trainingSet, classifiedSet) {

    if (trainingSet.length === 0) 
        throw new Error("Non-2D array argument 'trainingSet' was provided for 'train(algo, trainingSet, classifiedSet).");
    
    if (trainingSet[0].length === 0 || trainingSet === undefined)
        throw new Error("No argument or empty array 'trainingSet' was provided for 'train(algo, trainingSet, classifiedSet)."); 

    if (classifiedSet.length === 0 || classifiedSet === undefined)
        throw new Error("No argument or empty array 'classifiedSet' was provided for 'train(algo, trainingSet, classifiedSet)."); 

    switch(algo) {
        case 'nb':
            return trainnb(trainingSet, classifiedSet);
        case 'knn':
            return trainknn(trainingSet, classifiedSet);
        case 'p':
            return trainp(trainingSet, classifiedSet);
        default:
            throw new Error("'algo' argument must be specified to run the function 'train(algo, trainingSet, classifiedSet)'.");
    }
}

function trainnb(trainingSet, classifiedSet) {

    let classTable = {};
    // makes class tables
    for (let _class of classifiedSet) {
        if (classTable[_class] === undefined) 
            classTable[_class] = 1;
        else
            classTable[_class] += 1;
    }
    classTable['total'] = classifiedSet.length;

    console.log(classTable);
    let featTable = {};
    // makes feature tables
    for (let v = 0; v < trainingSet.length; v++) {
        for (let feat of trainingSet[v]) {
            if (featTable[feat] === undefined) {
                featTable[feat] = {};
                featTable[feat][classifiedSet[v]] = 1;
            } else {
                
                featTable[feat][classifiedSet[v]] += 1;
            }
        }
    }
    console.log(featTable);

    //nbTrainedSet = [featTables, classTable]; // sets a global trained set variable
}
 
/**
 * Trains the KNN algorithm by setting a global variable of the trained data
 * 
 * @param {[[]]} trainingSet    feature matrix of training data
 * @param {[]} classificationVector class labels vector
 */
function trainknn(trainingSet, classificationVector) {

    for(let i = 0; i < classificationVector.length; i++) {

        for (let f = 0; f < trainingSet[i].length; f++) {
            if (typeof(trainingSet[i][f]) == 'string')
                trainingSet[i][f] = trainingSet[i][f].hashCode();
        }
      
        if (!Object.keys(knnTrainedSet).includes(classificationVector[i][0])) {
            knnTrainedSet[classificationVector[i]] = [];
        }
        knnTrainedSet[classificationVector[i]].push(trainingSet[i]);
    } 
}

function trainp(trainingSet, classifiedSet) {
    
}

/**
 * 
 * @param {[[]]} matrix  feature matrix 
 */
function naiveBayes(matrix, options={}) {

    let data = parseInput(matrix); // parses all input types to numbers and returns feature matrix
    if (!data)
        throw new Error("Parameter 'data' for function 'KNN' could not be parsed to valid data types.");

    if(nbTrainedSet === undefined)
        throw new Error("Running the 'train' function is required to run before the 'KNN' function.");

    let classTable = nbTrainedSet[1];
    let classes = Object.keys(classTable);
    classes.splice(classes.indexOf('total'), 1); // gets rid of total from the keys
    let featTables = nbTrainedSet[0];

    let classifications = []; // all classifications of vectors in test matrix
    let probs = []; // probs of all classes for each vector in test matrix
    for (let v = 0; v < data.length; v++) { // loops through vectors of matrix
        for (let _class of classes) { // loops through classes to get final classification
            let prob = 1;
            for (let f = 0; f < data[v].length; f++) { // loops through features of vector

                let yns = featTables[f][data[v][f]]; // class counts for the particular feature f
              
                if (yns === undefined) {
                    prob *= 0.5;
                  
                } else if (yns[_class] === undefined) {
                    prob *= 1 / classTable[_class];
                    
                } else {
                    prob *= yns[_class] / classTable[_class];
                   
                }
            }
         
            if (probs[v] === undefined)
                probs[v] = {};
            probs[v][_class] = prob;
        } 
        
        // normalization, sums all probabilities of classes to calculate the dnominator
        let denominator = Object.values(probs[v]).reduce((v0, v1) => v0 + v1); 
        Object.keys(probs[v]).forEach(key => probs[v][key] = probs[v][key] / denominator); // calculates the normalized probabilities
        let classified = Object.keys(probs[v])
            .reduce((k0, k1) => probs[v][k0] > probs[v][k1] ? k0 : k1); // gets key (class) with largest value (probability)
        classifications.push(classified);
    }
  
    return classifications;
}

/**
 * Parses the feature matrix for KNN algorithm because it requires data to be 
 * numeric, this function parses strings and booleans to numbers
 * 
 * @param {[[]]} data your feature matrix 
 */
function parseInput(data) {

    let parsedMatrix = [];
    for (let vector of data) {
        for (let feat = 0; feat < vector.length; feat++) {
            if (!isNaN(vector[feat])) // if number
                vector[feat] = Number(vector[feat]);
            else if (vector[feat] === true) // true
                vector[feat] = 1;
            else if (vector[feat] == false) // false
                vector[feat] = 0;
            else // if string
                vector[feat] = vector[feat].hashCode();
        }
        parsedMatrix.push(vector);
    }   
   
    return parsedMatrix;
}

/**
 * 
 * @param {[[]]} data data you wish to be classified
 * @param {{ k: 3 }} options An optional param. k to specify how many neighbors to use
 */
function KNN(matrix, options={'k': 3 }) {
    let data = parseInput(matrix); // parses all input types to numbers and returns feature matrix
    if (!data)
        throw new Error("Parameter 'data' for function 'KNN' could not be parsed to valid data types.");
    
    let classifications = []; // final results
    let kSet = []; // set of k nearest points
    
    if(knnTrainedSet === undefined)
        throw new Error("Running the 'train' function is required to run before the 'KNN' function.");

    for(let d = 0; d < data.length; d++) { // loops all items you want to classify
        for(let key of Object.keys(knnTrainedSet)) { // loops vectors in training set
            for (let vector of knnTrainedSet[key]) {
                for(let feature of vector) { // loops features in training vector
                    // calculates euclidean distance between training vectors
                    // and vectors to be classified
                    let ed = (function euclideanDistance(feat, pt) { 
                        let difs = [feat].map((f, index) => Math.abs(f - pt[index])); 
                        let sqSum = difs.map((d) => Math.pow(d, 2));
                        return Math.sqrt(sqSum);
                    })(feature, data[d]);
            
                    kSet.push([key, feature, ed]);
                }
            }
        }

        // sorts the set by euclidean distance and gets the k closest neighbors of testpt
        kSet.sort((a, b) => a[2] - b[2]);
        kSet = kSet.slice(0, options.k);
    
        // creates the frequency empty dictionary containing { class : frequency }
        let freqSet = {};
        for(let i of kSet)
            freqSet[i[0]] = 0;

        // if only one key exists it returns that class
        if(Object.keys(freqSet) === 1 || Object.keys(freqSet) === options.k)
            return Object.keys(freqSet)[0];
        else {
            for(let i of kSet)
                freqSet[i[0]] += 1;
        }

        // iterates through the frequency dictionary to get the most frequent closest k class
        let keys = Object.keys(freqSet);
        let topClass = keys[0];
        for(let k0 = 0; k0 < keys.length; k0++)
            for(let k1 = k0 + 1; k1 < keys.length; k1++)
                if(freqSet[keys[k0]] >= freqSet[keys[k1]])
                    topClass = keys[k0];
    
        classifications.push(topClass);
    } 

    return classifications;
}

function perceptron(data, options) {
    return options;
}

/**
 * Converts a string to a 32 bit integer hashcode by trimming the string
 * and converting it to lowercase before generating the hash
 */
String.prototype.hashCode = function() { 
    let hash = 0;
    if (this.trim().length === 0) return hash;

    for (let i = 0; i < this.trim().length; i++) {
        let char = this.trim().toLowerCase().charCodeAt(i);
        hash  = ((hash << 5) - hash) + char;
        hash |= 0; 
    }
    return hash;
  }