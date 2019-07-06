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
    let fileLines = fs.readFileSync(filePath, { encoding: options.encoding })
                        .split('\n');
    

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
            else // if string
                features[feat] = features[feat].trim();
        }
        labelVector.push(features.slice(features.length - 1));
        featureMatrix.push(features.slice(0, features.length - 1));
    }

    return [featureMatrix, labelVector];
}

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

    let featTables = {};
    // makes feature tables
    for (let v = 0; v < trainingSet.length; v++) {
        for (let f = 0; f < trainingSet[v].length; f++) {
            if (featTables[f] === undefined)
                featTables[f] = {};
            else if (featTables[f][trainingSet[v][f]] === undefined) {
                featTables[f][trainingSet[v][f]] = {};
                featTables[f][trainingSet[v][f]][classifiedSet[v]] = 1;
            } else if (featTables[f][trainingSet[v][f][classifiedSet[v]]] === undefined)
                featTables[f][trainingSet[v][f]][classifiedSet[v]] = 1;
            else
                featTables[f][trainingSet[v][f]][classifiedSet[v]] += 1;
        }
    }
    nbTrainedSet = [featTables, classTable]; // sets a global trained set variable
}
 
function trainknn(trainingSet, classificationVector) {

    for(let i = 0; i < classificationVector.length; i++) {
        let FLAG = 0;
        for(let key of Object.keys(knnTrainedSet)) {
           if (classificationVector[i] === key) {
               FLAG = 1;
               break;
           }
        }

        if (!FLAG)
            knnTrainedSet[classificationVector[i]] = [];
        knnTrainedSet[classificationVector[i]].push(trainingSet[i]);
    } 
}

function trainp(trainingSet, classifiedSet) {
    
}

/**
 * 
 * @param {[[]]} data  feature matrix 
 */
function naiveBayes(data, options={}) {

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
              
                if (yns === undefined)
                    prob *= 0.5;
                else if (yns[_class] === undefined)
                    prob *= 1 / classTable[_class];
                else
                    prob *= yns[_class] / classTable[_class];
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
 * 
 * @param {[[]]} data data you wish to be classified
 * @param {{ k: 3 }} options An optional param. k to specify how many neighbors to use
 */
function KNN(data, options={'k': 3 }) {
    let classifications = []; // final results
    let kSet = []; // set of k nearest points
    
    if(knnTrainedSet === undefined)
        throw new Error("Running the 'train' function is required to run before the 'KNN' function.");

    for(let d = 0; d < data.length; d++) { // loops all items you want to classify

        for(let key = 0; key < knnTrainedSet.length; key++) { // loops vectors in training set
            for(let feature of knnTrainedSet[key]) { // loops features in training vector
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

        // sorts the set by euclidean distance and gets the k closest neighbors of testpt
        kSet.sort((a, b) => a[2] - b[2]);
        kSet = kSet.slice(0, options.k);
    
        // creates the frequency dictionary containing { class : frequency }
        let freqSet = {};
        for(let i of kSet) {
            freqSet[i[0]] = 0;
        }

        // if only one key exists it returns that class
        if(Object.keys(freqSet) === 1 || Object.keys(freqSet) === options.k)
            return Object.keys(freqSet)[0];
        else {
            for(let i of kSet) {
                freqSet[i[0]] += 1;
            }
        }

        // iterates through the frequency dictionary to get the most frequent closest k class
        let topClass = undefined;
        for(let k0 of Object.keys(freqSet)) {
            for(let k1 of Object.keys(freqSet)) {
                if(k0 >= k1)
                    topClass = k0;
            }
        }

        classifications.push(topClass);
    }

    return classifications;
}

function perceptron(data, options) {
    return options;
}