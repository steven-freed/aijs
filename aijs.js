var fs = require('fs');
 
module.exports.Perceptron = function() {

    let weights = [];
    let bias = 0;
    let classes = {
        0: undefined,
        1: undefined
    };

    function f(features) {
    
        if (features.length !== weights.length) return undefined;

        let score = 0;
        for (let i = 0; i < weights.length; i++) {
            score += weights[i] * features[i];
        }
        score += bias;

        return score > 0 ? 1 : 0;
    }

    /**
     * 
     * @param {*} featMatrix 
     * @param {*} labelVector 
     * @param {*} options 
     */
    function train(featMatrix, labelVector, options={ epochs: 100 }) {

        classes[1] = labelVector[0];

        while (options.epochs > 0) {

            for (let i = 0; i < featMatrix.length; i++) {
                let label = undefined;

                if (labelVector[i] === classes[1]) {
                    label = 1;
                } else {
                    label = 0;
                    classes[0] = labelVector[i];
                }

                if (featMatrix[i].length !== weights.length) {
                    weights = featMatrix[i];
                    bias = 1;
                }
        
                let classification = f(featMatrix[i]);
                if (classification !== label) {
                    let error = label - classification;
                    for (let j = 0; j < weights.length; j++) {
                        weights[j] += error * featMatrix[i][j];
                    }
                    bias += error;
                }
            }  
            options.epochs -= 1;
        }

    }

    function classify(featMatrix) {
        let classifications = [];
        for (let i = 0; i < featMatrix.length; i++) {
            let res = f(featMatrix[i]);
            if (res) {
                classifications.push(classes[1]);
            } else {
                classifications.push(classes[0]);
            }
        }
        return classifications;
    }
    
    return {
        'train': train,
        'classify': classify
    };
};

/**
 * 
 * @param {[[]]} matrix  feature matrix 
 */
module.exports.NaiveBayes = function() {

    let nbTrainedSet = undefined;

    function train(featMatrix, labelVector) {

        let classTable = {};
        // makes class tables
        for (let _class of labelVector) {
            if (classTable[_class] === undefined) 
                classTable[_class] = 1;
            else
                classTable[_class] += 1;
        }
        classTable['total'] = labelVector.length;
    
        let featTable = {};
        // makes feature tables
        for (let v = 0; v < featMatrix.length; v++) {
            for (let feat of featMatrix[v]) {
                if (featTable[feat] === undefined) {
                    featTable[feat] = {};
                    featTable[feat][labelVector[v]] = 1;
                } else {
                    
                    featTable[feat][labelVector[v]] += 1;
                }
            }
        }
    
        nbTrainedSet = [featTable, classTable]; // sets a global trained set variable
    }

    function classify(featMatrix) {
        
        let data = parseInput(featMatrix); // parses all input types to numbers and returns feature matrix

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
                    if (featTables[f] == undefined) {
                        prob *= 0.5;
                        continue;
                    }

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

    return {
        'train': train,
        'classify': classify
    };
};

module.exports.KNN = function() {

    let knnTrainedSet = {};

    /**
     * Trains the KNN algorithm by setting a global variable of the trained data
     * 
     * @param {[[]]} trainingSet    feature matrix of training data
     * @param {[]} classificationVector class labels vector
     */
    function train(featMatrix, labelVector) {

        for(let i = 0; i < labelVector.length; i++) {

            for (let f = 0; f < featMatrix[i].length; f++) {
                if (typeof(featMatrix[i][f]) == 'string')
                    featMatrix[i][f] = featMatrix[i][f].hashCode();
            }
        
            if (!Object.keys(knnTrainedSet).includes(labelVector[i])) {
                knnTrainedSet[labelVector[i]] = [];
            }
            knnTrainedSet[labelVector[i]].push(featMatrix[i]);
        } 
    }
    
    /**
     * 
     * @param {[[]]} matrix data you wish to be classified
     * @param {{ k: 3 }} options An optional param. k to specify how many neighbors to use
     */
    function classify(featMatrix, options={k: 3 }) {
        let data = parseInput(featMatrix); // parses all input types to numbers and returns feature matrix
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

    return {
        'train': train,
        'classify': classify
    };
};

///////// Util Functions ///////////

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
module.exports.readFile = function (filePath, options={ skipHeader: true, delimiter: ',', encoding: 'utf8' }) {
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
        labelVector.push(features.slice(features.length - 1)[0]);
        featureMatrix.push(features.slice(0, features.length - 1));
    }

    return {
        'featMatrix': featureMatrix, 
        'labelVector': labelVector
    };
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
 * Calculates the accuracy given the algorithm of your choices output 
 * and the known results of the true classifications
 * 
 * @param {[]} classifications classifications received by algorithm
 * @param {[]} classifiedSet already known classifications
 */
module.exports.getAccuracy = function (classifications, classifiedSet) {
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