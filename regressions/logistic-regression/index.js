require('@tensorflow/tfjs-node');

const plot = require('node-remote-plot');
const LogisticRegression = require('./logistic-regression');
const loadCSV = require('../data/load-csv');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: value =>
            value === 'TRUE' ? 1 : 0
    }
});

const regression = new LogisticRegression(features, labels, {
    learningRate: .5,
    iterations: 30,
    batchSize: 10,
    decisionBoundary: .6
});

regression.train();
console.log(regression.test(testFeatures, testLabels));

plot({
    x: regression.costHistory.reverse()
});