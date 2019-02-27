require('@tensorflow/tfjs-node');

const LinearRegression = require('./linear-regression');
const loadCSV = require('./load-csv');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 10,
    batchSize: 10,
});

regression.train();

console.log(
    `\nUpdated M is ${regression.weights.get(1, 0)}`,
    `\nUpdated B is ${regression.weights.get(0, 0)}`
);

const r2 = regression.test(testFeatures, testLabels);
console.log(`R2 = ${r2}`);

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
});

regression.predict([
    [120, 2, 380],
    [135, 2.1, 420]
]).print();