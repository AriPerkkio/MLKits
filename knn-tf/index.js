require('@tensorflow/tfjs-node');

const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

const sortTensors = (a, b) => a.get(0) > b.get(0) ? 1 : -1;
const reduceLabels = (acc, tensor) => acc + tensor.get(1);

function knn(features, labels, predictionPoint, k) {
    const { mean, variance } = tf.moments(features, 0);

    const scaledPredictionPoint = predictionPoint
        .sub(mean)
        .div(variance.pow(.5));

    return features
        .sub(mean)
        .div(variance.pow(.5))
        .sub(scaledPredictionPoint)
        .pow(2)
        .sum(1)
        .pow(.5)
        .expandDims(1)
        .concat(labels, 1)
        .unstack()
        .sort(sortTensors)
        .slice(0, k)
        .reduce(reduceLabels, 0) / k;
}

let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
    labelColumns: ['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);

// Clear tensorflow's warning logs
console.clear();

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];

    console.log(
        'Error', err * 100,
        '\nGuess ', result,
        '\nActual ', testLabels[i][0], '\n'
    );
});