function getSimpleUSEModel(config) {
	const model = tf.sequential();

	model.add(tf.layers.dense({
		inputShape: 512,
		units: 256,
		kernelInitializer: 'varianceScaling',
		activation: 'relu'
	}));

	model.add(tf.layers.dense({
		units: 128,
		kernelInitializer: 'varianceScaling',
		activation: 'softmax'
	}));

	model.add(tf.layers.dropout(0.2))

	model.add(tf.layers.dense({
		units: config.numClasses,
		kernelInitializer: 'varianceScaling',
		activation: 'softmax'
	}));

	const optimizer = tf.train.adam();
	model.compile({
	optimizer: optimizer,
	loss: 'categoricalCrossentropy',
	metrics: ['accuracy'],
	});

	return model;
}