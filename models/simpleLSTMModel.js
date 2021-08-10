function getSimpleLSTMModel(config) {
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
	// // The MaxPooling layer acts as a sort of downsampling using max values
	// // in a region instead of averaging.
	// model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

	// // Repeat another conv2d + maxPooling stack.
	// // Note that we have more filters in the convolution.
	// model.add(tf.layers.conv2d({
	// kernelSize: 5,
	// filters: 16,
	// strides: 1,
	// activation: 'relu',
	// kernelInitializer: 'varianceScaling'
	// }));
	// model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

	// // Now we flatten the output from the 2D filters into a 1D vector to prepare
	// // it for input into our last layer. This is common practice when feeding
	// // higher dimensional data to a final classification output layer.
	// model.add(tf.layers.flatten());

	// // Our last layer is a dense layer which has 10 output units, one for each
	// // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
	// const NUM_OUTPUT_CLASSES = 10;
	// model.add(tf.layers.dense({
	// units: NUM_OUTPUT_CLASSES,
	// kernelInitializer: 'varianceScaling',
	// activation: 'softmax'
	// }));


	// // Choose an optimizer, loss function and accuracy metric,
	// // then compile and return the model
	const optimizer = tf.train.adam();
	model.compile({
	optimizer: optimizer,
	loss: 'categoricalCrossentropy',
	metrics: ['accuracy'],
	});

	return model;
}