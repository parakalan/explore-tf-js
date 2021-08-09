class AttentionLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  build(inputShape) {
    let inputDim = inputShape[inputShape.length - 1];
    this.Uw = tf.layers.dense({
       units: inputDim,
       kernelInitializer: 'glorotUniform',
       useBias: false,
       trainable: true
    })
  }

  call(input) {
    let mult = tf.exp(tf.dot(input, this.Uw))
    let output = mult / (tf.sum(mult, 1) + 0.0000000000001)
    return tf.reshape(output, [output.shape[0], output.shape[1], 1])
  }

  getConfig() {
    const config = super.getConfig();
    Object.assign(config, {alpha: this.alpha});
    return config;
  }

  static get className() {
    return 'AttentionLayer';
  }
}