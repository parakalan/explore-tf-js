class USELayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  build(inputShape) {
    console.log(inputShape)

  }

  call(input) {
      useModel.embed(sentences).then(embeddings => {
        return embeddings;
      });
  }

  static get className() {
    return 'USELayer';
  }
}