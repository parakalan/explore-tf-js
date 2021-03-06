var scriptsImported = false;
function train(data) {
    if (!('function' === typeof importScripts)) {
        return;
    }

    if(!scriptsImported) {
        importScripts(
            'lib/tf.min.js',
            'lib/universal-sentence-encoder.js',
            'models/simpleUSEModel.js'
        )
        scriptsImported = true;
    }
    shuffleArray(data)
    shuffleArray(data)
    shuffleArray(data)
    // Split the data into train and val and append val at the end of train. TFJS internally samples from the back.
    data = splitReorderData(data);
    let classesConfig = processClasses(data);
    let model = getSimpleUSEModel({numClasses: classesConfig.uniqueClasses.length});
    model.summary();

    getEmbeddings(data).then(embeddings => {
        fit(model, embeddings, tf.tensor(classesConfig.oneHotClasses));
    })
}

function fit(model, X, y) {
    model.fit(X, y, {
        batchSize: 16,
        epochs: 100,
        callbacks: {onEpochEnd: (epoch, logs) => epochEnd(epoch, logs)},
        validationSplit: 0.1
    }).then(h => { console.log(h); });
}

function epochEnd(epoch, logs) {
    console.log(epoch, logs);
    postMessage(["increment_epoch", {"epoch": epoch+1, "logs": logs}]);
}

function processClasses(data) {
    let allClasses = [];
    let uniqueClasses = [];
    for (var i = 0; i < data.length; i++) {
        allClasses.push(data[i].Class);
        if(!uniqueClasses.includes(data[i].Class)) {
            uniqueClasses.push(data[i].Class);
        }
    }
    let classMap = {};
    for(var i=0; i < uniqueClasses.length; i++) {
        classMap[uniqueClasses[i]] = i;
    }

    let oneHotClasses = []
    let stub = Array(uniqueClasses.length).fill(0);
    for(var i=0; i < allClasses.length; i++) {
        let tmp = stub.slice();
        tmp[classMap[allClasses[i]]] = 1;
        oneHotClasses.push(tmp);
    }
    return {
        "allClasses": allClasses,
        "uniqueClasses": uniqueClasses,
        "classMap": classMap,
        "oneHotClasses": oneHotClasses
    }
}

function getEmbeddings(data) {
    let messages = [];
    for (var i = 0; i < data.length; i++) {
        messages.push(data[i].Message);
    }
    let embeddings = null

    return {
        async then(callback) {
            let model = await use.load();
            let embeddings = []
            for(var i=0; i < messages.length; i = i + 32) {
                let messageBatch = messages.slice(i, i + 32);
                let embeddingsBatch = null;
                try {
                    embeddingsBatch = await model.embed(messageBatch);

                } catch (error) {
                    console.log(messageBatch)
                    console.error(error);
                }
                if(embeddings == null) {
                    embeddings = embeddingsBatch;
                }
                else {
                    embeddings = tf.concat([embeddings, embeddingsBatch], 0)
                }
                console.log(i)
                postMessage(["increment_embeddings", i]);
            }
            postMessage(["increment_embeddings", messages.length]);
            return callback(embeddings);
        }
    }
}

function splitReorderData(data, frac=0.1) {
    let uniqueClassMessages = {};
    for (var i = 0; i < data.length; i++) {
        if (!(data[i].Class in uniqueClassMessages))
            uniqueClassMessages[data[i].Class] = new Array();
        uniqueClassMessages[data[i].Class].push(data[i]);
    }
    let trainingData = [];
    let validationData = [];
    for (const [key, value] of Object.entries(uniqueClassMessages)) {
        let pickCount = Math.round(frac * value.length);
        shuffleArray(value);
        validationData.push(...value.slice(0, pickCount));
        trainingData.push(...value.slice(pickCount, value.length));
    }
    trainingData.push(...validationData);
    return trainingData;
}

function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

onmessage = function(e) {
    train(e.data)
}