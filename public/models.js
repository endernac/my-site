// const ort = require('onnxruntime-web');
// const tf = require('@tensorflow/tfjs-node');
// const { downloadFileWithChunking } = require('./utils.js'); // Ensure this is correctly imported
// const ONNXModel = require('./ONNXModel.js'); // Import the ONNXModel class


// Session options for constructing ONNX Runtime Web (ORT) InferenceSession object
// See https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html
const INFERENCE_SESSION_OPTIONS = {
    // NOTE! WebGL doesn't support dynamic input shapes (e.g., variable batch size).
    // Adding "freeDimensionOverrides: { batch_size: 1 }" to the options list doesn't work.
    // See https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html
    // and https://github.com/microsoft/onnxruntime/issues/13909
    executionProviders: [
        'webgpu',
        // 'webgl', // disabled; see above
        'cpu'
    ]
}

const MODELS = [
    // new ONNXModel ({
    //     name: 'VIT Face Expression Classification',
    //     onnxPath: "https://huggingface.co/trpakov/vit-face-expression/resolve/main/onnx/model.onnx",
    //     load: async function(progressCallback) {
    //         this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), INFERENCE_SESSION_OPTIONS);
    //     },
    //     preprocess: async (inputTensor) => {
    //         // derived from https://huggingface.co/trpakov/vit-face-expression/blob/main/onnx/preprocessor_config.json
    //         var processedTensor = inputTensor;

    //         // Resize to 224x224
    //         processedTensor = tf.image.resizeBilinear(processedTensor, [224, 224]);

    //         // Rescale from [0, 255] to [0.0, 1.0]
    //         processedTensor = processedTensor.div(tf.scalar(255));

    //         // Normalize: (x - mean) / std
    //         const mean = tf.tensor([0.5, 0.5, 0.5]);
    //         const std = tf.tensor([0.5, 0.5, 0.5]);
    //         processedTensor = processedTensor.sub(mean).div(std);

    //         // Add outer batch size dimension if necessary
    //         if (processedTensor.shape.length === 3) {
    //             processedTensor = processedTensor.expandDims(0);
    //         }

    //         // Default Tensorflow shape is (batch_size * H * W * C)
    //         // but model uses shape (batch_size * C * H * W)
    //         processedTensor = processedTensor.transpose([0, 3, 1, 2]);

    //         // Convert tf tensor to ort inputs
    //         const ortInputs = {
    //             'pixel_values': new ort.Tensor('float32', new Float32Array(processedTensor.dataSync()), processedTensor.shape)
    //         };
    //         return ortInputs;
    //     },
    //     postprocess: function(ortOutputs) {
    //         const ortOutputTensor = ortOutputs[this.ortSession.outputNames[0]];
    //         const outputRaw = ortOutputTensor.data;

    //         // derived from https://huggingface.co/trpakov/vit-face-expression/blob/main/onnx/config.json
    //         const id2label = {
    //             0: "angry",
    //             1: "disgust",
    //             2: "fear",
    //             3: "happy",
    //             4: "neutral",
    //             5: "sad",
    //             6: "surprise"
    //           };

    //           const logits = tf.tensor(outputRaw);
    //           const probs = tf.softmax(logits);
    //           const predictedIndex = probs.argMax().dataSync()[0];
    //           const predictedLabel = id2label[predictedIndex];
    //           return { raw: logits.dataSync()[predictedIndex], softmaxed: probs, label: predictedLabel };
    //     }
    // }),
    new ONNXModel ({
        name: 'Super Resolution',
        onnxPath: "https://huggingface.co/endernac/onnx-superres/resolve/main/onnx/super_resolution.onnx",
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), INFERENCE_SESSION_OPTIONS);
        },
        preprocess: async (inputTensor) => {
            let processedTensor = tf.image.resizeBilinear(inputTensor, [224, 224]);

            if (processedTensor.shape[processedTensor.shape.length - 1] === 3) {
            processedTensor = tf.image.rgbToGrayscale(processedTensor);
            }

            processedTensor = processedTensor.div(tf.scalar(255));
            processedTensor = processedTensor.expandDims(0);
            processedTensor = processedTensor.transpose([0, 3, 1, 2]); // Convert to [batch, channels, height, width]

            const ortInputs = {
            'input': new ort.Tensor('float32', new Float32Array(processedTensor.dataSync()), processedTensor.shape),
            };
            return ortInputs;
        },
        postprocess: function (ortOutputs) {
            const ortOutputTensor = ortOutputs[this.ortSession.outputNames[0]];
            const outputData = ortOutputTensor.data;

            // Convert output data to a tensor and process it
            let outputTensor = tf.tensor(outputData, ortOutputTensor.dims);
            outputTensor = outputTensor.squeeze();
            outputTensor = outputTensor.mul(tf.scalar(255)).clipByValue(0, 255);
            outputTensor = tf.round(outputTensor); // Round values to integers

            // Convert tensor to uint8 array for JPEG encoding
            const uint8Array = new Uint8Array(outputTensor.dataSync().map(value => Math.round(value)));

            // // Reshape the uint8 array to match the tensor's shape
            // const reshapedTensor = tf.tensor(uint8Array, outputTensor.shape, 'int32');

            // // Use tf.node.encodeJpeg to encode the tensor as a JPEG image
            // const jpegData = tf.node.encodeJpeg(reshapedTensor);

            // Create a blob from the data of type application/octet-stream

            const outputBlob = new Blob([uint8Array], { type: 'application/octet-stream' });

            // Generate a URL for the blob
            const url = URL.createObjectURL(outputBlob);

            // Convert the url to a string
            const urlString = url.toString();

            // Model returns a tensor, but we need to return a number
            // TODO: change the index.js to handle a tensor
            return { raw: 5.555555555, softmaxed: 5.555555555, label: urlString };
        }
    })
]
