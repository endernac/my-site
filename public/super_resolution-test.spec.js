const { test, expect } = require('@playwright/test');
const ort = require('onnxruntime-web');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { downloadFileWithChunking } = require('./utils.js'); // Import the function

// Configuration for ONNX Runtime
const INFERENCE_SESSION_OPTIONS = {
    executionProviders: ['cpu'], // Use CPU for testing
};

// Super Resolution Model Configuration
const superResolutionModel = {
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

        let outputTensor = tf.tensor(outputData, ortOutputTensor.dims);
        outputTensor = outputTensor.squeeze();
        outputTensor = outputTensor.mul(tf.scalar(255)).clipByValue(0, 255);
        outputTensor = tf.round(outputTensor); // Round values to integers

        // Ensure the tensor is 3-dimensional (add a channel dimension for grayscale images)
        if (outputTensor.shape.length === 2) {
        outputTensor = outputTensor.expandDims(-1); // Add channel dimension
        }

        const outputShape = `${outputTensor.shape[1]}x${outputTensor.shape[0]}`;
        return { raw: outputTensor, label: outputShape };
    },
};

test.describe('Super Resolution Model', () => {
    test('should load, preprocess, run inference, and postprocess correctly', async () => {
        // console.log('Testing Super Resolution Model...');

        // Load the model
        await superResolutionModel.load();
        // console.log('Model loaded successfully.');

        // Load a sample image
        const imagePath = path.resolve(__dirname, './test.jpg'); // Replace with a valid image path
        if (!fs.existsSync(imagePath)) {
        throw new Error('Sample image not found. Please provide a valid image at ./test.jpg');
        }
        const imageBuffer = fs.readFileSync(imagePath);
        const inputTensor = tf.node.decodeImage(imageBuffer, 3); // Decode image as RGB tensor
        // console.log('Sample image loaded.');

        // Preprocess the input
        const ortInputs = await superResolutionModel.preprocess(inputTensor);
        // console.log('Input preprocessed.');

        // Run inference
        const ortOutputs = await superResolutionModel.ortSession.run(ortInputs);
        // console.log('Inference completed.');

        // Postprocess the output
        const result = superResolutionModel.postprocess(ortOutputs);
        // console.log('Postprocessing completed.');

        // Validate the results
        // console.log('Output Label (Resolution):', result.label);
        // console.log('Output Tensor Shape:', result.raw.shape);

        // Save the output image
        const outputImagePath = path.resolve(__dirname, './output.jpg');
        const outputImageBuffer = await tf.node.encodeJpeg(result.raw); // Await the Promise
        fs.writeFileSync(outputImagePath, outputImageBuffer);
        // console.log('Output image saved at:', outputImagePath);

        // Assertions
        expect(result.label).toMatch(/^\d+x\d+$/); // Ensure the label is in the format "widthxheight"
        expect(result.raw.shape).toEqual([672, 672, 1]); // Ensure the output tensor has the correct shape
        expect(fs.existsSync(outputImagePath)).toBeTruthy(); // Ensure the output image was saved
    });
});