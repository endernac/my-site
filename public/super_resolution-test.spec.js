const { test, expect } = require('@playwright/test');
const ort = require('onnxruntime-web');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { MODELS } = require('./models.js');
const { saveTensorToJSONNode } = require('./utils.js');

// Get the Super Resolution model from the imported models
const superResolutionModel = MODELS.find(model => model.name === 'Super Resolution');

// Configuration for ONNX Runtime
const INFERENCE_SESSION_OPTIONS = {
    executionProviders: ['cpu'], // Use CPU for testing
};

test.describe('Super Resolution Model', () => {
    let inputTensor;
    let ortInputs;
    let ortOutputs;
    let result;

    test.beforeAll(async () => {
        // Load the model
        await superResolutionModel.load();
        console.log('Model loaded successfully.');

        // Load a sample image
        const imagePath = path.resolve(__dirname, './test.jpg');
        if (!fs.existsSync(imagePath)) {
            throw new Error('Sample image not found. Please provide a valid image at ./test.jpg');
        }
        const imageBuffer = fs.readFileSync(imagePath);
        inputTensor = tf.node.decodeImage(imageBuffer, 3);
        console.log('Sample image loaded.');
    });

    test('should load the model successfully', async () => {
        expect(superResolutionModel.ortSession).toBeDefined();
        expect(superResolutionModel.ortSession.outputNames).toBeDefined();
    });

    test('should preprocess the input image correctly', async () => {
        ortInputs = await superResolutionModel.preprocess(inputTensor);
        
        expect(ortInputs).toBeDefined();
        expect(ortInputs.input).toBeDefined();
        expect(ortInputs.input.dims).toEqual([1, 1, 224, 224]); // Check expected dimensions
    });

    test('should run inference successfully', async () => {
        ortOutputs = await superResolutionModel.ortSession.run(ortInputs);
        
        expect(ortOutputs).toBeDefined();
        expect(ortOutputs[superResolutionModel.ortSession.outputNames[0]]).toBeDefined();
    });

    test('should postprocess the output correctly', async () => {
        result = superResolutionModel.postprocess(ortOutputs);
        
        expect(result).toBeDefined();
        // The model should output a tensor with shape [672, 672] for grayscale
        // We'll expand it to 3D for image saving
        expect(result.shape.length).toBe(2);
        expect(result.shape[0]).toBe(672);
        expect(result.shape[1]).toBe(672);

        // Expand dimensions to make it compatible with image saving
        result = result.expandDims(-1); // Add channel dimension
    });

    // TODO: Should save the output tensor to a file using utils.js
    test('should save the output tensor to a file using utils.js', async () => {
        const outputImagePath = path.resolve(__dirname, './output.json');
        await saveTensorToJSONNode(result, outputImagePath);
        
        // Verify the file was created
        expect(fs.existsSync(outputImagePath)).toBeTruthy();
        console.log('Output tensor saved at:', outputImagePath);
    });

    test.afterAll(async () => {
        // Cleanup
        if (inputTensor) {
            inputTensor.dispose();
        }
        if (result) {
            result.dispose();
        }
    });
});