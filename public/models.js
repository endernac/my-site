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
    new ONNXModel ({
        name: 'zeiss_3x3mm_FlowCube_z_img_245x1024x245_v1',
        onnxPath: "https://huggingface.co/endernac/octa_gan/resolve/main/zeiss_3x3mm_FlowCube_z_img_245x1024x245_v1.onnx",
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), INFERENCE_SESSION_OPTIONS);
        },
        preprocess: async (inputTensor) => {
            // let vol = inputTensor.reverse(1).toFloat();
            // // saveTensorBinary(vol, 'original.bin');

            // console.log('Begin normalization');
            // vol = normalize2(vol);
            // // saveTensorBinary(vol, 'normalized.bin');

            // console.log('Begin cropping');
            // vol = await crop_volume(vol, tol=0.75);
            // // saveTensorBinary(vol, 'cropped.bin');

            // console.log('Begin resizing');
            // vol = resize(vol, [128, 128, 128]);
            // // saveTensorBinary(vol, 'resized.bin');

            // console.log('Begin normalization');
            // vol = normalize2(vol);
            // // saveTensorBinary(vol, 'normalized2.bin');
    
            // vol = vol.reshape([1, 1, 128, 128, 128]);

            let vol = inputTensor.reverse(1);
            // saveTensorBinary(vol, 'original.bin');

            console.log('Calculate percentiles');
            const {low, high} = computeUint8PercentilesTensor(vol);
            // console.log(low, high);

            console.log('Begin cropping');
            vol = await crop_volume(vol, low + 0.875 * (high - low));
            // saveTensorBinary(vol, 'cropped.bin');

            console.log('Begin normalization');
            vol = normalize(vol, low, high);
            // saveTensorBinary(vol, 'normalized.bin');

            console.log('Begin resizing');
            vol = resize(vol, [128, 128, 128]);
            // saveTensorBinary(vol, 'resized.bin');

            console.log('Begin normalization');
            vol = normalize2(vol);
            // saveTensorBinary(vol, 'normalized2.bin');
    
            // Reshape to match ONNX input shape: [1, 1, 128, 128, 128]
            vol = vol.reshape([1, 1, 128, 128, 128]);

            console.log('Begin Inference');
            // Convert to ONNX Runtime input format
            const ortInputs = {
                'input': new ort.Tensor('float32', vol.dataSync(), vol.shape),
            };
    
            return ortInputs;
        },
        postprocess: function (ortOutputs) {
            const ortOutputTensor = ortOutputs[this.ortSession.outputNames[0]];
            const outputData = ortOutputTensor.data;
    
            // Extract the scalar output and determine the label
            const score = outputData[0];
            const label = score >= 3.8675 ? "Good" : "Suboptimal";
    
            return { score, label };
        }
    })
]
