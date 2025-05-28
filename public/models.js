// Session options for constructing ONNX Runtime Web (ORT) InferenceSession object
// See https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html
const INFERENCE_SESSION_OPTIONS = {
    executionProviders: [
    //     {
    //         name: 'webnn',
    //         deviceType: 'gpu',
    //         powerPreference: "default", // corrected powerPreference value
    //     },
    //   'webgpu',  // still try WebGPU first
      'cpu'      // fallback to CPU if something fails
    ]
};

const MODELS = [
    new ONNXModel ({
        name: 'OCTAGAN',
        onnxPath: "https://huggingface.co/endernac/octa_gan/resolve/main/zeiss_3x3mm_FlowCube_z_img_245x1024x245_v1.onnx",
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), INFERENCE_SESSION_OPTIONS);
        },
        preprocess: async (inputTensor) => {
            // Step 1: Reverse along axis 1
            const vol = tf.tidy(() => inputTensor.reverse(1));

            try {
                // Step 2: Compute percentiles
                console.log('Calculate percentiles');
                const { low, high } = tf.tidy(() => computeUintPercentilesTensor(vol));

                // Step 3: Crop volume (asynchronous)
                console.log('Begin cropping');
                const cropped = await crop_volume(vol, low + 0.875 * (high - low));
                vol.dispose(); // Dispose of 'vol' after cropping

                // Step 4: Normalize
                console.log('Begin normalization');
                const normalized = tf.tidy(() => normalize(cropped, low, high));
                cropped.dispose(); // Dispose of 'cropped' after normalization

                // Step 5: Resize
                console.log('Begin resizing');
                const resized = tf.tidy(() => resize(normalized, [128, 128, 128]));
                normalized.dispose(); // Dispose of 'normalized' after resizing

                // Step 6: Further normalization
                console.log('Begin normalization2');
                const normalized2 = tf.tidy(() => normalize2(resized));
                resized.dispose(); // Dispose of 'resized' after second normalization

                // Step 7: Reshape for ONNX input
                console.log('Reshaping for ONNX input');
                const reshaped = tf.tidy(() => normalized2.reshape([1, 1, 128, 128, 128]));
                normalized2.dispose(); // Dispose of 'normalized2' after reshaping

                // Step 8: Convert to ONNX Runtime input format
                console.log('Converting to ONNX Runtime input format');
                const input = new ort.Tensor('float32', reshaped.dataSync(), reshaped.shape);
                reshaped.dispose(); // Dispose of 'reshaped' after data extraction
                
                return { input };
            } catch (error) {
                // Dispose of any tensors that may have been created before the error
                tf.dispose([vol]);
                throw error;
            }
        },
        postprocess: function (ortOutputs) {
            const ortOutputTensor = ortOutputs[this.ortSession.outputNames[0]];
            try {
                const outputData = ortOutputTensor.data;
        
                // Extract the scalar output and determine the label
                const score = outputData[0];
                const label = score >= 3.8675 ? "Excellent/Good" : "Suboptimal";
                ortOutputTensor.dispose();

                return { score, label };
            }
            finally {
                ortOutputTensor.dispose(); // Dispose of the tensor to free memory
            }
        }
    })
]
