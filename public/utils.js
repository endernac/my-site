/**
 * A callback function to be called during progression of an iterative process, e.g., downloading a file
 * @callback progressCallback
 * @param {number} progressFraction - decimal value representing completion progress, e.g., 0.6 for 60%
 */

/**
 *
 * @param {string} fetchURL - URL of file to be downloaded using fetch()
 * @param {progressCallback} [progressCallback] - A callback function to be called during the download loop
 * @returns {Promise<Uint8Array>} data from response body, stored as array buffer
 */
async function downloadFileWithChunking(fetchURL, progressCallback=undefined) {
    const response = await fetch(fetchURL);
    const contentLength = response.headers.get("Content-Length");
    const totalSize = contentLength ? parseInt(contentLength, 10) : 0;

    if (totalSize) {
        const reader = response.body.getReader();
        const buffer = new Uint8Array(totalSize);
        let offset = 0;
        let loadedSize = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer.set(value, offset);
            offset += value.length;
            loadedSize += value.length;

            if (progressCallback) {
                progressCallback(loadedSize / totalSize);
            }
        }
        return buffer;
    } else {
        console.warn("Content-Length header is missing. Downloading without chunking.");
        const arrayBuffer = await response.arrayBuffer();
        return new Uint8Array(arrayBuffer);
    }
}

/**
 * Checks whether an ONNXModel object's underlying onnx file exists in the browser’s cache.
 * @param {ONNXModel} model - an ONNXModel object
 * @returns {Promise<boolean>} whether model exists in browser's cache
 */
async function modelIsCached(model) {
    return await caches.match(new Request(model.onnxPath)) ? true : false;
}

/**
 * Deletes all data from the browser cache
 */
async function clearCaches() {
    await caches.keys().then(cacheNames => {
        return Promise.all(
            cacheNames.map(cacheName => {
                return caches.delete(cacheName);
            })
        );
    }).then(function() {
        console.log('All caches cleared');
    }).catch(function(error) {
        console.error('Error clearing caches:', error);
    });
}

/**
 * Converts a JS file object containing TIFF data to a data URL
 * @param {File} file - JS file object containing TIFF data
 * @returns {Promise<string>} data URL, e.g., for use as the src of an img element
 */
function tiffFileToDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function (e) {
            const tiffData = e.target.result;
            const tiff = new Tiff({ buffer: tiffData });
            resolve(tiff.toCanvas().toDataURL());
        };
        reader.onerror = reject;
        reader.readAsArrayBuffer(file);
    });
}

/**
 * Converts a TensorFlow tensor into a JSON object for download;
 * e.g., useful for saving JS-preprocessed images to disk for comparison with
 * original preprocessing implementation. JSON is used to avoid introducing
 * changes to pixel values due to compression/decompression/etc., but there are
 * probably more elegant solutions than this.
 * @param {tf.tensor} tensor - TensorFlow tensor to be converted
 */
async function saveTensorToJSON(tensor) {
    const tensorArray = await tensor.array();
    const tensorJSON = JSON.stringify(tensorArray);
    const blob = new Blob([tensorJSON], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tensorData.json';
    a.click();
    URL.revokeObjectURL(url);
}

async function saveTensorBinary(tensor, fileName = 'tensorData.bin') {
    const flat = await tensor.data();             // Promise<Float32Array>
    const buffer = flat.buffer;                   // ArrayBuffer
    const blob = new Blob([buffer], {             // binary blob
      type: 'application/octet-stream'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(url);
  }
  

/**
 * Helper function for visualizing a TensorFlow tensor by converting it into a
 * dataURL and then setting the src of the passed img element. Scales pixel
 * values to range [0, 255] based on maximum (but NOT minimum) of tensor.
 * @param {tf.tensor} imgTensor - TensorFlow tensor to be visualized
 * @param {HTMLImageElement} imageEl - img element for visualizing the tensor
 */
async function visualizeImgTensor(imgTensor, imageEl) {
    var dispTensor = imgTensor.squeeze();  // Remove batch dimension, shape: [224, 224, 3]

    // Scale pixel values to [0, 255]
    dispTensor = imgTensor.mul(tf.scalar(dispTensor.max().dataSync()[0] <= 1.0 ? 255.0 : 1.0)).clipByValue(0, 255).cast('int32');

    const canvas = document.createElement('canvas');
    await tf.browser.toPixels(dispTensor, canvas);
    imageEl.src = canvas.toDataURL();
}

/**
 * Converts a JS File object containing image data into a TensorFlow tensor
 * @param {File} file - JS File object to be converted
 * @returns {tf.tensor} TensorFlow tensor containing image data
 */
function imageFileToTensor(file) {
    const reader = new FileReader();
    const imgElement = new Image();
    return new Promise(async (resolve, reject) => {
        if (file.name.endsWith(".json")) {
            const reader = new FileReader();

            // Read JSON file as text
            reader.onload = function (e) {
                try {
                    // Parse JSON data
                    const jsonData = JSON.parse(e.target.result);

                    // Check if JSON data is a valid 3D array (HxWx3)
                    if (!Array.isArray(jsonData) || !Array.isArray(jsonData[0]) || !Array.isArray(jsonData[0][0])) {
                        console.error("Invalid JSON structure");
                        return;
                    }

                    // Convert to flattened 1D tf tensor
                    const flatData = jsonData.flat(2);
                    const tensor = tf.tensor(flatData); // Convert to tf tensor

                    // Reshape tensor to tf shape (height, width, 3)
                    const height = jsonData.length;
                    const width = jsonData[0].length;
                    const reshapedTensor = tensor.reshape([height, width, 3]);
                    // const reshapedTensor = tensor.reshape([245, 1024, 245]);

                    // Print the tensor or use it for further processing
                    reshapedTensor.print();  // This will print the tensor to the console

                    resolve(reshapedTensor);
                } catch (error) {
                    console.error("Error reading or parsing JSON:", error);
                }
            };

            // Read JSON file as text
            reader.readAsText(file);
        } else if (file.type === "image/tiff") {
            imgElement.src = await tiffFileToDataURL(file);
            imgElement.onload = () => resolve(tf.browser.fromPixels(imgElement).toFloat());
            imgElement.onerror = reject;
        } else {
            reader.onload = e => {
                imgElement.src = e.target.result;
                imgElement.onload = () => resolve(tf.browser.fromPixels(imgElement).toFloat());
                imgElement.onerror = reject;
            };
            reader.readAsDataURL(file);
        }
    });
}

/**
 * Converts a .img file into a TensorFlow.js tensor.
 * @param {File} file - The .img file to be converted.
 * @param {Array<number>} volDim - The dimensions of the volume (e.g., [245, 1024, 245]).
 * @returns {Promise<tf.tensor>} A tf.tensor with the reshaped data.
 */
async function imgFileToTensor(file, volDim) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = function (event) {
            try {
                // Read the file as an ArrayBuffer
                const arrayBuffer = event.target.result;

                // Convert the ArrayBuffer to a Uint8Array
                const uint8Array = new Uint8Array(arrayBuffer);

                // Check if the size matches the expected volume dimensions
                const expectedSize = volDim.reduce((a, b) => a * b, 1);
                if (uint8Array.length !== expectedSize) {
                    throw new Error(
                        `File size (${uint8Array.length}) does not match expected dimensions (${volDim.join('x')}).`
                    );
                }

                // Convert the Uint8Array to a uint8Array
                const arr = new Uint8Array(arrayBuffer);

                // // Create a uint8 TensorFlow.js tensor and reshape it
                const tensor = tf.tensor(arr, volDim);

                resolve(tensor);
                // resolve(arr);
            } catch (error) {
                reject(error);
            }
        };

        reader.onerror = function (error) {
            reject(error);
        };

        // Read the file as an ArrayBuffer
        reader.readAsArrayBuffer(file);
    });
}

/**
 * Converts an HTML Table Element to comma-separated string and initiates download of resulting .csv file
 * @param {HTMLTableElement} tableEl - HTML Table Element to be converted
 */
function htmlTableToCSV(tableEl) {
    if (!tableEl) {
        console.warn(`Cannot convert invalid table element ${tableEl} to CSV!`);
        return;
    }
    const rows = Array.from(tableEl.querySelectorAll('tr'));
    const csv = rows.map(row =>
      Array.from(row.cells).map(cell => `"${cell.innerText.replace('"', '""')}"`).join(',')
    ).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'output.csv';
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Helper function for forcefully ensuring that a Bootstrap Modal hides/closes
 * @param {bootstrap.Modal} bsModal - Bootstrap modal object to be hidden/closed
 */
function forceHideBSModal(bsModal) {
    // Remove from DOM upon being hidden
    bsModal._element.addEventListener('hidden.bs.modal', function () {
        bsModal._element.remove();
    });
    bsModal.hide();
    // in case we're too fast and hide is called during the show animation
    bsModal._element.addEventListener('shown.bs.modal', function () {
        bsModal.hide();
    });
}

/**
 * Creates a modal div (for use with Bootstrap) and appends it to the document body
 * @param {string} modalContentHTML - string of HTML content to be placed within ".modal-content" div
 * @returns {HTMLDivElement} modal div element for use with Bootstrap
 */
function makeModalElement(modalContentHTML) {
    var modalElement = document.createElement('div');
    modalElement.id = "myModal";
    modalElement.classList.add('modal', 'fade');
    modalElement.innerHTML = `
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
            ${modalContentHTML}
        </div>
    </div>
    `;
    document.body.appendChild(modalElement);
    return modalElement;
}


/**
 * Normalize using percentile-based rescaling
 * @param {tensor} vol - Tensor to be normalized
 * @returns {tensor} Normalized tensor
 */
const normalize = (vol, lowerPercentile, upperPercentile) => {
    // check that the tensor is of int32 type
    if (vol.dtype !== 'int32') {
        throw new Error("Tensor must be of dtype 'int32'");
    }
    // check that the tensor is 3D
    if (vol.rank !== 3) {
        throw new Error("Input tensor must be 3D");
    }

    // create a volume normalized to the range [-1, 1]
    const normalized = vol.clipByValue(lowerPercentile, upperPercentile)
        .sub(lowerPercentile)
        .div(upperPercentile - lowerPercentile)
        .mul(2)
        .sub(1);
    
    // convert tensor to float32 and return
    return normalized.cast('float32');
};

/**
 * Normalize using percentile-based rescaling
 * @param {tensor} vol - Tensor to be normalized
 * @returns {tensor} Normalized tensor
 */
const normalize2 = (vol) => {
    // check that the tensor is of float32 type
    if (vol.dtype !== 'float32') {
        throw new Error("Tensor must be of dtype 'float32'");
    }
    // check that the tensor is 3D
    if (vol.rank !== 3) {
        throw new Error("Input tensor must be 3D");
    }

    const flatten = vol.flatten();
    const sorted = flatten.arraySync().sort((a, b) => a - b);
    const lowerPercentile = sorted[Math.floor(0.005 * sorted.length)];
    const upperPercentile = sorted[Math.floor(0.995 * sorted.length)];

    return vol.clipByValue(lowerPercentile, upperPercentile)
        .sub(lowerPercentile)
        .div(upperPercentile - lowerPercentile)
        .mul(2)
        .sub(1);
};

// /**
//  * Normalize a Uint8Array using percentile-based rescaling
//  * @param {Uint8Array} vol - 3D volume data
//  * @param {number} lowerPercentile - Lower percentile threshold (0-255)
//  * @param {number} upperPercentile - Upper percentile threshold (0-255)
//  * @returns {Float32Array} Normalized Float32 volume in range [-1, 1]
//  */
// function normalize(vol, lowerPercentile, upperPercentile) {
//     if (!(vol instanceof Uint8Array)) {
//         throw new Error("Input must be a Uint8Array");
//     }

//     if (lowerPercentile >= upperPercentile) {
//         throw new Error("Lower percentile must be less than upper percentile");
//     }

//     const range = upperPercentile - lowerPercentile;
//     const normalized = new Float32Array(vol.length);

//     for (let i = 0; i < vol.length; i++) {
//         // Clip to percentile range
//         let val = vol[i];
//         if (val < lowerPercentile) val = lowerPercentile;
//         if (val > upperPercentile) val = upperPercentile;

//         // Normalize to [0, 1]
//         let norm = (val - lowerPercentile) / range;

//         // Scale to [-1, 1]
//         norm = norm * 2 - 1;

//         normalized[i] = norm;
//     }

//     return normalized;
// }

// /**
//  * Normalize a Float32Array using percentile-based rescaling
//  * @param {Float32Array} vol - Flat 3D volume data
//  * @param {number} lowerPercentile - Lower percentile threshold (0-255)
//  * @param {number} upperPercentile - Upper percentile threshold (0-255)
//  * @returns {Float32Array} Normalized Float32 volume in range [-1, 1]
//  */
// function normalize(vol, lowerPercentile, upperPercentile) {
//     if (!(vol instanceof Float32Array)) {
//         throw new Error("Input must be a Float32Array");
//     }

//     if (lowerPercentile >= upperPercentile) {
//         throw new Error("Lower percentile must be less than upper percentile");
//     }

//     const range = upperPercentile - lowerPercentile;
//     const normalized = new Float32Array(vol.length);

//     for (let i = 0; i < vol.length; i++) {
//         // Clip to percentile range
//         let val = vol[i];
//         if (val < lowerPercentile) val = lowerPercentile;
//         if (val > upperPercentile) val = upperPercentile;

//         // Normalize to [0, 1]
//         let norm = (val - lowerPercentile) / range;

//         // Scale to [-1, 1]
//         norm = norm * 2 - 1;

//         normalized[i] = norm;
//     }

//     return normalized;
// }

/**
 * Compute percentiles from a 3D tf.Tensor of uint8 values.
 *
 * @param {tf.Tensor3D} vol - A 3D Tensor of dtype 'uint8' or 'int32'.
 * @param {number} pLow        - Lower percentile (e.g. 0.005).
 * @param {number} pHigh       - Upper percentile (e.g. 0.995).
 * @returns {{ low: number, high: number }} - Percentile gray-levels.
 */
function computeUint8PercentilesTensor(vol, pLow = 0.005, pHigh = 0.995) {
    // check that the tensor is of int32 type
    if (vol.dtype !== 'int32') {
        throw new Error("Tensor must be of dtype 'int32'");
    }
    if (vol.rank !== 3) {
        throw new Error("Input tensor must be 3D");
    }
  
    const hist = new BigUint64Array(256);
    const data = vol.dataSync(); // Uint8Array or Int32Array (no copy for uint8)
    const total = data.length;
  
    // 1) Histogram
    for (let i = 0; i < total; i++) {
        const val = data[i];
        hist[val]++;
    }
  
    // 2) Cumulative histogram
    const cdf = new BigUint64Array(256);
    let cumulative = BigInt(0);
    for (let v = 0; v < 256; v++) {
        cumulative += hist[v];
        cdf[v] = cumulative;
    }
  
    // 3) Targets
    const targetLow  = Math.floor(pLow  * total);
    const targetHigh = Math.floor(pHigh * total);
  
    // 4) Percentile values
    let low = 0, high = 255;
    for (let v = 0; v < 256; v++) {
        if (cdf[v] >= targetLow) {
            low = v;
            break;
        }
    }
    for (let v = 0; v < 256; v++) {
        if (cdf[v] >= targetHigh) {
            high = v;
            break;
        }
    }
  
    return { low, high };
};

/**
 * Crop the volume to the region of interest
 * @param {tensor} vol - Tensor to be cropped
 * @returns {tensor} Cropped tensor
 */
const crop_volume = async (vol, tol = 0.75) => {
    // check that the tensor is of int32 type
    if (vol.dtype !== 'int32' && vol.dtype !== 'float32') {
        throw new Error("Tensor must be of dtype 'int32' or 'float32'");
    }
    if (vol.rank !== 3) {
        throw new Error("Input tensor must be 3D");
    }

    // get the mask of voxel above the threshold
    const mask = vol.greater(tf.scalar(tol));

    // get the coordinates of the non-zero elements using tensorflow.js
    const coords = await tf.whereAsync(mask);

    if (coords.size === 0) {
        throw new Error('No valid region found in the tensor for cropping.');
    }

    // get x, y, z dimensions
    const [x_max, y_max, z_max] = vol.shape; 
    const [x0, y0, z0] = coords.min(0).arraySync();
    const [x1, y1, z1] = coords.max(0).arraySync().map((v) => v + 1);

    // fix to make syntax correct
    return vol.slice([x0, y0, z0], [x1 - x0, y1 - y0, z1 - z0]); 
};

/**
 * 3D trilinear interpolation of a tf.Tensor3D, align_corners=false.
 *
 * @param {tf.Tensor3D} vol    Input volume tensor of shape [D_in, H_in, W_in]
 * @param {[number,number,number]} size  Desired output shape [D_out, H_out, W_out]
 * @returns {tf.Tensor3D}       Interpolated tensor [D_out, H_out, W_out]
 */
function resize(vol, size) {
    if (vol.dtype !== 'float32') {
        throw new Error("Tensor must be of dtype 'float32'");
    }
    if (vol.rank !== 3) {
        throw new Error("Input tensor must be 3D");
    }
    
    // 1. Read input shape and flat data
    const [D_in, H_in, W_in] = vol.shape;                         // tf.Tensor3D.shape :contentReference[oaicite:3]{index=3}
    const inputData = vol.dataSync();                             // Float32Array length D_in*H_in*W_in :contentReference[oaicite:4]{index=4}

    // 2. Prepare output dims and buffer
    const [D_out, H_out, W_out] = size;
    const outData = new Float32Array(D_out * H_out * W_out);

    // 3. Compute scales (align_corners=false): old_dim / new_dim
    const scaleD = D_in / D_out;
    const scaleH = H_in / H_out;
    const scaleW = W_in / W_out;

    // 4. Helper to fetch with edge padding
    function getVoxel(z, y, x) {
        if (z < 0)   z = 0;
        if (y < 0)   y = 0;
        if (x < 0)   x = 0;
        if (z >= D_in) z = D_in - 1;
        if (y >= H_in) y = H_in - 1;
        if (x >= W_in) x = W_in - 1;
        return inputData[z * H_in * W_in + y * W_in + x];
    }

    // 5. Main tricubic loops
    let outIndex = 0;
    for (let k = 0; k < D_out; k++) {
        const z = (k + 0.5) * scaleD - 0.5;  // align_corners=false mapping :contentReference[oaicite:5]{index=5}
        const z0 = Math.floor(z), z1 = z0 + 1, dz = z - z0;

        for (let j = 0; j < H_out; j++) {
        const y = (j + 0.5) * scaleH - 0.5;
        const y0 = Math.floor(y), y1 = y0 + 1, dy = y - y0;

        for (let i = 0; i < W_out; i++, outIndex++) {
            const x = (i + 0.5) * scaleW - 0.5;
            const x0 = Math.floor(x), x1 = x0 + 1, dx = x - x0;

            // Corners
            const c000 = getVoxel(z0, y0, x0);
            const c100 = getVoxel(z0, y0, x1);
            const c010 = getVoxel(z0, y1, x0);
            const c110 = getVoxel(z0, y1, x1);
            const c001 = getVoxel(z1, y0, x0);
            const c101 = getVoxel(z1, y0, x1);
            const c011 = getVoxel(z1, y1, x0);
            const c111 = getVoxel(z1, y1, x1);

            // Interpolate X
            const c00 = c000 * (1 - dx) + c100 * dx;
            const c10 = c010 * (1 - dx) + c110 * dx;
            const c01 = c001 * (1 - dx) + c101 * dx;
            const c11 = c011 * (1 - dx) + c111 * dx;

            // Interpolate Y
            const c0 = c00 * (1 - dy) + c10 * dy;
            const c1 = c01 * (1 - dy) + c11 * dy;

            // Interpolate Z
            outData[outIndex] = c0 * (1 - dz) + c1 * dz;
        }
        }
    }

    // 6. Pack and return a new tf.Tensor3D
    return tf.tensor3d(outData, [D_out, H_out, W_out], 'float32');  // tf.tensor3d :contentReference[oaicite:6]{index=6}
}
  