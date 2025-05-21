# ONNX + PWA Template & Demo Application

Example application using ONNX Runtime Web & Progressive Web Application (PWA) principles to deploy an image classifier CNN in-browser, on client devices, with offline functionality.


## For JHU RAIL Usage (Running locally)
0) Clone this repo, and on the command line, change directory to `onnx-pwa-demo/`
1) Install Node Version Manager from this [link](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating)
2) Install node package manager: `nvm install --lts`
3) Run the setup script: `npm run setup`
4) Run tests to verify installation: `npm test`
5) Run the local dev server: `netlify dev`
6) (Hint) if you want to use webgpu, ensure that you are running a supported [browser](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)


## Converting pytorch models to onnx files
0) Please follow this [demo](https://docs.pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
1) Host you model on huggingface.
2) Go the `public/models.js` and add a new model. Use the existing model as a template. You will need to write your own pre/post processing functions
3) Write a new model test file based on for your new model `public/super_resolution-test.spec.js` to ensure your model is working correctly
4) (Hint) if your model doesn't work, it's probably the pre and post processing functions double check those.


## TODO

- [] Limit the number of files user can upload and check their types.
- [x] Ensure that model is return tensors in models.js
- [] Create a utility function that takes the output tensors, converts each one into an array, creates a zip folder using jszip (https://stuk.github.io/jszip/documentation/examples.html), adds each array file to the folder as an img file, zips the file, and lets the user download the file when a button in pushed, using Filesaver.js (https://github.com/eligrey/FileSaver.js/).
- [] On the page, list run times and show a single download button at the bottom.
- [] Write test suite
