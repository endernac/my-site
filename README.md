# ONNX + PWA Template & Demo Application

Example application using ONNX Runtime Web & Progressive Web Application (PWA) principles to deploy an image classifier CNN in-browser, on client devices, with offline functionality.


## For JHU RAIL Usage (Running locally)
0) Clone this repo, and on the command line, change directory to `onnx-pwa-demo/`
1) Install Node Version Manager from this [link](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating)
2) Install node package manager: `nvm install --lts`
3) Run the setup script: `npm run setup`
4) Run tests to verify installation: `npm test`
5) Run the local dev server: `netlify dev`


TODO:
- [ ] Create a new function `naive_resize()` in utils.js to return a greyscale resized image w/o superresolution
- [ ] modify index.js to create a side by side comparison of `naive_resize()` and the output of the superresolution model


## Usage (Original)
0) **Prerequisite:** Convert a trained neural network to ONNX format. For details, see [ONNX Tutorials](https://github.com/onnx/tutorials?tab=readme-ov-file#converting-to-onnx-format).
1) Clone this repository
2) Create `ONNXModel` object and implement data pre-/post-processing functions in JavaScript as needed
    - See [`models.js`](./models.js) for example(s)
3) Modify output rendering UI (if applicable)
    - Example application tabulates output, which works for image classification models. Other model types may require custom UI implementation. Please feel free to file issues and/or submit pull requests to contribute modular UI components to this repo!
4) Deploy the customized static website via web hosting
    - Many free services exist for static website hosting, e.g., GitHub Pages, Netlify, etc.
    - Large files such as ONNX models may be served independently (e.g. using Cloudflare R2, AWS, HuggingFace, etc.) and linked via URL in [`models.js`](./models.js)


## Contributions
Issues and PRs are welcomed and encouraged!
