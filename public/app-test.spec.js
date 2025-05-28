const { test, expect } = require('@playwright/test');
const path = require('path');

test.describe('ONNX Models on Webserver', () => {
  const serverUrl = 'http://localhost:8888'; // Replace with your webserver's URL if different
  const sampleImagePath = path.resolve(__dirname, './test.jpg'); // Path to the sample image

  test('VIT Face Expression Classification model runs correctly', async ({ page }) => {
    // Navigate to the webserver
    await page.goto(serverUrl);

    // Ensure the page has loaded
    await expect(page).toHaveTitle("ONNX + PWA Demo");

    // Select the "VIT Face Expression Classification" model
    await page.selectOption('#modelSelect', { label: 'VIT Face Expression Classification' });

    // Wait for the model to load
    await page.waitForSelector('#modelDownloadProgress', { state: 'detached' });

    // Simulate a file drop
    const fileInput = await page.$('input[type="file"]'); // Locate the hidden file input inside the dropzone
    await fileInput.setInputFiles(sampleImagePath);

    // Click the "Run" button
    await page.click('#runButton');

    // Wait for the results to appear
    await page.waitForSelector('#multiOutputContainer:not(.d-none)');

    // Verify the results
    const resultLabel = await page.textContent('#multiOutputContainer table tbody tr td:nth-child(2)');
    expect(resultLabel).toMatch(/angry|disgust|fear|happy|neutral|sad|surprise/);
  });

  test('Super Resolution model runs correctly', async ({ page }) => {
    // Navigate to the webserver
    await page.goto(serverUrl);

    // Ensure the page has loaded
    await expect(page).toHaveTitle("ONNX + PWA Demo");

    // Select the "Super Resolution" model
    await page.selectOption('#modelSelect', { label: 'Super Resolution' });

    // Wait for the model to load
    await page.waitForSelector('#modelDownloadProgress', { state: 'detached' });

    // Simulate a file drop
    const fileInput = await page.$('input[type="file"]'); // Locate the hidden file input inside the dropzone
    await fileInput.setInputFiles(sampleImagePath);

    // Click the "Run" button
    await page.click('#runButton');

    // Wait for the results to appear
    await page.waitForSelector('#multiOutputContainer:not(.d-none)');

    // Verify the results
    const resultLabel = await page.textContent('#multiOutputContainer table tbody tr td:nth-child(2)');
    expect(resultLabel).toMatch(/^\d+x\d+$/); // Expect a resolution format like "672x672"
  });
});