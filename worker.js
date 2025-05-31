// Worker has its own scope and no direct access to functions/objects of the
// global scope. We import the generated JS file to make `wasm_bindgen`
// available which we need to initialize our Wasm code.
importScripts('./pkg/resnet_on_wasm.js');

console.log('Initializing worker')

// In the worker, we have a different struct that we want to use as in
// `index.js`.
const { setup_worker_message_handler } = wasm_bindgen;

async function init_wasm_in_worker() {
    // Load the Wasm file by awaiting the Promise returned by `wasm_bindgen`.
    await wasm_bindgen('./pkg/resnet_on_wasm_bg.wasm');

    // Setup message handler for the worker
    setup_worker_message_handler();
    
    console.log('Worker initialized and ready');
};

init_wasm_in_worker(); 