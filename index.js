const {startup} = wasm_bindgen;

async function run_wasm() {
    await wasm_bindgen();

    startup();
}

run_wasm();
