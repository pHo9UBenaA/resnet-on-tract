# 

https://github.com/onnx/models/tree/main/validated/vision/classification/efficientnet-lite4

```bash
# EfficientNet-Lite4-int8
curl -O https://media.githubusercontent.com/media/onnx/models/refs/heads/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx;


curl -O https://raw.githubusercontent.com/onnx/models/refs/heads/main/validated/vision/classification/efficientnet-lite4/dependencies/labels_map.txt
```

```bash
wasm-pack build --target no-modules

RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target no-modules
```
