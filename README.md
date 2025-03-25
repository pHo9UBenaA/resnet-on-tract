# 

https://github.com/onnx/models/tree/main/validated/vision/classification/resnet

```bash
curl -O https://media.githubusercontent.com/media/onnx/models/refs/heads/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx;


curl -O https://raw.githubusercontent.com/onnx/models/refs/heads/main/validated/vision/classification/synset.txt;
```

```bash
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target no-modules
```

```bash
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack test --release --chrome
```
