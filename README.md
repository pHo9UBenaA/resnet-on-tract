# Resnet on Wasm

## セットアップ

### モデルとラベルの取得

```bash
# モデルを取得
curl -O https://media.githubusercontent.com/media/onnx/models/refs/heads/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx

# ラベルを取得
curl -O https://raw.githubusercontent.com/onnx/models/refs/heads/main/validated/vision/classification/synset.txt
```

## 開発コマンド

### Wasm関連

```bash
# Wasmビルド
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --release --target no-modules

# Wasmテスト
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack test --release --chrome --headless 
```

### コード検証

```bash
cargo check --release;
cargo fmt --check;
cargo clippy --release;
```

### 開発サーバー起動

```bash
pnpm run dev
```

## 参考リンク

https://github.com/onnx/models/tree/main/validated/vision/classification/resnet
