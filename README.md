# Zayn's MiniMind-V

A compact implementation of single-image language-model training, adapted from [MiniMind-V](https://github.com/jingyaogong/minimind-v) and [MiniMind](https://github.com/jingyaogong/minimind). It combines a MiniMind language model, a frozen SigLIP vision encoder, and a LayerNorm/MLP projector. See [NOTICE](NOTICE) and [LICENSE](LICENSE) for attribution.

The implementation uses a wider projector, validates one contiguous image span per example, inserts image features with batched tensor operations, and masks image/padding positions from language loss. Exports bundle the vision weights and processor. The language model shares its parameter layout with [Zayn's MiniMind](https://github.com/Zaynmagic/Zayn-s-minimind).

## Setup and quick run

Use Python 3.10 or newer. Install PyTorch for your hardware, then:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m training.run --allow-random-llm --allow-random-vision --steps 6
```

The included colored-square fixtures and random tiny model check the training pipeline; they do not produce a useful vision assistant. Recreate fixtures with `python -m training.fixtures`.

To use the tiny language checkpoint produced by the companion repository, place both checkouts side by side:

```bash
python -m training.run --llm-checkpoint ../Zayn-s-minimind/runs/smoke/sft.pt --allow-random-vision --steps 8
```

Alignment trains only the projector. SFT trains the language model and projector. The vision encoder stays frozen and in evaluation mode during both stages. Each run verifies that frozen weights did not change and that the projector updated.

## Training on your data

Create `align.jsonl` and `sft.jsonl` in a data directory. Image paths are relative to that directory. Each record contains one image and starts with a user message containing one `<image>` placeholder:

```json
{"image":"images/example.jpg","conversations":[{"role":"user","content":"<image> Describe the scene."},{"role":"assistant","content":"A dog running across grass."}]}
```

Use a pretrained SigLIP checkpoint and a matching language-model checkpoint for a full run:

```bash
python -m training.run \
  --config configs/64m.json \
  --vision-source jingyaogong/siglip2-base-p32-256-ve \
  --vision-revision 9465d1dc89db6bc6227c5b6b0e0ca9b940325d62 \
  --llm-checkpoint ../Zayn-s-minimind/runs/dense/sft.pt \
  --data /path/to/image-corpus \
  --steps 10000 --batch-size 4 --length 512 \
  --precision bf16 --output runs/full
```

`--steps` counts microbatches per stage; `--accumulation` groups them into optimizer updates. Increase `--length` if the image span or assistant response is truncated. Choose `--stages align` or `--stages sft` to run a single stage. BF16 requires hardware support; FP16 requires CUDA. Random vision/language initialization requires the explicit flags shown in the quick run.

CPU/CUDA distributed training uses `torchrun`:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m training.run --allow-random-llm --allow-random-vision \
  --steps 3 --accumulation 2 --output runs/ddp-check
```

## Inference and exports

Runs write stage checkpoints, `metrics.json`, and a self-contained Hugging Face export under `hf/`. Run a local image question with:

```bash
python inference.py --model runs/smoke/hf \
  --image data/synthetic/images/red.png --question "What color is the square?"
```

The export includes custom Python code; load only exports you trust. The CLI uses greedy cached decoding and supports `--device cuda`. Model assets retain the original tokenizer IDs for compatibility. Checkpoints and generated outputs stay outside version control.

Tests cover frozen training stages, image-dependent outputs, input validation, loss masking, cached decoding, and export/reload. CI also runs CPU smoke training and two-process DDP. These are correctness checks; full-corpus quality and GPU performance require separate training and evaluation.
