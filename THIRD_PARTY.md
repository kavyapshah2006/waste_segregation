# Third-party components

This project is licensed under the **GNU AGPL-3.0** ([LICENSE](LICENSE)), chosen because the detector is trained with Ultralytics, whose code and trained weights are AGPL-3.0. Licences below were checked on 2026-11-18 against each project's own licence file or model card.

## Models shipped or used by the app

| Component | Use | Licence | Obligations for a release |
|---|---|---|---|
| Llama 3.2 1B Instruct (Meta), incl. the `executorch-community` SpinQuant/QLoRA `.pte` files | Explanations | Llama 3.2 Community License + Acceptable Use Policy | Ship a copy of the licence with the model; display **"Built with Llama"** prominently (README, app About screen); keep the notice in [NOTICE](NOTICE); a fine-tuned model's name must **start with "Llama"**; follow the Acceptable Use Policy (llama.com/llama3_2/use-policy) |
| YOLO26n (Ultralytics) and our trained weights | Detector | AGPL-3.0 | Source of the whole app available under AGPL-3.0 (this repo) |
| SigLIP 2 base patch16-224 (Google, `google/siglip2-base-patch16-224`) | Unknown-item fallback | Apache-2.0 | Keep licence and notices |
| Omnilingual ASR 300M CTC v2 (Meta), sherpa-onnx int8 build | Hindi speech recognition | Apache-2.0 | Keep licence and notices |

## Libraries

| Component | Licence |
|---|---|
| ExecuTorch (runtime, Android AAR) | BSD-3-Clause |
| PyTorch, torchao | BSD-3-Clause |
| pytorch-tokenizers | BSD-3-Clause |
| sherpa-onnx | Apache-2.0 |
| Hugging Face transformers, PEFT, Accelerate (training only) | Apache-2.0 |
| Ultralytics (training and export only) | AGPL-3.0 |

## Data

| Data | Use | Licence | Note |
|---|---|---|---|
| TACO (pedropro/TACO) | Detector training | Code MIT; **images mixed**: 466 ODbL (OpenLitterMap), 319 marked only "CC", 715 with no licence field | Images are not redistributed. Before publishing trained weights, check that training on the unlabelled-licence images is acceptable, or train without them |
| Solid Waste Management Rules, 2026 (MoEFCC) | KB stream and bin rules | Government publication | Cited in each KB item's `sources` |
| US EPA clean-up guidance (broken CFLs, mercury thermometers) | KB steps | US Government work | Cited in the KB items |

## Used on the laptop only (not shipped)

| Component | Use | Licence |
|---|---|---|
| Piper voice `hi_IN-pratham-medium` | Synthetic Hindi audio for testing speech recognition | Trained on CC BY-NC-SA 4.0 data (non-commercial); never shipped. The app uses Android's own text-to-speech |
| Dolphin CTC, Whisper base | Compared against Omnilingual ASR | Apache-2.0, MIT |

Rejected: Apple MobileCLIP (weights licensed for non-commercial research only).
