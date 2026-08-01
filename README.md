[![codecov](https://codecov.io/github/OpenVoiceOS/ovos-translate-plugin-nllb/branch/dev/graph/badge.svg?token=TOViE9yLEg)](https://codecov.io/github/OpenVoiceOS/ovos-translate-plugin-nllb)

# OVOS No Language Left Behind Plugin (CTranslate2)

This is a language plugin for [NLLB-200](https://ai.facebook.com/research/no-language-left-behind/), a translation model from Meta AI.

## Overview

`NLLB200Translator` translates text between languages with the NLLB-200 models. It plugs into the OVOS framework and uses CTranslate2 to run translation fast and efficiently.

## Model Options

The table below lists the available models for the NLLB-200 translator.

| Model Name                        | Source                                                    | Description                          |
|-----------------------------------|-----------------------------------------------------------|--------------------------------------|
| flores200_sacrebleu_tokenizer_spm | CTranslate2                                               | Tokenizer model                      |
| nllb-200_600M_int8                | CTranslate2                                               | 600M parameter model, int8 quantized |
| nllb-200_1.2B_int8                | CTranslate2                                               | 1.2B parameter model, int8 quantized |
| nllb-200_3.3B_int8                | CTranslate2                                               | 3.3B parameter model, int8 quantized |
| nllb-200-distilled-1.3B-ct2-int8  | HuggingFace Hub: OpenNMT/nllb-200-distilled-1.3B-ct2-int8 | 1.3B distilled model, int8 quantized |
| nllb-200-3.3B-ct2-int8            | HuggingFace Hub: OpenNMT/nllb-200-3.3B-ct2-int8           | 3.3B model, int8 quantized           |

## Install

Install the plugin with pip:

```bash
pip install ovos-translate-plugin-nllb
```

## Usage

### OVOS Integration

Use the plugin inside OVOS to translate utterances and text on demand. Add this configuration to one of the configuration files, for example `~/.config/mycroft/mycroft.conf`:

```json
{
  "language": {
    "translation_module": "ovos-translate-plugin-nllb",
    "ovos-translate-plugin-nllb": {
      "model": "nllb-200_600M_int8"
    }
  }
}
```

### Python Script

This example shows how to use `NLLB200Translator` in a Python script.

```python
from ovos_translate_plugin_nllb import NLLB200Translator

src = "es"
tgt = "en-us"
tx = NLLB200Translator(config={"model": "nllb-200_600M_int8"})

utts = ["Hola Mundo"]
print("Translations:", tx.translate(utts, tgt, src))

utts = "hello world"
print("Translations:", tx.translate(utts, src, tgt))
```

## Advanced Configuration

### HuggingFace Integration

To use a model hosted on the HuggingFace Hub, set the `model` parameter to the HuggingFace model ID. `NLLB200Translator` downloads and loads the model from HuggingFace.

### Additional Parameters

- `beam_size`: the beam size used for translation. A larger value trades speed for translation quality.
- `device`: the device to run on, either `cpu` or `cuda` (GPU).

```python
from ovos_translate_plugin_nllb import NLLB200Translator
tx = NLLB200Translator(config={
    "model": "nllb-200_600M_int8",
    "beam_size": 5,
    "device": "cuda"
})
```

## Using CUDA/GPU

Set `NLLB200Translator` to the `cuda` device to use GPU acceleration. This can speed up translation for large batches or long texts.

### Prerequisites

You need a CUDA-compatible GPU and the CUDA drivers for your system. See the [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for details.

### Example

This example configures and runs `NLLB200Translator` with CUDA.

```python
from ovos_translate_plugin_nllb import NLLB200Translator

if __name__ == "__main__":
    src = "es"
    tgt = "en-us"
    tx = NLLB200Translator(config={
        "model": "nllb-200-3.3B-int8",
        "beam_size": 5,
        "device": "cuda"
    })
    
    utts = ["Hola Mundo"]
    print("Translations:", tx.translate(utts, tgt, src))
    
    utts = "hello world"
    print("Translations:", tx.translate(utts, src, tgt))
```

## Related Projects

- [OpenVoiceOS/ovos-translate-server-plugin](https://github.com/OpenVoiceOS/ovos-translate-server-plugin) - a translation plugin that calls a remote translate server
- [OpenVoiceOS/ovos-translate-server](https://github.com/OpenVoiceOS/ovos-translate-server) - a server that exposes translation plugins over HTTP
- [OpenVoiceOS/ovos-bidirectional-translation-plugin](https://github.com/OpenVoiceOS/ovos-bidirectional-translation-plugin) - a plugin that chains two translators for bidirectional translation

## License

This project is licensed under the [Apache License 2.0](LICENSE).
