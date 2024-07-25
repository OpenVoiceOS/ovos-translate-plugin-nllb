[![codecov](https://codecov.io/github/OpenVoiceOS/ovos-translate-plugin-nllb/branch/dev/graph/badge.svg?token=TOViE9yLEg)](https://codecov.io/github/OpenVoiceOS/ovos-translate-plugin-nllb)

# OVOS No Language Left Behind Plugin (CTranslate2)

Language Plugin for [NLLB200](https://ai.facebook.com/research/no-language-left-behind/) language translator.

## Overview

The `NLLB200Translator` is a translation service that uses the NLLB-200 models to translate text between different languages. It integrates with the OVOS framework and utilizes CTranslate2 for fast and efficient translation.

## Model Options

Below are the available model options for the NLLB-200 Translator:

| Model Name                        | Source                                                    | Description                          |
|-----------------------------------|-----------------------------------------------------------|--------------------------------------|
| flores200_sacrebleu_tokenizer_spm | CTranslate2                                               | Tokenizer model                      |
| nllb-200_600M_int8                | CTranslate2                                               | 600M parameter model, int8 quantized |
| nllb-200_1.2B_int8                | CTranslate2                                               | 1.2B parameter model, int8 quantized |
| nllb-200_3.3B_int8                | CTranslate2                                               | 3.3B parameter model, int8 quantized |
| nllb-200-distilled-1.3B-ct2-int8  | HuggingFace Hub: OpenNMT/nllb-200-distilled-1.3B-ct2-int8 | 1.3B distilled model, int8 quantized |
| nllb-200-3.3B-ct2-int8            | HuggingFace Hub: OpenNMT/nllb-200-3.3B-ct2-int8           | 3.3B model, int8 quantized           |

## Usage

### OVOS Integration

The plugin can be used within the OVOS framework to translate utterances or texts on demand. Below is an example configuration for integrating the translator within OVOS.

#### Configuration

Add the following configuration to one of the configuration files (e.g., `~/.config/mycroft/mycroft.conf`):

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

## Example

Here's an example of how to use the `NLLB200Translator` in a Python script:

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

To use models hosted on the HuggingFace Hub, configure the model parameter with the respective HuggingFace model ID. The `NLLB200Translator` class handles downloading and loading the model from HuggingFace.

### Additional Parameters

- `beam_size`: Configure the beam size used for translation to balance between translation quality and speed.
- `device`: Specify the device type, either 'cpu' or 'cuda' (for GPU).

Example:

```python
from ovos_translate_plugin_nllb import NLLB200Translator
tx = NLLB200Translator(config={
    "model": "nllb-200_600M_int8",
    "beam_size": 5,
    "device": "cuda"
})
```


## Using CUDA/GPU

To leverage GPU acceleration with CUDA, configure the `NLLB200Translator` to use the `cuda` device. This can significantly speed up the translation process, especially for large batches or longer texts.

### Prerequisites

Ensure you have a CUDA-compatible GPU and the necessary CUDA drivers installed on your system. Refer to the [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for more details.

### Example

Here's a complete example demonstrating how to configure and use the `NLLB200Translator` with CUDA:

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