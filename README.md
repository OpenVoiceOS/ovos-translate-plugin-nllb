[![codecov](https://codecov.io/github/OpenVoiceOS/ovos-translate-plugin-nllb/branch/dev/graph/badge.svg?token=TOViE9yLEg)](https://codecov.io/github/OpenVoiceOS/ovos-translate-plugin-nllb)

# OVOS No Language Left Behind Plugin (Ctranslate2)

Language Plugin for [NLLB200](https://ai.facebook.com/research/no-language-left-behind/) language translator


## Usage

### OVOS

The plugin is used in a wider context to translate utterances/texts on demand (e.g. from a [UniversalSkill]())

_Configuration_
```python
# add this to one of the configuration files (eg ~./config/mycroft/mycroft.conf)

"language": {
    "translation_module": "ovos-translate-plugin-nllb",
    "ovos-translate-plugin-nllb": {
        "model": "nllb-200_600M_int8"
    }
}

```