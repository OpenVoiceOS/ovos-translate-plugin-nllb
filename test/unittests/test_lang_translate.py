import sys
import unittest
from os.path import dirname, realpath

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from ovos_translate_plugin_nllb import NLLB200Translator


class LangTranslateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.translator = NLLB200Translator()

    def test_translate_spec_input(self):
        translated = self.translator.translate("Hello World", "es", "en")
        self.assertEqual(translated.lower(), "hola mundo")
        # full lang code
        translated = self.translator.translate("Hello World", "es-es", "en-us")
        self.assertEqual(translated.lower(), "hola mundo")


if __name__ == '__main__':
    unittest.main()
