import io
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from ovos_translate_plugin_nllb import NLLB200Translator

MODEL = "nllb-200_600M_int8"


def _model_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("nllb-200_600M_int8_ct2/model.bin", b"weights")
    return buf.getvalue()


class _InterruptedResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b"PK\x03\x04partial"
        raise ConnectionError("connection reset")


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.data_home = tempfile.TemporaryDirectory()
        self.addCleanup(self.data_home.cleanup)
        patcher = mock.patch("ovos_translate_plugin_nllb.xdg_data_home",
                             return_value=self.data_home.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.base = os.path.join(self.data_home.name, "ctranslate2")
        os.makedirs(self.base)
        self.zipped = os.path.join(self.base, "nllb-200_600M_int8_ct2.zip")

    def test_interrupted_download_leaves_no_file_at_final_path(self):
        with mock.patch("ovos_translate_plugin_nllb.requests.get",
                        return_value=_InterruptedResponse()):
            with self.assertRaises(ConnectionError):
                NLLB200Translator._download_file("http://example.invalid/m.zip", self.zipped)
        self.assertFalse(os.path.exists(self.zipped))

    def test_truncated_archive_is_downloaded_again(self):
        with open(self.zipped, "wb") as f:
            f.write(_model_zip_bytes()[:20])

        def fake_download(url, local_filename):
            with open(local_filename, "wb") as f:
                f.write(_model_zip_bytes())
            return local_filename

        with mock.patch.object(NLLB200Translator, "_download_file",
                               side_effect=fake_download) as dl:
            path = NLLB200Translator.download(MODEL)

        dl.assert_called_once()
        self.assertEqual(path, os.path.join(self.base, MODEL))
        with open(os.path.join(path, "model.bin"), "rb") as f:
            self.assertEqual(f.read(), b"weights")
        self.assertFalse(os.path.exists(self.zipped))


if __name__ == "__main__":
    unittest.main()
