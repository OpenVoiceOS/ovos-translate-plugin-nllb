import ctranslate2
import os
import requests
import sentencepiece as spm
import shutil
from huggingface_hub import hf_hub_download
from ovos_plugin_manager.templates.language import LanguageTranslator
from ovos_utils import classproperty
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home
from typing import Union, List, Set
from zipfile import ZipFile


class NLLB200Translator(LanguageTranslator):
    """
    NLLB200Translator is a translation service that uses NLLB-200 models to translate text between different languages.
    """

    __base_url = "https://pretrained-nmt-models.s3.us-west-004.backblazeb2.com/CTranslate2/nllb"

    MODEL_URLS = {
        "nllb-200_600M_int8": f"{__base_url}/nllb-200_600M_int8_ct2.zip",
        "nllb-200_1.2B_int8": f"{__base_url}/nllb-200_1.2B_int8_ct2.zip",
        "nllb-200-3.3B-int8": f"{__base_url}/nllb-200_3.3B_int8_ct2.zip",
        "flores200_sacrebleu_tokenizer_spm": f"{__base_url}/flores200_sacrebleu_tokenizer_spm.model"
    }
    HF_MODELS = {
        "nllb-200-distilled-1.3B-ct2-int8": "OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
        "nllb-200-3.3B-ct2-int8": "OpenNMT/nllb-200-3.3B-ct2-int8"
    }
    # convert BCP 47 to primary language subtag
    LANG_MAP = {
        "ace_Arab": "ace",
        "ace_Latn": "ace",
        "acm_Arab": "acm",
        "acq_Arab": "acq",
        "aeb_Arab": "aeb",
        "afr_Latn": "af",
        "ajp_Arab": "ajp",
        "aka_Latn": "ak",
        "amh_Ethi": "am",
        "apc_Arab": "apc",
        "arb_Arab": "ar",
        "ars_Arab": "ars",
        "ary_Arab": "ary",
        "arz_Arab": "arz",
        "asm_Beng": "as",
        "ast_Latn": "ast",
        "awa_Deva": "awa",
        "ayr_Latn": "ayr",
        "azb_Arab": "azb",
        "azj_Latn": "az",
        "bak_Cyrl": "ba",
        "bam_Latn": "bm",
        "ban_Latn": "ban",
        "bel_Cyrl": "be",
        "bem_Latn": "bem",
        "ben_Beng": "bn",
        "bho_Deva": "bho",
        "bjn_Arab": "bjn",
        "bjn_Latn": "bjn",
        "bod_Tibt": "bo",
        "bos_Latn": "bs",
        "bug_Latn": "bug",
        "bul_Cyrl": "bg",
        "cat_Latn": "ca",
        "ceb_Latn": "ceb",
        "ces_Latn": "cs",
        "cjk_Latn": "cjk",
        "ckb_Arab": "ckb",
        "crh_Latn": "crh",
        "cym_Latn": "cy",
        "dan_Latn": "da",
        "deu_Latn": "de",
        "dik_Latn": "dik",
        "dyu_Latn": "dyu",
        "dzo_Tibt": "dz",
        "ell_Grek": "el",
        "eng_Latn": "en",
        "epo_Latn": "eo",
        "est_Latn": "et",
        "eus_Latn": "eu",
        "ewe_Latn": "ee",
        "fao_Latn": "fo",
        "pes_Arab": "pes",
        "fij_Latn": "fj",
        "fin_Latn": "fi",
        "fon_Latn": "fon",
        "fra_Latn": "fr",
        "fur_Latn": "fur",
        "fuv_Latn": "fuv",
        "gla_Latn": "gd",
        "gle_Latn": "ga",
        "glg_Latn": "gl",
        "grn_Latn": "gn",
        "guj_Gujr": "gu",
        "hat_Latn": "ht",
        "hau_Latn": "ha",
        "heb_Hebr": "he",
        "hin_Deva": "hi",
        "hne_Deva": "hne",
        "hrv_Latn": "hr",
        "hun_Latn": "hu",
        "hye_Armn": "hy",
        "ibo_Latn": "ig",
        "ilo_Latn": "ilo",
        "ind_Latn": "id",
        "isl_Latn": "is",
        "ita_Latn": "it",
        "jav_Latn": "jv",
        "jpn_Jpan": "ja",
        "kab_Latn": "kab",
        "kac_Latn": "kac",
        "kam_Latn": "kam",
        "kan_Knda": "kn",
        "kas_Arab": "kas",
        "kas_Deva": "ks",
        "kat_Geor": "ka",
        "knc_Arab": "knc",
        "knc_Latn": "knc",
        "kaz_Cyrl": "kk",
        "kbp_Latn": "kbp",
        "kea_Latn": "kea",
        "khm_Khmr": "km",
        "kik_Latn": "ki",
        "kin_Latn": "rw",
        "kir_Cyrl": "ky",
        "kmb_Latn": "kmb",
        "kon_Latn": "kg",
        "kor_Hang": "ko",
        "kmr_Latn": "kmr",
        "lao_Laoo": "lo",
        "lvs_Latn": "lv",
        "lij_Latn": "lij",
        "lim_Latn": "li",
        "lin_Latn": "ln",
        "lit_Latn": "lt",
        "lmo_Latn": "lmo",
        "ltg_Latn": "ltg",
        "ltz_Latn": "lb",
        "lua_Latn": "lua",
        "lug_Latn": "lg",
        "luo_Latn": "luo",
        "lus_Latn": "lus",
        "mag_Deva": "mag",
        "mai_Deva": "mai",
        "mal_Mlym": "ml",
        "mar_Deva": "mr",
        "min_Latn": "min",
        "mkd_Cyrl": "mk",
        "plt_Latn": "plt",
        "mlt_Latn": "mt",
        "mni_Beng": "mni",
        "khk_Cyrl": "khk",
        "mos_Latn": "mos",
        "mri_Latn": "mi",
        "zsm_Latn": "zsm",
        "mya_Mymr": "my",
        "nld_Latn": "nl",
        "nno_Latn": "nn",
        "nob_Latn": "nb",
        "npi_Deva": "npi",
        "nso_Latn": "nso",
        "nus_Latn": "nus",
        "nya_Latn": "ny",
        "oci_Latn": "oc",
        "gaz_Latn": "gaz",
        "ory_Orya": "or",
        "pag_Latn": "pag",
        "pan_Guru": "pa",
        "pap_Latn": "pap",
        "pol_Latn": "pl",
        "por_Latn": "pt",
        "prs_Arab": "prs",
        "pbt_Arab": "pbt",
        "quy_Latn": "quy",
        "ron_Latn": "ro",
        "run_Latn": "rn",
        "rus_Cyrl": "ru",
        "sag_Latn": "sg",
        "san_Deva": "sa",
        "sat_Beng": "sat",
        "scn_Latn": "scn",
        "shn_Mymr": "shn",
        "sin_Sinh": "si",
        "slk_Latn": "sk",
        "slv_Latn": "sl",
        "smo_Latn": "sm",
        "sna_Latn": "sn",
        "snd_Arab": "sd",
        "som_Latn": "so",
        "sot_Latn": "st",
        "spa_Latn": "es",
        "als_Latn": "gsw",
        "srd_Latn": "sc",
        "srp_Cyrl": "sr",
        "ssw_Latn": "ss",
        "sun_Latn": "su",
        "swe_Latn": "sv",
        "swh_Latn": "sw",
        "szl_Latn": "szl",
        "tam_Taml": "ta",
        "tat_Cyrl": "tt",
        "tel_Telu": "te",
        "tgk_Cyrl": "tg",
        "tgl_Latn": "tl",
        "tha_Thai": "th",
        "tir_Ethi": "ti",
        "taq_Latn": "taq",
        "taq_Tfng": "taq",
        "tpi_Latn": "tpi",
        "tsn_Latn": "tn",
        "tso_Latn": "ts",
        "tuk_Latn": "tk",
        "tum_Latn": "tum",
        "tur_Latn": "tr",
        "twi_Latn": "tw",
        "tzm_Tfng": "tzm",
        "uig_Arab": "ug",
        "ukr_Cyrl": "uk",
        "umb_Latn": "umb",
        "urd_Arab": "ur",
        "uzn_Latn": "uz",
        "vec_Latn": "vec",
        "vie_Latn": "vi",
        "war_Latn": "war",
        "wol_Latn": "wo",
        "xho_Latn": "xh",
        "ydd_Hebr": "yi",
        "yor_Latn": "yo",
        "yue_Hant": "yue",
        "zho_Hans": "zh",
        "zho_Hant": "zh",
        "zul_Latn": "zu"
    }

    def __init__(self, *args, **kwargs):
        """
        Initialize the NLLB200Translator with the specified configuration.
        """
        super().__init__(*args, **kwargs)
        model = self.config.get("model", "nllb-200_600M_int8")
        tokenizer = self.config.get("tokenizer", "flores200_sacrebleu_tokenizer_spm")

        self.ct_model_path = self.download(model)
        self.sp_model_path = self.download_tokenizer(tokenizer)
        self.beam_size = self.config.get("beam_size", 4)
        self.device = self.config.get("device", "cpu")
        # Load the source SentecePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.sp_model_path)

        self.translator = ctranslate2.Translator(self.ct_model_path, self.device)

    @staticmethod
    def _download_file(url: str, local_filename: str) -> str:
        """
        Download a file from a given URL and save it locally.

        Args:
            url (str): The URL of the file to download.
            local_filename (str): The local path where the file should be saved.

        Returns:
            str: The local file path of the downloaded file.
        """
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    @classmethod
    def _download_from_hf(cls, repo_id: str) -> str:
        if "facebook" in cls.HF_MODELS[repo_id]:
            files = ["config.json", "shared_vocabulary.json", "model.bin"]
        for f in files:
            p = hf_hub_download(repo_id=cls.HF_MODELS[repo_id], filename=f)
            model = os.path.dirname(p)
        return model

    @classmethod
    def download_tokenizer(cls, tokenizer: str = "flores200_sacrebleu_tokenizer_spm") -> str:
        """
        Download and extract the specified tokenizer.

        Args:
            tokenizer (str, optional): The tokenizer to download. Defaults to "flores200_sacrebleu_tokenizer_spm".

        Returns:
            str: Path to the downloaded model.
        """
        base_path = f"{xdg_data_home()}/ctranslate2"
        os.makedirs(base_path, exist_ok=True)
        tokenizer_path = f"{base_path}/{tokenizer}.model"
        if not os.path.isfile(tokenizer_path):
            cls._download_file(cls.MODEL_URLS[tokenizer], tokenizer_path)
        return tokenizer_path

    @classmethod
    def download(cls, model: str = "nllb-200_600M_int8") -> str:
        """
        Download and extract the specified model.

        Args:
            model (str, optional): The model to download. Defaults to "nllb-200_600M_int8".

        Returns:
            str: Path to the downloaded model.
        """
        base_path = f"{xdg_data_home()}/ctranslate2"
        os.makedirs(base_path, exist_ok=True)

        model_path = model if os.path.isdir(model) else f"{base_path}/{model}"

        if model in cls.HF_MODELS:
            model_path = cls._download_from_hf(model)
            return model_path

        if not os.path.isdir(model_path):
            if model_path.startswith("http"):
                url = model_path
            else:
                url = cls.MODEL_URLS[model]

            zipped = f"{base_path}/{url.split('/')[-1]}"
            if not os.path.isfile(zipped):
                LOG.info(f"Downloading {url}")
                cls._download_file(url, zipped)

            with ZipFile(zipped, 'r') as z:
                tmp = f"{base_path}/.extracted"
                LOG.debug(f"Unzipping downloaded model to {model_path}")
                # Extracting all the members of the zip
                # into a specific location.
                z.extractall(path=tmp)
                dl_path = f"{tmp}/{os.listdir(tmp)[0]}"
                shutil.move(dl_path, model_path)
                shutil.rmtree(tmp)

            LOG.debug(f"Deleting temporary zip file: {zipped}")
            os.remove(zipped)

        return model_path

    def translate(self, text: Union[str, List[str]], target: str = "", source: str = "") -> Union[str, List[str]]:
        """
        Translate text(s) into the target language using the NLLB200 model.

        Args:
            text (Union[str, List[str]]): The sentence(s) to translate.
            target (str, optional): The target language code. Defaults to "".
            source (str, optional): The source language code. Defaults to "".

        Returns:
            Union[str, List[str]]: The translated sentence(s).
        """
        if isinstance(text, str):
            utterances = [text]
        else:
            utterances = text

        mapping = {v: k for k, v in self.LANG_MAP.items()}
        tgt_lang = target
        src_lang = source

        if source not in self.LANG_MAP:
            lang = mapping.get(source) or mapping.get(source.split("-")[0])
            if not lang:
                raise ValueError(f"Invalid source language: {source}")
            src_lang = lang

        if target not in self.LANG_MAP:
            lang = mapping.get(target) or mapping.get(target.split("-")[0])
            if not lang:
                raise ValueError(f"Invalid target language: {target}")
            tgt_lang = lang

        source_sents = [sent.strip() for sent in utterances]
        target_prefix = [[tgt_lang]] * len(source_sents)

        # Subword the source sentences
        source_sents_subworded = self.sp.encode(source_sents, out_type=str)
        source_sents_subworded = [sent + ["</s>", src_lang] for sent in source_sents_subworded]

        results = self.translator.translate_batch(
            source_sents_subworded,
            batch_type="tokens",
            max_batch_size=2024,
            beam_size=self.beam_size,
            target_prefix=target_prefix
        )

        translations = [translation.hypotheses[0] for translation in results]
        translations_desubword = [sent[len(tgt_lang):].strip() for sent in self.sp.decode(translations)]

        if len(utterances) == 1:
            return translations_desubword[0]
        return translations_desubword

    @classproperty
    def available_languages(cls) -> Set[str]:
        """
        Get the available target languages with the service.

        Returns:
            Set[str]: A set of language codes.
        """
        return set(cls.LANG_MAP.values())

    def supported_translations(self, source_lang: str) -> Set[str]:
        """
        Get the set of target languages to which the source language can be translated.

        Args:
            source_lang (Optional[str]): The source language code.

        Returns:
            Set[str]: A set of language codes that the source language can be translated to.
        """
        return self.available_languages


if __name__ == "__main__":
    src = "es"
    tgt = "en-us"

    tx = NLLB200Translator(config={"model": "nllb-200-3.3B-int8",
                                   "beam_size": 5,
                                   "device": "cuda"})

    utts = ["Hola Mundo"]
    print("Translations:", tx.translate(utts, tgt, src))

    utts = "hello world"
    print("Translations:", tx.translate(utts, src, tgt))
