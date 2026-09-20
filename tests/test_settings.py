from geo_mini_rag import settings


def test_one_model_each_for_answering_and_embedding():
    cfg = settings.load_rag_config()
    assert settings.chat_model() == cfg["answer"]["model"]
    assert settings.embed_model() == cfg["embed"]["model"]
    assert settings.chat_model("someone/other-model") == "someone/other-model"


def test_data_dirs_exist():
    for d in (settings.MANIFEST_DIR, settings.OCR_DIR, settings.INDEX_DIR):
        assert d.is_dir()
