"""realmodels.py - load the small real models used in the lesson and read their tables.

What it does
    * all-MiniLM-L6-v2 (Apache-2.0): a small public model that turns a sentence into 384 numbers. We read its token table (30,522 rows),
      its learned position table (512 rows) and run it with the Hugging Face transformers library, doing the pooling by hand.
    * GloVe 50-number word vectors (public domain, PDDL licence) through gensim, for the classic king - man + woman example.
    * config.json files of GPT-2 and Qwen2.5-0.5B-Instruct: only a few kilobytes each, they hold the table sizes and the RoPE settings.
What it needs
    torch, transformers, huggingface_hub and gensim (see pyproject.toml). The first run downloads about 90 MB for MiniLM, 100 MB for GloVe
    and a few kilobytes of config files; later runs use the saved copies.
What to expect
    Each loader raises an ordinary error if a download is not possible; try_load() in the demos turns that into one printed line.
"""
import json

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
GLOVE_REPO = "fse/glove-wiki-gigaword-50"


def load_minilm():
    """Return (tokenizer, model) for all-MiniLM-L6-v2, with the model in evaluation mode."""
    from transformers import AutoModel, AutoTokenizer
    return AutoTokenizer.from_pretrained(MINILM), AutoModel.from_pretrained(MINILM).eval()


def sentence_vectors(texts, tokenizer, model):
    """Turn texts into unit-length sentence vectors: run the model, average the token vectors (ignoring padding), scale to length 1.

    This is the same recipe the model's authors publish (mean pooling, then normalisation), written out so every step is visible.
    """
    import torch
    import torch.nn.functional as F
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state                # (texts, tokens, 384): one vector per token, already mixed by attention
    mask = batch["attention_mask"].unsqueeze(-1).float()         # 1 for real tokens, 0 for padding
    mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return F.normalize(mean, dim=1).numpy()


def token_vectors(text, tokenizer, model):
    """Return (tokens, vectors): the pieces of one text and the model's output vector for each piece (tokens x 384)."""
    import torch
    batch = tokenizer([text], return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state[0]
    return tokenizer.convert_ids_to_tokens(batch["input_ids"][0]), hidden.numpy()


def token_table(model):
    """The input embedding table as a NumPy array: shape (vocabulary size, 384). Row i belongs to token id i."""
    return model.embeddings.word_embeddings.weight.detach().numpy()


def position_table(model):
    """The learned position table as a NumPy array: shape (512, 384). Row p is added to the token at position p."""
    return model.embeddings.position_embeddings.weight.detach().numpy()


def hub_config(repo):
    """Download and parse config.json of a Hugging Face repository (a few kilobytes; the model weights stay where they are)."""
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as handle:
        return json.load(handle)


def load_glove():
    """Return gensim KeyedVectors with 400,000 GloVe words of 50 numbers each (trained on Wikipedia and a news archive)."""
    from gensim.models import KeyedVectors
    from huggingface_hub import hf_hub_download
    hf_hub_download(GLOVE_REPO, "glove-wiki-gigaword-50.model.vectors.npy")     # the big array of numbers; it must sit next to the .model file
    return KeyedVectors.load(hf_hub_download(GLOVE_REPO, "glove-wiki-gigaword-50.model"))


def try_load(label, loader, *args):
    """Run a loader; on a failure return None and print one short line so the demo can carry on."""
    try:
        return loader(*args)
    except Exception as error:  # downloads and installs can fail in many ways; the demo reports it and moves on
        print(f"  [{label} unavailable: {type(error).__name__}: {str(error)[:90]}]")
        return None
