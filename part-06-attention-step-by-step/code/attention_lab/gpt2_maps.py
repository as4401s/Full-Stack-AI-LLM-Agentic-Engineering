"""gpt2_maps.py - read the real attention weights of the small public model GPT-2.

What it does
    Loads GPT-2 (124 million parameters, 12 layers, 12 heads per layer) and returns, for any text, the tokens and a tensor of attention
    weights with shape (layers, heads, seq, seq). Also provides the sentence of the lesson and a few plain sentences for head statistics.

What it needs
    transformers and torch. The first run downloads about 550 MB of model files into the cache; later runs read the cache.

What to expect
    attention_for("the animal stayed home because it was tired") gives 8 tokens and a (12, 12, 8, 8) tensor of weights.
"""
import torch

SENTENCE = "the animal stayed home because it was tired"

# Plain sentences written for this lesson, used to see what each head usually does (no outside text is involved).
STATS_SENTENCES = [
    "the doctor told the patient that the results were ready",
    "after the long meeting the team went home because everyone was tired",
    "the nurse checked the chart and then called the pharmacy",
    "she put the book on the table because it was heavy",
    "the model reads every word before it writes the next one",
    "when the rain stopped the children ran outside to play",
    "the engineer fixed the bug and wrote a short test for it",
    "our hospital search finds notes that mention kidney failure",
    "the cat chased the mouse until it hid under the sofa",
    "he opened the window because the room was warm",
]

_CACHE = {}


def load_gpt2():
    """Load the tokenizer and the model once (eager attention, so the weights can be returned)."""
    if "model" not in _CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _CACHE["tok"] = AutoTokenizer.from_pretrained("gpt2")
        _CACHE["model"] = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
    return _CACHE["tok"], _CACHE["model"]


def attention_for(text):
    """Return (tokens, weights): tokens as readable strings and weights of shape (layers, heads, seq, seq)."""
    tok, model = load_gpt2()
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, output_attentions=True)
    weights = torch.stack([layer[0] for layer in out.attentions])      # (layers, heads, seq, seq)
    tokens = [t.replace("Ġ", "") for t in tok.convert_ids_to_tokens(ids["input_ids"][0])]   # the leading marker means "a space came before"
    return tokens, weights
