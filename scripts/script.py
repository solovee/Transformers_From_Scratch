import sentencepiece as spm
import unicodedata


def normalize_text(text):
    # remove acentos
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # deixa lowercase
    text = text.lower()
    # remove espaços extras
    text = ' '.join(text.split())
    return text

corpus = []
with open("pares_organizados.txt", "r", encoding="utf-8") as f:
    for line in f:
        en, pt = line.strip().split("->")
        corpus.append(normalize_text(en))
        corpus.append(normalize_text(pt))

with open("corpus.txt", "w", encoding="utf-8") as f:
    for sent in corpus:
        f.write(sent + "\n")


corpus = []
with open("pares_organizados.txt", "r", encoding="utf-8") as f:
    for line in f:
        en, pt = line.strip().split("->")
        corpus.append(normalize_text(en))
        corpus.append(normalize_text(pt))

with open("corpus.txt", "w", encoding="utf-8") as f:
    for sent in corpus:
        f.write(sent + "\n")
spm.SentencePieceTrainer.Train(
    '--input=corpus.txt --model_prefix=spm --vocab_size=8000 --character_coverage=1.0 --model_type=bpe'
)
sp = spm.SentencePieceProcessor()
sp.Load("spm.model")



with open("vocab.txt", "w", encoding="utf-8") as f:
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        f.write(f"{i} -> {piece}\n")

