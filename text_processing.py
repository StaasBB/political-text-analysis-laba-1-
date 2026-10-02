from pathlib import Path
import stanza

def process_file(source: Path, destination: Path):
    stanza.download('ru')

    text = source.read_text(encoding='utf-8')

    nlp = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse')

    doc = nlp(text)
    result = []
    for sent in doc.sentences:
        for w in sent.words:
            result.append({"origin": w.text, "lemma": w.lemma, "pos": w.upos})

    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        str(result),
        encoding='utf-8'
    )