"""
Лингвистическая разметка текста через Stanza.
    1. Pipeline создаётся ОДИН РАЗ (не на каждый файл).
    2. download вызывается ОДИН РАЗ.
    3. Убран depparse — не нужен для текущих метрик (POS + N-граммы).
    4. batch_size увеличен для лучшей параллелизации на CPU.
    5. verbose=False — не тратит время на логи.
"""

from pathlib import Path
import stanza


# ---------- глобальный Pipeline
_NLP = None

def get_nlp():
    
    global _NLP

    if _NLP is None:
        stanza.download('ru', verbose=False)

        _NLP = stanza.Pipeline(
            'ru',
            processors='tokenize,pos,lemma',   # ← без depparse
            use_gpu=False,
            verbose=False,
            batch_size=64,                     # ← крупнее батч = лучше на CPU
            num_workers=2,                     # ← параллельная загрузка данных
        )

    return _NLP


def process_file(source: Path, destination: Path):
    """Размечает один файл и сохраняет результат."""
    text = source.read_text(encoding='utf-8')
    nlp = get_nlp()
    doc = nlp(text)

    result = []
    for sent_id, sent in enumerate(doc.sentences):
        for w in sent.words:
            result.append({
                "origin":  w.text,
                "lemma":   w.lemma,
                "pos":     w.upos,
                "sent_id": sent_id,
            })

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(str(result), encoding='utf-8')