"""
test_P2.py
Split a prompt into Role / Task / Examples / Rules / Padding / Context with a READY-MADE
zero-shot model. No LLM, no training, no keyword/regex rules.

How it works
  1. pysbd breaks the prompt into sentences (each line separately, so list items stay apart).
  2. A pretrained NLI model (Hugging Face "zero-shot-classification" pipeline) scores every
     sentence against a plain-English description of each label and picks the best one.
  3. Neighbouring sentences with the same label are joined into one part.

pip install transformers torch pysbd
Default model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli   (bigger: facebook/bart-large-mnli)
"""
import pysbd

# plain-English description of each label (what the NLI model checks each sentence against)
LABELS = {
    "Role":     "gives the assistant a persona, identity or area of expertise",
    "Task":     "asks the assistant to do or produce something",
    "Examples": "shows a sample answer to demonstrate the expected output",
    "Rules":    "sets a restriction or condition the answer must follow",
    "Padding":  "is a greeting, thanks or small talk",
    "Context":  "gives background information or the input data to work on",
}
NAMES = list(LABELS)
TEMPLATE = "This text {}."


class PromptSplitter:
    def __init__(self, model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        from transformers import pipeline
        self.seg = pysbd.Segmenter(language="en", clean=False)
        self.zs = pipeline("zero-shot-classification", model=model)
        self.hyp = {d: n for n, d in LABELS.items()}

    def _segment(self, text):
        out = []
        for line in text.splitlines():
            if line.strip():
                out += [s.strip() for s in self.seg.segment(line.strip()) if s.strip()]
        return out

    def _predict(self, units):
        res = self.zs(units, candidate_labels=list(self.hyp), hypothesis_template=TEMPLATE)
        res = [res] if isinstance(res, dict) else res
        return [(self.hyp[r["labels"][0]], r["scores"][0]) for r in res]

    def split(self, prompt):
        units = self._segment(prompt)
        out = {n: [] for n in NAMES}
        cur, buf = None, []
        for u, (p, _) in zip(units, self._predict(units) if units else []):
            if p == cur:
                buf.append(u)
            else:
                if buf:
                    out[cur].append(" ".join(buf))
                cur, buf = p, [u]
        if buf:
            out[cur].append(" ".join(buf))
        return out


def classify_prompt(prompt, splitter=None, **kw):
    splitter = splitter or PromptSplitter(**kw)
    result = splitter.split(prompt)
    print("=" * 70)
    for n in NAMES:
        print(f"\n{n.upper()}  (count = {len(result[n])})")
        for i, v in enumerate(result[n], 1):
            print(f"  {i}. {v}")
    print("\n" + "=" * 70)
    print("COUNTS:", {n: len(result[n]) for n in NAMES})
    return result


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    test = """hey, hope you’re doing good. so i have this customer feedback thing from our grocery delivery app and i need some help figuring out what people are saying lol.

basically can you read each feedback and tell me if its positive, negative or neutral. please only use those 3 words, dont write stuff like mixed or unknown or whatever.

also if someone complained about something, tell me the main issue but keep it short, like just a few words from what they actually said. if theres no complaint then just write None i guess.

oh and please dont make anything up ok? only use what is in the feedback. dont add advice or explanations or anything extra.

here are the feedbacks:

1. “The groceries arrived 25 minutes late and two items were missing.”
2. “The delivery person was very polite and the vegetables were fresh.”
3. “The app crashed during payment, but the order eventually went through.”
4. “The packaging was damaged, although the products inside were fine.”
5. “Everything arrived on time and the prices were reasonable.”

can you put it in a table or something so its easy to read? maybe with the id, whether its positive negative neutral, the main issue if any, and maybe a small quote from the feedback. keep the same order please.

thanks a lot, really appreciate it!"""
    classify_prompt(test, model=model)
