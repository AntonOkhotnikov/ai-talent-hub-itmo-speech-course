# Assignment 2. ASR Decoding - [20 pts]

In this exercise you are required to implement **4 CTC decoding strategies** for a pre-trained acoustic model and study the effect of language model integration, temperature scaling, and domain shift on recognition quality.

The acoustic model is [`facebook/wav2vec2-base-100h`](https://huggingface.co/facebook/wav2vec2-base-100h) — a [wav2vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2) model trained on 100 hours of [LibriSpeech](https://www.openslr.org/12).


## Description

You are provided with:
- [`Wav2Vec2Decoder`](wav2vec2decoder.py) — a class skeleton where you implement the decoding logic. **Calling any method from `self.processor` or `self.model` is prohibited** beyond what is already implemented for obtaining the logits matrix
- A pre-trained [3-gram KenLM language model](http://www.openslr.org/11/) trained on LibriSpeech text (`lm/3-gram.pruned.1e-7.arpa.gz`)
- [`examples/`](examples/) — a small set of short audio clips with reference transcripts, intended **for debugging only** (use them to quickly sanity-check your decoder output before running full evaluations)
- Evaluation test sets with ground truth transcripts:
    - `data/librispeech_test_other/` — **In-Domain** evaluation set (LibriSpeech `test-other`, 200 samples)
    - `data/earnings22_test/` — **Out-of-Domain** evaluation set ([Earnings22](https://huggingface.co/datasets/distil-whisper/earnings22), real-world financial earnings calls, 200 samples)
- `data/earnings22_train/corpus.txt` — a financial-domain text corpus (~5000 lines, ~100k words) extracted from Earnings22, provided as a starting point for training a KenLM language model in Task 8. You are encouraged to supplement it with additional financial text data of your own to improve the LM quality.


**Acoustic model vocabulary:**
```python
{0: '<pad>', 1: '<s>', 2: '</s>', 3: '<unk>', 4: '|', 5: 'E', 6: 'T', 7: 'A', 8: 'O', 9: 'N', 10: 'I', 11: 'H', 12: 'S', 13: 'R', 14: 'D', 15: 'L', 16: 'U', 17: 'M', 18: 'W', 19: 'C', 20: 'F', 21: 'G', 22: 'Y', 23: 'P', 24: 'B', 25: 'V', 26: 'K', 27: "'", 28: 'X', 29: 'J', 30: 'Q', 31: 'Z'}
```

- `<pad>` — blank symbol in CTC decoding
- `|` — word separator, interchangeable with space
- `<s>`, `</s>`, `<unk>` — **not used** by this model

All ground truth transcripts are **lowercase** with numbers written as words (e.g. "two thousand and twenty two"). Your decoder output must be lowercased before computing metrics. Use [CER](https://lightning.ai/docs/torchmetrics/stable/text/char_error_rate.html) and [WER](https://en.wikipedia.org/wiki/Word_error_rate) as metrics throughout (via the [jiwer](https://jitsi.github.io/jiwer/) package).


## Installation

Requires **Python >= 3.10, < 3.14** — `kenlm` is incompatible with Python 3.14+.

On Linux, first install `cmake`:

```bash
sudo sh -c 'apt-get update && apt-get upgrade && apt-get install cmake'
```

Then install all Python dependencies:

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install -r requirements.txt
```


## Tasks

### Part 1 — CTC Decoding

**Task 1.** Implement `greedy_decode` [[line 41]](wav2vec2decoder.py#41).

Evaluate on `data/librispeech_test_other/` and report CER & WER. Reference values: **WER ≈ 10.4%, CER ≈ 3.5%**

---

**Task 2.** Implement `beam_search_decode` [[line 54]](wav2vec2decoder.py#54).

Evaluate on `data/librispeech_test_other/` and report CER & WER. Reference values: **WER ≈ 9.9%, CER ≈ 3.4%**

Vary `beam_width` (e.g. 1, 3, 10, 50) and observe quality vs. compute trade-off, add corresponding graph/table to report.

---

**Task 3.** Implement **temperature scaling** for acoustic model outputs.

The `temperature` parameter is already wired into `Wav2Vec2Decoder.__init__`. Inside `decode()`:
```python
logits = logits / self.temperature   # already implemented — do NOT modify
log_probs = torch.log_softmax(logits, dim=-1)
```

Make sure your decoders use `log_probs` (not raw logits) computed this way.

Run a sweep over `T ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0}` on `data/librispeech_test_other/` using **greedy decoding only** and observe how WER changes.

Explain in your report what effect does temperature have on greedy decoding?

> The interaction between temperature and LM fusion is studied in Task 7 on out-of-domain data, where the effect is much more pronounced.


### Part 2 — Language Model Integration

**Task 4.** Implement `beam_search_with_lm` [[line 76]](wav2vec2decoder.py#76) — shallow fusion of the provided **3-gram LM**.

Evaluate on `data/librispeech_test_other/`. Reference values at best params: **WER ≈ 9.7%, CER ≈ 3.4%**

Run a sweep over `alpha ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0}` and `beta ∈ {0.0, 0.5, 1.0, 1.5}`. Report results in a heatmap or table and identify the best configuration.

The extended alpha range is intentional: at very low alpha the LM has no influence, at very high alpha it dominates and degrades quality. Note: the acoustic model is already strong in-domain, so the optimal alpha could be very small.

---

**Task 5.** Download the **4-gram LM** from [openslr.org/11](http://www.openslr.org/11/) and plug it into `beam_search_with_lm`.

Evaluate on `data/librispeech_test_other/` with the best `alpha`/`beta` from Task 4. Report results in a table alongside the 3-gram baseline.

---

**Task 6.** Implement `lm_rescore` [[line 95]](wav2vec2decoder.py#95) — second-pass LM rescoring of beam hypotheses.

Evaluate on `data/librispeech_test_other/`. Reference values at best params: **WER ≈ 9.6%, CER ≈ 3.3%**

Run a sweep over `alpha` and `beta` (same grid as Task 4: `alpha ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0}`, `beta ∈ {0.0, 0.5, 1.0, 1.5}`). Report results in a table. Compare rescoring vs shallow fusion — which is more stable to large `alpha` values and why?

Pick **5–10 samples** from `data/librispeech_test_other/` where at least one LM method changes the hypothesis relative to plain beam search, and present them as a qualitative comparison:

```
REF:  he had taken the wrong road entirely
BEAM: he had taken the rong road entirely
SF:   he had taken the wrong road entirely   ✓ corrected
RS:   he had taken the wrong road entirely   ✓ corrected

REF:  the committee met on thursday
BEAM: the committee met on thursday
SF:   the comittee met on thurzday          ✗ introduced error
RS:   the committee met on thursday         ✓ unchanged
```

In your report, look for patterns and answer:
- What kinds of errors does the LM tend to fix? (e.g. real-word confusions, rare words, sentence endings)
- What kinds of errors does it fail to fix or make worse? (e.g. acoustically similar but domain-mismatched words)
- Are there cases where shallow fusion and rescoring disagree? What does that reveal about the two methods?

---

**Task 7.** Evaluate your best shallow-fusion and rescoring configurations (from Tasks 4–6) on `data/earnings22_test/`.

Present a comparison table across all 4 decoding methods on both test sets:

| Method | LibriSpeech WER | LibriSpeech CER | Earnings22 WER | Earnings22 CER |
|---|---|---|---|---|
| Greedy | - | - | - | - |
| Beam search | - | - | — | — |
| Beam + 3-gram (shallow fusion) | - | - | — | — |
| Beam + 3-gram (rescoring) | - | - | — | — |

*(Fill in all values from your experiments.)*

Discuss the gap between in-domain and out-of-domain performance. Why does the LibriSpeech LM provide almost no benefit on financial speech?

**Task 7b.** Run a temperature sweep on `data/earnings22_test/` using your best shallow-fusion configuration from Task 4.

Sweep `T ∈ {0.5, 1.0, 1.5, 2.0}` and **plot WER vs T** for:
- Greedy decoding (reference: still flat — confirm this)
- Beam search with LM shallow fusion

Compare the resulting plot with the flat greedy curve from Task 3 (LibriSpeech). In your report, answer:
- Does higher temperature help or hurt LM fusion on out-of-domain speech, and why?
- On LibriSpeech the acoustic model is well-calibrated, so T > 1 degrades it. Is the same true for Earnings22? *(Hint: the acoustic model was never trained on financial speech — its confidence may be unreliable even at T = 1.)*

---

**Task 8.** Train a **financial-domain KenLM** model using `data/earnings22_train/corpus.txt` as your base training corpus (~5000 lines of real earnings-call speech).

You are free to extend the corpus with any additional financial text you can find (e.g. earnings call transcripts, financial news, SEC filings). More and more diverse data will produce a better LM and more interesting results in Task 9.

First, build the KenLM command-line tools (needed once — see the [KenLM docs](https://github.com/kpu/kenlm) for details):

```bash
# Linux
sudo apt-get install cmake libboost-all-dev
git clone --depth=1 https://github.com/kpu/kenlm /tmp/kenlm_build
mkdir /tmp/kenlm_build/build && cd /tmp/kenlm_build/build
cmake .. && make -j4 lmplz build_binary

# macOS
brew install cmake boost
git clone --depth=1 https://github.com/kpu/kenlm /tmp/kenlm_build
mkdir /tmp/kenlm_build/build && cd /tmp/kenlm_build/build
cmake .. && make -j4 lmplz build_binary
```

Then train a 3-gram model:

```bash
/tmp/kenlm_build/build/bin/lmplz -o 3 --discount_fallback \
    < data/earnings22_train/corpus.txt > /tmp/financial-3gram.arpa
gzip -c /tmp/financial-3gram.arpa > lm/financial-3gram.arpa.gz
```

The resulting `lm/financial-3gram.arpa.gz` can be loaded by `Wav2Vec2Decoder` via the `lm_model_path` argument.

---

**Task 9.** Apply your two best decoding methods using **both available LMs** (LibriSpeech 3-gram and your financial-domain LM) on both test sets.

Report all results in a table and add a bar chart comparing WER/CER per domain per LM. In your report, answer:
- Which LM works best in-domain? Out-of-domain?
- Does domain-matched LM help more than a larger general LM?


## Extra

Any other LM can be used for hypotheses rescoring (e.g. BERT-based neural LMs). Include your observations in the report if you explore this direction.


## Notes

- Don't forget to apply `torch.log_softmax()` to logits to get log probabilities
- Use Python's `heapq` module for storing most likely hypotheses during beam search
- Scores in beam search with LM: `score = log_p_acoustic + alpha * log_p_lm + beta * num_words`


## Resources
- [DLA course CTC decoding lecture slides](https://docs.google.com/presentation/d/1cBXdNIbowwYNp42WhJmd1Pp85oeslOrKNmGyZa5HKBQ/edit?usp=sharing)
- [HuggingFace wav2vec2 tutorial with n-gram LMs](https://huggingface.co/blog/wav2vec2-with-ngram)
- [KenLM training tutorial](https://github.com/kpu/kenlm)
- [Earnings22 dataset](https://huggingface.co/datasets/revdotcom/earnings22)
