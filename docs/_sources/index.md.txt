# mathipy

Multimodal item feature extraction for K-12 math assessment. Analyze readability with math-aware normalization via [textstat](https://github.com/textstat/textstat) and [NLTK](https://www.nltk.org/), classify math content by [Common Core State Standards for Mathematics](https://www.thecorestandards.org/Math/) domain, estimate cognitive load components, extract visual complexity features from images using [OpenCV](https://opencv.org/) and [Pillow](https://pillow.readthedocs.io/), and perform multimodal optical character recognition (OCR) through [Gemini](https://ai.google.dev/) and [OpenAI](https://platform.openai.com/) vision APIs.

## Installation

```bash
pip install mathipy
```

With optional dependencies:

```bash
pip install mathipy[nlp]        # readability (textstat, nltk)
pip install mathipy[vision]     # visual analysis (opencv, pillow)
pip install mathipy[ocr]        # OCR via vision LLMs (httpx)
pip install mathipy[documents]  # document parsing (python-docx, pdfplumber)
pip install mathipy[all]        # all features
```

## Citation

- Shin, M. (2026). *mathipy: Multimodal item feature extraction for K-12 math assessment* (Python package version 0.1.2) \[Computer software\]. <a href="https://github.com/mshin77/mathipy">https://github.com/mshin77/mathipy</a>

```{toctree}
:hidden:

getting-started
vignettes/naep-demo
api
support
```
