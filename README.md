<img src="https://raw.githubusercontent.com/mshin77/mathipy/main/docs_src/_static/logo.svg" alt="MathiPy Logo" align="right" width="220px"/>

[![PyPI version](https://img.shields.io/pypi/v/mathipy?v=0.1.1)](https://pypi.org/project/mathipy/)
[![Python versions](https://img.shields.io/pypi/pyversions/mathipy?v=0.1.1)](https://pypi.org/project/mathipy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

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

From GitHub:

```bash
pip install git+https://github.com/mshin77/mathipy.git[all]
```

## Getting Started

See [Quick Start](https://mshin77.github.io/mathipy/getting-started.html) and [Analyzing Math Items](https://mshin77.github.io/mathipy/vignettes/naep-demo.html) for tutorials.

## Citation

- Shin, M. (2026). *MathiPy: Multimodal item feature extraction for K-12 math assessment* (Python package version 0.1.1) [Computer software]. <https://mshin77.github.io/MathiPy>
