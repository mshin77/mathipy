<img src="https://raw.githubusercontent.com/mshin77/mathipy/main/logo.svg" alt="mathipy Logo" align="right" width="220px"/>

[![PyPI version](https://img.shields.io/pypi/v/mathipy?cacheSeconds=0)](https://pypi.org/project/mathipy/)
[![Python versions](https://img.shields.io/pypi/pyversions/mathipy?cacheSeconds=0)](https://pypi.org/project/mathipy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Multimodal item feature extraction for K-12 math assessment. Reads item text and images through one entry point, with math-aware normalization via [textstat](https://github.com/textstat/textstat) and [NLTK](https://www.nltk.org/). Measure readability, [Common Core State Standards for Mathematics](https://www.thecorestandards.org/Math/) domain, cognitive load, and the relations carried by wording or by notation. Classify item images into visual types with an instructional role, read shape subtypes with [OpenCV](https://opencv.org/) and [Pillow](https://pillow.readthedocs.io/), and extract text through [Gemini](https://ai.google.dev/) and [OpenAI](https://platform.openai.com/) vision APIs.

## Installation

```bash
pip install mathipy
```

With optional dependencies:

```bash
pip install "mathipy[nlp]"        # readability (textstat, nltk)
pip install "mathipy[vision]"     # visual analysis (opencv, pillow)
pip install "mathipy[ocr]"        # OCR via vision LLMs (httpx)
pip install "mathipy[documents]"  # document parsing (python-docx, pdfplumber)
pip install "mathipy[all]"        # all features
```

From GitHub:

```bash
pip install "mathipy[all] @ git+https://github.com/mshin77/mathipy.git"
```

## Getting Started

See [Quick Start](https://mshin77.github.io/mathipy/getting-started.html) and [Analyzing Math Items](https://mshin77.github.io/mathipy/vignettes/item-analysis.html) for tutorials.

## Citation

- Shin, M. (2026). *mathipy: Multimodal item feature extraction for K-12 math assessment* (Python package version 0.4.4) [Computer software]. <https://github.com/mshin77/mathipy>
