# Getting Started

## Installation

```bash
pip install mathipy
```

With optional dependencies:

```bash
# Readability analysis (textstat, nltk)
pip install mathipy[nlp]

# Visual analysis (opencv, pillow)
pip install mathipy[vision]

# OCR via vision LLMs (httpx)
pip install mathipy[ocr]

# Document parsing (python-docx, pdfplumber)
pip install mathipy[documents]

# All features
pip install mathipy[all]
```

From GitHub:

```bash
pip install git+https://github.com/mshin77/mathipy.git[all]
```

## Basic Usage

### Readability Analysis

```python
from mathipy import ReadabilityAnalyzer

analyzer = ReadabilityAnalyzer()
result = analyzer.analyze("Solve for x: 2x + 5 = 15. What is the value of x?")

print(f"Grade Level: {result['flesch_kincaid_grade']:.1f}")
print(f"Reading Ease: {result['flesch_reading_ease']:.1f}")
```

### Math Content Analysis

```python
from mathipy import MathContentAnalyzer

analyzer = MathContentAnalyzer()
result = analyzer.analyze("Solve 2x + 5 = 15 for x")

print(f"Domain: {result['domain_classification']['primary']}")
print(f"Math density: {result['math_density']:.2f}")
```

### Cognitive Load Estimation

```python
from mathipy import CognitiveLoadEstimator

estimator = CognitiveLoadEstimator()
result = estimator.estimate("Solve 2x + 5 = 15 for x")

print(f"Total load: {result['total_cognitive_load']:.2f}")
print(f"Intrinsic: {result['intrinsic_cognitive_load']:.2f}")
```

### Visual Feature Extraction

```python
from mathipy import VisualFeatureExtractor

extractor = VisualFeatureExtractor()
features = extractor.extract("math_problem.png")

print(f"Complexity: {features['complexity_score']['level']}")
print(f"Shapes found: {features['structural_elements']['shapes']}")
```

### Multimodal OCR

Requires a `GEMINI_API_KEY` or `OPENAI_API_KEY` in your `.env` file.

```python
from mathipy import MultimodalOCR

ocr = MultimodalOCR(provider="gemini")
result = ocr.extract("math_problem.png")

print(result["full_text"])
print(result["math_expressions"])
```
