# API Reference

## Readability

### `ReadabilityAnalyzer`

Class for analyzing text readability with math-aware normalization.

```python
from mathipy import ReadabilityAnalyzer

analyzer = ReadabilityAnalyzer()
result = analyzer.analyze("Solve for x: 2x + 5 = 15")
```

Returns a dictionary with:

flesch_reading_ease
: Flesch Reading Ease score (0–100)

flesch_kincaid_grade
: Flesch-Kincaid grade level

gunning_fog
: Gunning Fog index

smog_index
: SMOG index

automated_readability_index
: ARI score

coleman_liau_index
: Coleman-Liau index

linsear_write_formula
: Linsear Write formula

dale_chall_readability
: Dale-Chall readability score

average_grade_level
: Average of FK, Fog, and SMOG

low_confidence
: `True` if text is shorter than 20 words

## Math Content

### `MathContentAnalyzer`

Class for math content analysis and CCSSM domain classification.

```python
from mathipy import MathContentAnalyzer

analyzer = MathContentAnalyzer()
result = analyzer.analyze("What is the area of a triangle with base 6 and height 4?")
```

Returns a dictionary with:

pattern_matches
: Detected math patterns (equations, fractions, etc.)

symbol_counts
: Counts of math symbols by type

total_math_symbols
: Total math symbols found

numbers
: Extracted numbers with count, range, and properties

vocabulary
: Matched math terms and counts

domain_classification
: Primary domain, confidence, and scores

math_density
: Ratio of math patterns to word count

**Domain categories:** arithmetic, algebra, geometry, statistics, calculus, fractions

## Cognitive Load

### `CognitiveLoadEstimator`

Class for estimating cognitive load components.

```python
from mathipy import CognitiveLoadEstimator

estimator = CognitiveLoadEstimator()
result = estimator.estimate(text, readability_grade=5.2, math_terms=["equation", "solve"])
```

Returns a dictionary with:

intrinsic_cognitive_load
: Load from inherent item complexity (0–1)

extraneous_cognitive_load
: Load from presentation/language (0–1)

germane_cognitive_load
: Load from schema building (0–1)

total_cognitive_load
: Weighted total (0–1)

numeric_elements
: Count of numbers in text

variable_count
: Count of single-letter variables

operation_count
: Count of math operations

## Visual

### `VisualFeatureExtractor`

Class for extracting complexity features from assessment images.

Requires: `pip install mathipy[vision]`

```python
from mathipy import VisualFeatureExtractor

extractor = VisualFeatureExtractor()
features = extractor.extract("item_image.png")
```

Accepts a file path, `Path` object, or numpy array. Returns a dictionary with:

dimensions
: Width, height, aspect ratio, channels

pixel_statistics
: Mean, std, min, max, median, contrast

edge_metrics
: Canny edge ratio, Sobel/Laplacian statistics

structural_elements
: Detected lines, circles, shapes (triangles, rectangles, etc.)

frequency_domain
: Low/mid/high frequency energy ratios

complexity_score
: Overall score (0–1) and level (low/medium/high)

## OCR

### `MultimodalOCR`

Class for extracting text and math from images using vision LLMs.

Requires: `pip install mathipy[ocr]` and a `GEMINI_API_KEY` or `OPENAI_API_KEY`.

```python
from mathipy import MultimodalOCR

ocr = MultimodalOCR(provider="gemini")
result = ocr.extract("item_image.png")
```

Parameters:

provider
: `"gemini"` or `"openai"`

model
: Model name (defaults to `gemini-2.5-flash` or `gpt-4o`)

api_key
: API key (or set via `.env` file)

Accepts image path, URL, bytes, PDF, DOCX, or text file. Returns a dictionary with:

content_type
: `text_only`, `image_only`, or `mixed`

full_text
: All extracted text

image_description
: Visual content description

question_text
: Main question/problem statement

math_expressions
: List of LaTeX expressions

answer_choices
: Dictionary of answer choices

extraction_confidence
: Confidence score (0–1)

## Sample Data

```python
from mathipy.data import get_sample_csv, get_sample_image, list_sample_images

csv_path = get_sample_csv()
images = list_sample_images()
image_path = get_sample_image("2024-4M10 #2")
```
