# Analyzing Math Items

This vignette demonstrates mathipy's feature extraction pipeline using released items from the National Assessment of Educational Progress (NAEP). mathipy includes a small sample of 5 NAEP mathematics items spanning grades 4 and 8, multiple years (2017–2024), and difficulty levels.

SOURCE: U.S. Department of Education, Institute of Education Sciences, National Center for Education Statistics, National Assessment of Educational Progress (NAEP), 2017, 2022, and 2024 Mathematics Assessments. Items obtained from the [NAEP Questions Tool](https://www.nationsreportcard.gov/nqt/).

## Setup

```bash
pip install mathipy[all]
```

## Load Sample Data

```python
import csv
from mathipy.data import get_sample_csv, get_sample_image, list_sample_images

with open(get_sample_csv()) as f:
    reader = csv.DictReader(f)
    items = list(reader)

for item in items:
    print(f"{item['item_id']} | Grade {item['grade']} | {item['difficulty']} | {item['content']}")
```

```
2024-4M10 #2 | Grade 4 | Easy      | Algebra
2017-4M1 #4  | Grade 4 | Medium    | Number Properties and Operations
2024-4M13 #2 | Grade 4 | Hard      | Measurement
2022-8M1 #2  | Grade 8 | Easy      | Geometry
2017-8M3 #2  | Grade 8 | Easy      | Data Analysis, Statistics, and Probability
```

Available sample images:

```python
print(list_sample_images())
```

```
['2017-4M1 #4.png', '2017-8M3 #2.png', '2022-8M1 #2.png',
 '2024-4M10 #2.png', '2024-4M13 #2.png']
```

## Step 1: Extract Text with OCR

Use `MultimodalOCR` to extract text and math expressions from item images. This requires a Gemini or OpenAI API key.

```python
from mathipy import MultimodalOCR

ocr = MultimodalOCR(provider="gemini")
image_path = str(get_sample_image("2024-4M10 #2"))
result = ocr.extract(image_path)

print(result["full_text"])
print(result["math_expressions"])
print(result["answer_choices"])
```

For this example, the extracted text is:

> Dwayne is D years old. Dwayne's sister is 3 years younger than Dwayne. Which expression represents Dwayne's sister's age in years? A) D+3 B) D-3 C) D×3 D) D÷3

## Step 2: Readability Analysis

Analyze extracted text with math-aware normalization. LaTeX and math symbols are replaced with placeholders so they don't inflate complexity scores.

```python
from mathipy import ReadabilityAnalyzer

analyzer = ReadabilityAnalyzer()
texts = {
    "2024-4M10 #2": "Dwayne is D years old. Dwayne sister is 3 years younger than Dwayne. "
                     "Which expression represents Dwayne sister age in years? "
                     "A) D+3 B) D-3 C) D*3 D) D/3",
    "2017-4M1 #4":  "Divide. 228 / 4 =",
    "2022-8M1 #2":  "The seven points shown on the map represent towns. "
                     "The distances, in kilometers, along roads between towns are given. "
                     "Tamara wants to travel from her town to Brook town in the shortest distance. "
                     "On which roads should Tamara travel? "
                     "Select the appropriate roads to show your answer.",
}

for item_id, text in texts.items():
    result = analyzer.analyze(text)
    print(f"{item_id}: FK Grade={result['flesch_kincaid_grade']:.1f}, "
          f"Reading Ease={result['flesch_reading_ease']:.1f}")
```

```
2024-4M10 #2: FK Grade=1.9, Reading Ease=94.5
2017-4M1 #4:  FK Grade=1.3, Reading Ease=91.0
2022-8M1 #2:  FK Grade=5.9, Reading Ease=69.5
```

The grade 8 geometry item (2022-8M1 #2) has higher linguistic complexity — longer sentences describing a map with distances — while the grade 4 arithmetic item (2017-4M1 #4) is a minimal two-word prompt.

## Step 3: Math Content Analysis

Classify items by CCSSM math domain and extract math features.

```python
from mathipy import MathContentAnalyzer

math_analyzer = MathContentAnalyzer()
for item_id, text in texts.items():
    result = math_analyzer.analyze(text)
    domain = result["domain_classification"]
    print(f"{item_id}: domain={domain['primary']} "
          f"(confidence={domain['confidence']:.2f}), "
          f"density={result['math_density']:.2f}, "
          f"terms={result['vocabulary']['math_terms']}")
```

```
2024-4M10 #2: domain=algebra    (confidence=1.00), density=0.31, terms=['expression']
2017-4M1 #4:  domain=arithmetic (confidence=1.00), density=0.20, terms=['divide']
2022-8M1 #2:  domain=arithmetic (confidence=0.50), density=0.00, terms=['point', 'even']
```

The algebra item is correctly classified with high confidence. The geometry item (2022-8M1 #2) has low math density because its complexity is visual (a map diagram), not textual.

## Step 4: Cognitive Load Estimation

Estimate intrinsic, extraneous, and germane cognitive load components.

```python
from mathipy import CognitiveLoadEstimator

estimator = CognitiveLoadEstimator()
for item_id, text in texts.items():
    result = estimator.estimate(text)
    print(f"{item_id}: total={result['total_cognitive_load']:.3f} "
          f"(intrinsic={result['intrinsic_cognitive_load']:.3f}, "
          f"extraneous={result['extraneous_cognitive_load']:.3f}, "
          f"germane={result['germane_cognitive_load']:.3f})")
```

```
2024-4M10 #2: total=0.478 (intrinsic=0.621, extraneous=0.466, germane=0.300)
2017-4M1 #4:  total=0.391 (intrinsic=0.800, extraneous=0.138, germane=0.100)
2022-8M1 #2:  total=0.273 (intrinsic=0.000, extraneous=0.610, germane=0.300)
```

The arithmetic item (2017-4M1 #4) has high intrinsic load (numbers and division) but very low extraneous load (minimal text). The geometry item has the opposite pattern — high extraneous load from lengthy instructions but no numeric elements in the text itself.

## Step 5: Visual Feature Extraction

Extract image complexity features from the item screenshots.

```python
from mathipy import VisualFeatureExtractor

extractor = VisualFeatureExtractor()
for item in items[:3]:
    image_path = str(get_sample_image(item["item_id"]))
    features = extractor.extract(image_path)
    dims = features["dimensions"]
    score = features["complexity_score"]
    edges = features["edge_metrics"]
    shapes = features["structural_elements"]
    print(f"{item['item_id']}: "
          f"{dims['width']}x{dims['height']}, "
          f"complexity={score['level']} ({score['overall']:.3f}), "
          f"lines={shapes['line_count']}, "
          f"shapes={shapes['total_shapes']}")
```

```
2024-4M10 #2: 520x390,  complexity=medium (0.506), lines=39, shapes=1
2017-4M1 #4:  872x147,  complexity=medium (0.372), lines=0,  shapes=1
2024-4M13 #2: (measurement item with visual content)
```

## Combining Features

The full pipeline produces a feature vector per item that can be used for research:

```python
import csv
from mathipy import VisualFeatureExtractor
from mathipy.data import get_sample_csv, get_sample_image

extractor = VisualFeatureExtractor()

with open(get_sample_csv()) as f:
    items = list(csv.DictReader(f))

results = []
for item in items:
    image_path = str(get_sample_image(item["item_id"]))
    visual = extractor.extract(image_path)

    results.append({
        "item_id": item["item_id"],
        "grade": item["grade"],
        "difficulty": item["difficulty"],
        "content": item["content"],
        "complexity": visual["complexity_score"]["overall"],
        "complexity_level": visual["complexity_score"]["level"],
        "lines": visual["structural_elements"]["line_count"],
        "shapes": visual["structural_elements"]["total_shapes"],
    })

for r in results:
    print(f"{r['item_id']:16s} | {r['difficulty']:6s} | "
          f"complexity={r['complexity_level']:6s} ({r['complexity']:.3f}) | "
          f"lines={r['lines']:3d}")
```

Text-based features (readability, math content, cognitive load) can be added after OCR extraction, building a complete feature table suitable for statistical analysis.

