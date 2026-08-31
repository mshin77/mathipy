# mathipy 0.4.6

- Enlarges a wide, short figure, which the previous size test passed over.
- Caps the enlarged size and leaves an image alone when enlarging would shrink it.

# mathipy 0.4.5

- Enlarges an image below 512 pixels on its long side before a vision call, so a
  small answer-choice figure is read as a figure rather than as text.
- Leaves the instructional function empty when an image carries no figure, and
  states each function in the prompt.

# mathipy 0.4.4

- Counts one equation per equals sign, reads a hyphen as a minus only between
  numbers, and treats a fraction as a single value.
- Widens what counts as a reference into the image, changing counts.

# mathipy 0.4.3

- Reads equations stored as Word objects, which paragraph text leaves out.
- Adds public reading of paragraphs inside tables and insertion of operators an export dropped.

# mathipy 0.4.2

- Treats a classification that names nothing as unclassified rather than as a type.
- Leaves the primary type empty instead of labelling it other when nothing is named.

# mathipy 0.4.1

- Captures every sub-question in a multi-part item.
- Reports items read by the backup method.
- Retries an image when a call fails.

# mathipy 0.4.0

- Reads items, classroom talk, and student writing through one entry point.
- Measures words, numbers, and the relations they carry.
- Looks at the figure in an image, not the whole page.
- Finds shape types in the image.
- Writes math notation one way.

# mathipy 0.3.0

- Covers 26 visual types, up from 20.
- Labels each image by the role it plays in the item.
- Takes item text as context for the label.
- Merges repeated calls and groups types into six families.
- Tells a failed call apart from an image with no visual.

# mathipy 0.2.0

First public release.
