# License Plate Recognition System

## Introduction
This code aims to extract the text from an image of a license plate and match it with the respective city name. The code compares the results obtained from both `pytesseract` and `easyocr` libraries to recognize the text from the image.

## Prerequisites
Before running the code, make sure that you have the following libraries installed:
- `cv2`
- `re`
- `pytesseract`
- `PIL`
- `easyocr`
- `matplotlib`

## Code Overview
- The necessary libraries are imported in the code.
- The `dic` dictionary stores the mapping of the license plate codes to their respective city names.
- The code uses `cv2` to load the cascade classifier.
- The image is opened using the `Image` class from the `PIL` library and is passed to both `pytesseract` and `easyocr` for recognition of text.
- The recognized text is then matched with the respective city name using the `dic` dictionary.
- The result is plotted using `matplotlib` for visual comparison between the results obtained from both libraries.

## Running the Code
You can run the code by simply executing it in your Python environment. The image file used in this code should be in the same directory as the code.

## Conclusion
This code provides a basic implementation of license plate recognition system using `pytesseract` and `easyocr` libraries. The code can be further modified and optimized to achieve better results.
