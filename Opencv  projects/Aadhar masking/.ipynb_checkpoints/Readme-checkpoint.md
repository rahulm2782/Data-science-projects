# Aadhar Card Masking

This code uses the following libraries:
1. cv2
2. re
3. easyocr
4. pytesseract
5. matplotlib

The code masks first 8 digits of  an image of an Aadhar card
The code first converts it to grayscale, and visualizes it using matplotlib.

The `easyocr` library is then used to read the text in the grayscale image. The text is filtered using confidence and length criteria, and the bounding box coordinates of the filtered text are stored.

Two rectangles are drawn on the original image to cover the filtered text, using the `cv2.rectangle()` method. The final image is visualized using matplotlib.

## Input
The input to the code is an image of an Aadhar card, located at the path `"aadhar-card-900-16509875443x2.jpg"`.

## Output
Aadhar card masked with its first 8 digit using an opaque layer