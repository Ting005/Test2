## Solution #1 (Implemented)

We utilize vLLM for image understanding and experiment with Qwen/Qwen2.5-VL-7B-Instruct, achieving promising results.

## Requirements

To run Solution #1, ensure that the transformers package is installed.

### Steps

1. Download the model files and store them locally from the following link:Qwen/Qwen2.5-VL-7B-Instruct

2. Use the model path to initialize the Captcha class.

Solution #2 (Not Implemented)

We fine-tune a simple CNN model to recognize individual characters, classifying uppercase letters (A-Z), digits (0-9), and other characters.

Preparing the Training Dataset

1. Since "the font and spacing are the same each time," we can use a fixed width to segment the image into individual characters. Each segmented character, along with its label, forms a training instance.

2. Train a classifier using a CNN model.

Inference

1. Apply the same preprocessing technique to segment the input CAPTCHA image into individual characters.

2. Feed each segmented character into the trained classifier to obtain the prediction results.

