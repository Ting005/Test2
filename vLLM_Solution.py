from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import torch

sys_prompt = """
Extract the CAPTCHA from this image.

** Rules **
1. Strictly 5 characters only.
2. Characters MUST be uppercase letters (A-Z) or digits (0-9).
    ** Character disambiguation: **
    - '0' (zero) NEVER becomes `O` (letter).
    - '1' (one) NEVER becomes `I` (letter) or `l` (letter).
    - 'O' (letter) MUST BE 'O', never convert to zero.
    - 'I' (letter) MUST BE 'I', never convert to one.
3. Output **ONLY** the 5-character sequence.
4. No explanations, punctuation, or formatting.

Example of valid output: XKMS2, VLI2C, OAHOV, O1R7Q, 605W1
"""


class Captcha:
    def __init__(self, model_path):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self, im_path, save_path):
        """
        Perform inference on the given image.
        
        Args:
            im_path (str): Path to the input image (.jpg) for inference.
            save_path (str): Path to save the output.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im_path},
                    {"type": "text", "text": sys_prompt},
                ],
            }
        ]

        # Prepare input for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]  # Extracting string from list

        # Save output to file
        with open(save_path, "w") as file:
            file.write(output_text)

        print(output_text)