import os
import json
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np

p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception:
        return {}

def save_config(config):
    config_path = os.path.join(p, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

class ComfyUI_RanoBanano:
    def __init__(self, api_key=None):
        env_key = os.environ.get("GEMINI_API_KEY")

        # Common placeholder values to ignore
        placeholders = {"token_here", "place_token_here", "your_api_key",
                        "api_key_here", "enter_your_key", "<api_key>"}

        if env_key and env_key.lower().strip() not in placeholders:
            self.api_key = env_key
        else:
            self.api_key = api_key
            if self.api_key is None:
                config = get_config()
                self.api_key = config.get("GEMINI_API_KEY")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Generate a high-quality, photorealistic image",
                    "multiline": True,
                    "tooltip": "Describe what you want to generate or edit"
                }),
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {
                    "default": "generate",
                    "tooltip": "Choose the type of image operation"
                }),
            },
            "optional": {
                "model": ([
                    "gemini-2.5-flash-image",
                    "gemini-3-pro-image-preview",
                    "gemini-3.1-flash-image-preview",
                ], {
                    "default": "gemini-2.5-flash-image",
                    "tooltip": "Gemini model: 2.5 Flash (stable), 3 Pro (advanced reasoning), 3.1 Flash (latest, fastest)"
                }),
                "reference_image_1": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Primary reference image for editing/style transfer"
                }),
                "reference_image_2": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Second reference image (optional)"
                }),
                "reference_image_3": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Third reference image (optional)"
                }),
                "reference_image_4": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Fourth reference image (optional)"
                }),
                "reference_image_5": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Fifth reference image (optional)"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your Gemini API key (paid tier required)"
                }),
                "batch_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "tooltip": "Number of images to generate (costs multiply)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Creativity level (0.0 = deterministic, 1.0 = very creative)"
                }),
                "quality": (["standard", "high"], {
                    "default": "high",
                    "tooltip": "Image generation quality"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "1:1",
                    "tooltip": "Output image aspect ratio (passed natively to API)"
                }),
                "character_consistency": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Maintain character consistency across edits"
                }),

                "enable_safety": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable content safety filters"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "operation_log")
    FUNCTION = "rano_banano_generate"
    CATEGORY = "Rano Banano"
    DESCRIPTION = "Generate and edit images using Google Gemini image generation models. Requires paid API access."

    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor.cpu()
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor[0]

        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        return Image.fromarray(image_np, mode='RGB')

    def resize_image(self, image, max_size=2048):
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            return image.resize((new_width, new_height), Image.LANCZOS)
        return image

    def create_placeholder_image(self, width=512, height=512):
        """Create a placeholder image when generation fails"""
        img = Image.new('RGB', (width, height), color=(100, 100, 100))
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            draw.text((width//2-50, height//2), "Generation\nFailed", fill=(255, 255, 255))
        except Exception:
            pass

        image_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_array).unsqueeze(0)

    def prepare_images_for_api(self, img1=None, img2=None, img3=None, img4=None, img5=None):
        """Convert up to 5 tensor images to base64 format for API"""
        encoded_images = []

        for i, img in enumerate([img1, img2, img3, img4, img5], 1):
            if img is not None:
                if isinstance(img, torch.Tensor):
                    if len(img.shape) == 4:
                        pil_image = self.tensor_to_image(img[0])
                    else:
                        pil_image = self.tensor_to_image(img)

                    encoded_images.append(self._image_to_base64(pil_image))

        return encoded_images

    def _image_to_base64(self, pil_image):
        """Convert PIL image to base64 format for API"""
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(img_bytes).decode('utf-8')
            }
        }

    def build_prompt_for_operation(self, prompt, operation, has_references=False, character_consistency=True):
        """Build optimized prompt based on operation type"""

        base_quality = "Generate a high-quality, photorealistic image"

        if operation == "generate":
            if has_references:
                final_prompt = f"{base_quality} inspired by the style and elements of the reference images. {prompt}."
            else:
                final_prompt = f"{base_quality} of: {prompt}."

        elif operation == "edit":
            if not has_references:
                return "Error: Edit operation requires reference images"
            final_prompt = f"Edit the provided reference image(s). {prompt}. Maintain the original composition and quality while making the requested changes."

        elif operation == "style_transfer":
            if not has_references:
                return "Error: Style transfer requires reference images"
            final_prompt = f"Apply the style from the reference images to create: {prompt}. Blend the stylistic elements naturally."

        elif operation == "object_insertion":
            if not has_references:
                return "Error: Object insertion requires reference images"
            final_prompt = f"Insert or blend the following into the reference image(s): {prompt}. Ensure natural lighting, shadows, and perspective."

        if character_consistency and has_references:
            final_prompt += " Maintain character consistency and visual identity from the reference images."

        return final_prompt

    def call_rano_banano_api(self, prompt, encoded_images, model, temperature, aspect_ratio, batch_count, enable_safety):
        """Make API call to Gemini image generation model"""

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return [], "Package 'google-genai' is required. Install with: pip install google-generativeai\n"

        try:
            client = genai.Client(api_key=self.api_key)

            # Build generation config with native image_config
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                response_modalities=['Text', 'Image'],
                image_config=types.ImageGenerationConfig(
                    aspect_ratio=aspect_ratio,
                ),
            )

            # Set up content with proper encoding
            parts = [{"text": prompt}]

            for img_data in encoded_images:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_data["inline_data"]["data"]
                    }
                })

            content_parts = [{"parts": parts}]

            all_generated_images = []
            operation_log = ""

            for i in range(batch_count):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=content_parts,
                        config=generation_config
                    )

                    batch_images = []
                    response_text = ""

                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        response_text += part.text + "\n"

                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        try:
                                            image_binary = part.inline_data.data
                                            batch_images.append(image_binary)
                                        except Exception as img_error:
                                            operation_log += f"Error extracting image: {str(img_error)}\n"

                    if batch_images:
                        all_generated_images.extend(batch_images)
                        operation_log += f"Batch {i+1}: Generated {len(batch_images)} images\n"
                    else:
                        operation_log += f"Batch {i+1}: No images found. Text: {response_text[:100]}...\n"

                except Exception as batch_error:
                    operation_log += f"Batch {i+1} error: {str(batch_error)}\n"

            # Process generated images into tensors
            generated_tensors = []
            if all_generated_images:
                for img_binary in all_generated_images:
                    try:
                        image = Image.open(BytesIO(img_binary))

                        if image.mode != "RGB":
                            image = image.convert("RGB")

                        img_np = np.array(image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_np)[None,]
                        generated_tensors.append(img_tensor)
                    except Exception as e:
                        operation_log += f"Error processing image: {e}\n"

            return generated_tensors, operation_log

        except Exception as e:
            operation_log = f"API error: {str(e)}\n"
            return [], operation_log

    def rano_banano_generate(self, prompt, operation, model="gemini-2.5-flash-image",
                            reference_image_1=None, reference_image_2=None,
                            reference_image_3=None, reference_image_4=None, reference_image_5=None,
                            api_key="", batch_count=1, temperature=0.7, quality="high",
                            aspect_ratio="1:1", character_consistency=True, enable_safety=True):

        # Validate and set API key
        if api_key.strip():
            self.api_key = api_key
            save_config({"GEMINI_API_KEY": self.api_key})

        if not self.api_key:
            error_msg = "RANO BANANO ERROR: No API key provided!\n\n"
            error_msg += "Gemini image generation requires a PAID API key.\n"
            error_msg += "Get yours at: https://aistudio.google.com/app/apikey\n"
            error_msg += "Note: Free tier users cannot access image generation models."
            return (self.create_placeholder_image(), error_msg)

        try:
            # Process reference images (up to 5)
            encoded_images = self.prepare_images_for_api(
                reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5
            )
            has_references = len(encoded_images) > 0

            # Build optimized prompt
            final_prompt = self.build_prompt_for_operation(
                prompt, operation, has_references, character_consistency
            )

            if "Error:" in final_prompt:
                return (self.create_placeholder_image(), final_prompt)

            # Add quality instructions
            if quality == "high":
                final_prompt += " Use the highest quality settings available."

            # Log operation start
            operation_log = f"RANO BANANO OPERATION LOG\n"
            operation_log += f"Model: {model}\n"
            operation_log += f"Operation: {operation.upper()}\n"
            operation_log += f"Reference Images: {len(encoded_images)}\n"
            operation_log += f"Batch Count: {batch_count}\n"
            operation_log += f"Temperature: {temperature}\n"
            operation_log += f"Quality: {quality}\n"
            operation_log += f"Aspect Ratio: {aspect_ratio}\n"
            operation_log += f"Character Consistency: {character_consistency}\n"
            operation_log += f"Safety Filters: {enable_safety}\n"
            operation_log += f"Note: Output resolution determined by API (max ~1024px)\n"
            operation_log += f"Prompt: {final_prompt[:150]}...\n\n"

            # Make API call
            generated_images, api_log = self.call_rano_banano_api(
                final_prompt, encoded_images, model, temperature, aspect_ratio, batch_count, enable_safety
            )

            operation_log += api_log

            # Process results
            if generated_images:
                combined_tensor = torch.cat(generated_images, dim=0)

                approx_cost = len(generated_images) * 0.039
                operation_log += f"\nEstimated cost: ~${approx_cost:.3f}\n"
                operation_log += f"Successfully generated {len(generated_images)} image(s)!"

                return (combined_tensor, operation_log)
            else:
                operation_log += "\nNo images were generated. Check the log above for details."
                return (self.create_placeholder_image(), operation_log)

        except Exception as e:
            error_log = f"RANO BANANO ERROR: {str(e)}\n"
            error_log += "Please check your API key, internet connection, and paid tier status."
            return (self.create_placeholder_image(), error_log)

# Node registration
NODE_CLASS_MAPPINGS = {
    "ComfyUI_RanoBanano": ComfyUI_RanoBanano,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_RanoBanano": "Rano Banano (Gemini Image)",
}
