import os
import json
import base64
import time
import traceback
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from .utilities import ColoredLogger

logger = ColoredLogger("RANO-BANANO")

p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        logger.debug(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Config loaded successfully. Keys present: {list(config.keys())}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {os.path.join(p, 'config.json')}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Config file contains invalid JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading config: {type(e).__name__}: {e}")
        return {}

def save_config(config):
    config_path = os.path.join(p, 'config.json')
    logger.debug(f"Saving config to: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info("Config saved successfully")

class ComfyUI_RanoBanano:
    def __init__(self, api_key=None):
        logger.info("=== Initializing ComfyUI_RanoBanano node ===")
        env_key = os.environ.get("GEMINI_API_KEY")

        # Common placeholder values to ignore
        placeholders = {"token_here", "place_token_here", "your_api_key",
                        "api_key_here", "enter_your_key", "<api_key>"}

        if env_key:
            logger.debug(f"Found GEMINI_API_KEY env variable (length={len(env_key)}, first 4 chars='{env_key[:4]}...')")
            if env_key.lower().strip() in placeholders:
                logger.warning(f"GEMINI_API_KEY env variable contains placeholder value: '{env_key}' - ignoring it")
            else:
                self.api_key = env_key
                logger.info("API key loaded from environment variable GEMINI_API_KEY")
                return
        else:
            logger.debug("No GEMINI_API_KEY environment variable found")

        self.api_key = api_key
        if self.api_key:
            logger.info(f"API key provided via constructor (length={len(self.api_key)})")
        else:
            logger.debug("No API key from constructor, trying config file...")
            config = get_config()
            self.api_key = config.get("GEMINI_API_KEY")
            if self.api_key:
                logger.info(f"API key loaded from config.json (length={len(self.api_key)})")
            else:
                logger.warning("No API key found in config.json either - API key is NOT set!")

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
        logger.debug(f"tensor_to_image: input tensor shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        tensor = tensor.cpu()
        if len(tensor.shape) == 4:
            logger.debug(f"tensor_to_image: 4D tensor detected, batch_size={tensor.shape[0]}")
            tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor[0]

        logger.debug(f"tensor_to_image: final tensor shape={tensor.shape}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        pil_image = Image.fromarray(image_np, mode='RGB')
        logger.debug(f"tensor_to_image: PIL image created, size={pil_image.size}, mode={pil_image.mode}")
        return pil_image

    def resize_image(self, image, max_size=2048):
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        logger.debug(f"resize_image: input size={width}x{height}, max_size={max_size}")
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            logger.info(f"resize_image: resizing from {width}x{height} to {new_width}x{new_height}")
            return image.resize((new_width, new_height), Image.LANCZOS)
        logger.debug(f"resize_image: no resize needed, image within max_size={max_size}")
        return image

    def create_placeholder_image(self, width=512, height=512):
        """Create a placeholder image when generation fails"""
        logger.warning(f"create_placeholder_image: creating placeholder {width}x{height} (generation failed)")
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
        logger.info("=== Preparing reference images for API ===")
        encoded_images = []

        for i, img in enumerate([img1, img2, img3, img4, img5], 1):
            if img is not None:
                logger.debug(f"Reference image #{i}: type={type(img).__name__}")
                if isinstance(img, torch.Tensor):
                    logger.debug(f"Reference image #{i}: tensor shape={img.shape}, dtype={img.dtype}")
                    if len(img.shape) == 4:
                        logger.debug(f"Reference image #{i}: 4D tensor, using first frame (batch dim={img.shape[0]})")
                        pil_image = self.tensor_to_image(img[0])
                    else:
                        pil_image = self.tensor_to_image(img)

                    logger.debug(f"Reference image #{i}: converted to PIL, size={pil_image.size}")
                    encoded = self._image_to_base64(pil_image)
                    encoded_images.append(encoded)
                    logger.info(f"Reference image #{i}: encoded successfully (base64 length={len(encoded['inline_data']['data'])})")
                else:
                    logger.warning(f"Reference image #{i}: unexpected type {type(img).__name__}, expected torch.Tensor - SKIPPING")
            else:
                logger.debug(f"Reference image #{i}: not provided (None)")

        logger.info(f"Total reference images encoded: {len(encoded_images)}")
        return encoded_images

    def _image_to_base64(self, pil_image):
        """Convert PIL image to base64 format for API"""
        logger.debug(f"_image_to_base64: encoding PIL image size={pil_image.size}, mode={pil_image.mode}")
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        b64_data = base64.b64encode(img_bytes).decode('utf-8')

        logger.debug(f"_image_to_base64: PNG bytes={len(img_bytes)}, base64 length={len(b64_data)}")

        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": b64_data
            }
        }

    def build_prompt_for_operation(self, prompt, operation, has_references=False, character_consistency=True):
        """Build optimized prompt based on operation type"""
        logger.info(f"=== Building prompt for operation: {operation} ===")
        logger.debug(f"build_prompt: has_references={has_references}, character_consistency={character_consistency}")
        logger.debug(f"build_prompt: raw prompt (first 200 chars): '{prompt[:200]}'")

        base_quality = "Generate a high-quality, photorealistic image"

        if operation == "generate":
            if has_references:
                final_prompt = f"{base_quality} inspired by the style and elements of the reference images. {prompt}."
            else:
                final_prompt = f"{base_quality} of: {prompt}."
            logger.debug(f"build_prompt: 'generate' operation, has_references={has_references}")

        elif operation == "edit":
            if not has_references:
                logger.error("build_prompt: 'edit' operation requires reference images but none provided!")
                return "Error: Edit operation requires reference images"
            final_prompt = f"Edit the provided reference image(s). {prompt}. Maintain the original composition and quality while making the requested changes."

        elif operation == "style_transfer":
            if not has_references:
                logger.error("build_prompt: 'style_transfer' operation requires reference images but none provided!")
                return "Error: Style transfer requires reference images"
            final_prompt = f"Apply the style from the reference images to create: {prompt}. Blend the stylistic elements naturally."

        elif operation == "object_insertion":
            if not has_references:
                logger.error("build_prompt: 'object_insertion' operation requires reference images but none provided!")
                return "Error: Object insertion requires reference images"
            final_prompt = f"Insert or blend the following into the reference image(s): {prompt}. Ensure natural lighting, shadows, and perspective."

        if character_consistency and has_references:
            final_prompt += " Maintain character consistency and visual identity from the reference images."
            logger.debug("build_prompt: character consistency directive appended")

        logger.info(f"build_prompt: final prompt length={len(final_prompt)} chars")
        logger.debug(f"build_prompt: final prompt (first 300 chars): '{final_prompt[:300]}'")
        return final_prompt

    def call_rano_banano_api(self, prompt, encoded_images, model, temperature, aspect_ratio, batch_count, enable_safety):
        """Make API call to Gemini image generation model"""
        logger.info("=" * 60)
        logger.info("=== STARTING API CALL TO GEMINI ===")
        logger.info("=" * 60)
        logger.info(f"API call params: model={model}, temperature={temperature}, aspect_ratio={aspect_ratio}")
        logger.info(f"API call params: batch_count={batch_count}, enable_safety={enable_safety}")
        logger.info(f"API call params: encoded_images count={len(encoded_images)}")
        logger.debug(f"API call params: prompt length={len(prompt)} chars")
        logger.debug(f"API call params: prompt (first 300 chars): '{prompt[:300]}'")

        try:
            logger.debug("Attempting to import google.genai...")
            from google import genai
            from google.genai import types
            logger.info(f"google.genai imported successfully. Version: {getattr(genai, '__version__', 'unknown')}")
        except ImportError as e:
            logger.critical(f"IMPORT ERROR: Cannot import google.genai: {e}")
            logger.critical("Install with: pip install google-generativeai")
            return [], "Package 'google-genai' is required. Install with: pip install google-generativeai\n"

        try:
            logger.debug(f"Creating genai.Client with API key (length={len(self.api_key)}, first 4='{self.api_key[:4]}...')")
            client = genai.Client(api_key=self.api_key)
            logger.info("genai.Client created successfully")

            # Build generation config with native image_config
            logger.debug("Building GenerateContentConfig...")
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                response_modalities=['Text', 'Image'],
                image_config=types.ImageGenerationConfig(
                    aspect_ratio=aspect_ratio,
                ),
            )
            logger.info(f"GenerateContentConfig built: temperature={temperature}, response_modalities=['Text', 'Image'], aspect_ratio={aspect_ratio}")

            # Set up content with proper encoding
            parts = [{"text": prompt}]
            logger.debug(f"Content parts: starting with text part (length={len(prompt)})")

            for idx, img_data in enumerate(encoded_images):
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_data["inline_data"]["data"]
                    }
                })
                logger.debug(f"Content parts: added inline image #{idx+1} (base64 length={len(img_data['inline_data']['data'])})")

            content_parts = [{"parts": parts}]
            logger.info(f"Content assembled: {len(parts)} parts total ({1} text + {len(encoded_images)} images)")

            all_generated_images = []
            operation_log = ""

            for i in range(batch_count):
                logger.info(f"--- Batch {i+1}/{batch_count}: sending API request ---")
                batch_start_time = time.time()
                try:
                    logger.debug(f"Batch {i+1}: calling client.models.generate_content(model='{model}')...")
                    response = client.models.generate_content(
                        model=model,
                        contents=content_parts,
                        config=generation_config
                    )
                    batch_elapsed = time.time() - batch_start_time
                    logger.info(f"Batch {i+1}: API response received in {batch_elapsed:.2f}s")

                    # Detailed response inspection
                    logger.debug(f"Batch {i+1}: response type={type(response).__name__}")
                    logger.debug(f"Batch {i+1}: response attributes={[a for a in dir(response) if not a.startswith('_')]}")

                    if hasattr(response, 'prompt_feedback'):
                        logger.debug(f"Batch {i+1}: prompt_feedback={response.prompt_feedback}")
                    if hasattr(response, 'usage_metadata'):
                        logger.debug(f"Batch {i+1}: usage_metadata={response.usage_metadata}")

                    batch_images = []
                    response_text = ""

                    if hasattr(response, 'candidates') and response.candidates:
                        num_candidates = len(response.candidates)
                        logger.info(f"Batch {i+1}: found {num_candidates} candidate(s) in response")

                        for c_idx, candidate in enumerate(response.candidates):
                            logger.debug(f"Batch {i+1}: candidate[{c_idx}] type={type(candidate).__name__}")
                            if hasattr(candidate, 'finish_reason'):
                                logger.debug(f"Batch {i+1}: candidate[{c_idx}] finish_reason={candidate.finish_reason}")
                            if hasattr(candidate, 'safety_ratings'):
                                logger.debug(f"Batch {i+1}: candidate[{c_idx}] safety_ratings={candidate.safety_ratings}")

                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                num_parts = len(candidate.content.parts)
                                logger.debug(f"Batch {i+1}: candidate[{c_idx}] has {num_parts} part(s)")

                                for p_idx, part in enumerate(candidate.content.parts):
                                    logger.debug(f"Batch {i+1}: candidate[{c_idx}].part[{p_idx}] type={type(part).__name__}, attributes={[a for a in dir(part) if not a.startswith('_')]}")

                                    if hasattr(part, 'text') and part.text:
                                        logger.info(f"Batch {i+1}: candidate[{c_idx}].part[{p_idx}] TEXT: '{part.text[:200]}'")
                                        response_text += part.text + "\n"

                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        logger.info(f"Batch {i+1}: candidate[{c_idx}].part[{p_idx}] IMAGE FOUND!")
                                        logger.debug(f"Batch {i+1}: inline_data type={type(part.inline_data).__name__}")
                                        logger.debug(f"Batch {i+1}: inline_data mime_type={getattr(part.inline_data, 'mime_type', 'N/A')}")
                                        try:
                                            image_binary = part.inline_data.data
                                            data_len = len(image_binary) if image_binary else 0
                                            logger.info(f"Batch {i+1}: extracted image binary data, size={data_len} bytes")
                                            if data_len == 0:
                                                logger.warning(f"Batch {i+1}: image binary data is EMPTY (0 bytes)!")
                                            else:
                                                batch_images.append(image_binary)
                                                logger.info(f"Batch {i+1}: image added to batch (total so far: {len(batch_images)})")
                                        except Exception as img_error:
                                            logger.error(f"Batch {i+1}: error extracting image data: {type(img_error).__name__}: {img_error}")
                                            logger.error(f"Batch {i+1}: traceback: {traceback.format_exc()}")
                                            operation_log += f"Error extracting image: {str(img_error)}\n"
                                    else:
                                        has_inline = hasattr(part, 'inline_data')
                                        has_text = hasattr(part, 'text')
                                        logger.debug(f"Batch {i+1}: candidate[{c_idx}].part[{p_idx}] - has_inline_data={has_inline}, has_text={has_text}")
                                        if has_inline and not part.inline_data:
                                            logger.warning(f"Batch {i+1}: candidate[{c_idx}].part[{p_idx}] - inline_data attribute exists but is None/falsy!")
                            else:
                                has_content = hasattr(candidate, 'content')
                                has_parts = hasattr(candidate.content, 'parts') if has_content else False
                                logger.warning(f"Batch {i+1}: candidate[{c_idx}] has no content.parts (has_content={has_content}, has_parts={has_parts})")
                                if has_content:
                                    logger.debug(f"Batch {i+1}: candidate[{c_idx}].content={candidate.content}")
                    else:
                        has_candidates = hasattr(response, 'candidates')
                        candidates_val = response.candidates if has_candidates else 'N/A'
                        logger.warning(f"Batch {i+1}: NO CANDIDATES in response! has_candidates={has_candidates}, candidates={candidates_val}")
                        logger.warning(f"Batch {i+1}: full response repr: {repr(response)[:500]}")

                    if batch_images:
                        all_generated_images.extend(batch_images)
                        logger.info(f"Batch {i+1}: SUCCESS - {len(batch_images)} image(s) generated (total: {len(all_generated_images)})")
                        operation_log += f"Batch {i+1}: Generated {len(batch_images)} images\n"
                    else:
                        logger.warning(f"Batch {i+1}: NO IMAGES generated!")
                        logger.warning(f"Batch {i+1}: response text was: '{response_text[:200]}'")
                        operation_log += f"Batch {i+1}: No images found. Text: {response_text[:100]}...\n"

                except Exception as batch_error:
                    batch_elapsed = time.time() - batch_start_time
                    logger.error(f"Batch {i+1}: EXCEPTION after {batch_elapsed:.2f}s: {type(batch_error).__name__}: {batch_error}")
                    logger.error(f"Batch {i+1}: full traceback:\n{traceback.format_exc()}")
                    operation_log += f"Batch {i+1} error: {str(batch_error)}\n"

            # Process generated images into tensors
            logger.info(f"=== Processing {len(all_generated_images)} generated image(s) into tensors ===")
            generated_tensors = []
            if all_generated_images:
                for img_idx, img_binary in enumerate(all_generated_images):
                    try:
                        logger.debug(f"Image {img_idx+1}: binary size={len(img_binary)} bytes, first 4 bytes={img_binary[:4] if len(img_binary) >= 4 else img_binary}")
                        image = Image.open(BytesIO(img_binary))
                        logger.debug(f"Image {img_idx+1}: PIL opened successfully, size={image.size}, mode={image.mode}, format={image.format}")

                        if image.mode != "RGB":
                            logger.debug(f"Image {img_idx+1}: converting from {image.mode} to RGB")
                            image = image.convert("RGB")

                        img_np = np.array(image).astype(np.float32) / 255.0
                        logger.debug(f"Image {img_idx+1}: numpy array shape={img_np.shape}, dtype={img_np.dtype}, min={img_np.min():.4f}, max={img_np.max():.4f}")
                        img_tensor = torch.from_numpy(img_np)[None,]
                        logger.debug(f"Image {img_idx+1}: tensor shape={img_tensor.shape}, dtype={img_tensor.dtype}")
                        generated_tensors.append(img_tensor)
                        logger.info(f"Image {img_idx+1}: converted to tensor successfully")
                    except Exception as e:
                        logger.error(f"Image {img_idx+1}: FAILED to convert: {type(e).__name__}: {e}")
                        logger.error(f"Image {img_idx+1}: traceback: {traceback.format_exc()}")
                        operation_log += f"Error processing image: {e}\n"
            else:
                logger.warning("No images to process - all_generated_images is empty")

            logger.info(f"=== API call complete: {len(generated_tensors)} tensor(s) produced from {len(all_generated_images)} binary image(s) ===")
            return generated_tensors, operation_log

        except Exception as e:
            logger.critical(f"FATAL API ERROR: {type(e).__name__}: {e}")
            logger.critical(f"Full traceback:\n{traceback.format_exc()}")
            operation_log = f"API error: {str(e)}\n"
            return [], operation_log

    def rano_banano_generate(self, prompt, operation, model="gemini-2.5-flash-image",
                            reference_image_1=None, reference_image_2=None,
                            reference_image_3=None, reference_image_4=None, reference_image_5=None,
                            api_key="", batch_count=1, temperature=0.7, quality="high",
                            aspect_ratio="1:1", character_consistency=True, enable_safety=True):

        logger.info("=" * 60)
        logger.info("=== RANO BANANO GENERATE - START ===")
        logger.info("=" * 60)
        logger.info(f"Parameters:")
        logger.info(f"  model            = {model}")
        logger.info(f"  operation        = {operation}")
        logger.info(f"  batch_count      = {batch_count}")
        logger.info(f"  temperature      = {temperature}")
        logger.info(f"  quality          = {quality}")
        logger.info(f"  aspect_ratio     = {aspect_ratio}")
        logger.info(f"  char_consistency = {character_consistency}")
        logger.info(f"  enable_safety    = {enable_safety}")
        logger.info(f"  prompt length    = {len(prompt)} chars")
        logger.debug(f"  prompt           = '{prompt[:300]}'")
        logger.info(f"  api_key provided = {bool(api_key.strip()) if api_key else False}")
        logger.info(f"  ref images: img1={'YES' if reference_image_1 is not None else 'no'}, "
                     f"img2={'YES' if reference_image_2 is not None else 'no'}, "
                     f"img3={'YES' if reference_image_3 is not None else 'no'}, "
                     f"img4={'YES' if reference_image_4 is not None else 'no'}, "
                     f"img5={'YES' if reference_image_5 is not None else 'no'}")

        # Validate and set API key
        if api_key.strip():
            logger.info(f"API key provided via node input (length={len(api_key.strip())}), updating stored key")
            self.api_key = api_key
            save_config({"GEMINI_API_KEY": self.api_key})

        if not self.api_key:
            logger.error("NO API KEY AVAILABLE! Cannot proceed with image generation.")
            logger.error("Possible sources checked: env var GEMINI_API_KEY, node input, config.json")
            error_msg = "RANO BANANO ERROR: No API key provided!\n\n"
            error_msg += "Gemini image generation requires a PAID API key.\n"
            error_msg += "Get yours at: https://aistudio.google.com/app/apikey\n"
            error_msg += "Note: Free tier users cannot access image generation models."
            return (self.create_placeholder_image(), error_msg)

        logger.info(f"API key is set (length={len(self.api_key)}, first 4='{self.api_key[:4]}...')")

        try:
            # Process reference images (up to 5)
            logger.info("--- Step 1: Processing reference images ---")
            encode_start = time.time()
            encoded_images = self.prepare_images_for_api(
                reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5
            )
            encode_elapsed = time.time() - encode_start
            has_references = len(encoded_images) > 0
            logger.info(f"Reference images encoded in {encode_elapsed:.2f}s: {len(encoded_images)} image(s), has_references={has_references}")

            # Build optimized prompt
            logger.info("--- Step 2: Building optimized prompt ---")
            final_prompt = self.build_prompt_for_operation(
                prompt, operation, has_references, character_consistency
            )

            if "Error:" in final_prompt:
                logger.error(f"Prompt building returned error: {final_prompt}")
                return (self.create_placeholder_image(), final_prompt)

            # Add quality instructions
            if quality == "high":
                final_prompt += " Use the highest quality settings available."
                logger.debug("High quality directive appended to prompt")

            logger.info(f"Final prompt ready (length={len(final_prompt)} chars)")

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
            logger.info("--- Step 3: Making API call ---")
            api_start = time.time()
            generated_images, api_log = self.call_rano_banano_api(
                final_prompt, encoded_images, model, temperature, aspect_ratio, batch_count, enable_safety
            )
            api_elapsed = time.time() - api_start
            logger.info(f"API call completed in {api_elapsed:.2f}s, returned {len(generated_images)} image tensor(s)")

            operation_log += api_log

            # Process results
            logger.info("--- Step 4: Processing results ---")
            if generated_images:
                logger.info(f"Concatenating {len(generated_images)} image tensors...")
                for t_idx, t in enumerate(generated_images):
                    logger.debug(f"  tensor[{t_idx}]: shape={t.shape}, dtype={t.dtype}")
                combined_tensor = torch.cat(generated_images, dim=0)
                logger.info(f"Combined tensor shape: {combined_tensor.shape}")

                approx_cost = len(generated_images) * 0.039
                operation_log += f"\nEstimated cost: ~${approx_cost:.3f}\n"
                operation_log += f"Successfully generated {len(generated_images)} image(s)!"

                logger.info(f"SUCCESS! Generated {len(generated_images)} image(s), estimated cost: ~${approx_cost:.3f}")
                logger.info("=== RANO BANANO GENERATE - COMPLETE ===")
                return (combined_tensor, operation_log)
            else:
                logger.warning("FAILURE: No images were generated!")
                logger.warning(f"API log output: {api_log}")
                operation_log += "\nNo images were generated. Check the log above for details."
                logger.info("=== RANO BANANO GENERATE - COMPLETE (no images) ===")
                return (self.create_placeholder_image(), operation_log)

        except Exception as e:
            logger.critical(f"FATAL ERROR in rano_banano_generate: {type(e).__name__}: {e}")
            logger.critical(f"Full traceback:\n{traceback.format_exc()}")
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
