# RanaBanana

A ComfyUI custom node for Google's **Gemini** image generation models — supporting text-to-image, editing, style transfer, and object insertion.

> **Fork notice**: This project is a fork of [ComfyUI-NanoBanano](https://github.com/ShmuelRonen/ComfyUI-NanoBanano) by [ShmuelRonen](https://github.com/ShmuelRonen). Original work is licensed under MIT.

## What changed in this fork

- **Renamed** to **RanaBanana** (node, category, and display name)
- **Added detailed diagnostic logging** using `ColoredLogger` — every step of the pipeline (API key resolution, image encoding, prompt building, API request/response, tensor conversion) is logged to the ComfyUI console with color-coded severity levels, timings, and full tracebacks on errors
- **Multi-model support** — added `gemini-3-pro-image-preview` and `gemini-3.1-flash-image-preview` alongside the original `gemini-2.5-flash-image`
- **Branding** — Publicis Groupe Poland Technology ([SharePoint](https://publicisgroupe.sharepoint.com/sites/TechnologyPoland/))

## Features

- **Multi-Modal Operations**: Generate, edit, style transfer, and object insertion
- **Up to 5 Reference Images**: Support for complex multi-image operations
- **Character Consistency**: Maintain identity across edits and generations
- **Batch Processing**: Generate up to 4 images per request
- **Quality Control**: Temperature and quality settings
- **Aspect Ratio Support**: Multiple format options (1:1, 16:9, 9:16, 4:3, 3:4)
- **Cost Tracking**: Built-in cost estimation (~$0.039 per image)

## Requirements

- ComfyUI
- **Paid Google Gemini API Key** (Free tier does not support image generation)
- Python packages (installed automatically):
  - `google-generativeai`
  - `torch`
  - `pillow`
  - `numpy`
  - `requests`

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/nb3rt/RanaBanana.git
cd RanaBanana
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## API Key Setup

### 1. Get Your API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in and **enable billing** (paid tier required)
3. Generate API key (starts with `AIza...`)

### 2. Configure the Key

**Environment Variable (Recommended):**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Or enter directly in the node's `api_key` field.**

## Usage

1. **Find the Node**: Search **"RanaBanana"** in ComfyUI's node menu
2. **Select Operation**:
   - **Generate**: Create new images from text
   - **Edit**: Modify existing images
   - **Style Transfer**: Apply styles from references
   - **Object Insertion**: Add elements to scenes

3. **Key Parameters**:
   - `prompt`: Describe what you want
   - `model`: Choose Gemini model (`gemini-2.5-flash-image`, `gemini-3-pro-image-preview`, `gemini-3.1-flash-image-preview`)
   - `reference_image_1-5`: Upload reference images
   - `temperature`: Creativity (0.0–1.0)
   - `batch_count`: Images per run (1–4)
   - `aspect_ratio`: Output format (1:1, 16:9, 9:16, 4:3, 3:4)

## Examples

### Basic Generation
```
Operation: generate
Prompt: "A dragon flying over a cyberpunk city at sunset"
Aspect Ratio: 16:9
```

### Image Editing
```
Operation: edit
Reference Image: [Your photo]
Prompt: "Add falling snow and winter atmosphere"
```

### Style Transfer
```
Operation: style_transfer
Reference Image 1: [Content]
Reference Image 2: [Style reference]
Prompt: "Apply watercolor painting style"
```

## Troubleshooting

Check the **ComfyUI console/terminal** — RanaBanana logs every step with color-coded messages (`[RANO-BANANO|INFO]`, `[RANO-BANANO|ERROR]`, etc.).

**"API key not valid"**
- Ensure billing is enabled in Google Cloud Console
- Free tier cannot access image generation models

**"No images found in response"**
- Check console logs for `safety_ratings` and `finish_reason` — the request may be filtered
- Try more explicit prompts: "Generate an image of..."
- Check API rate limits and billing status

**Module errors**
```bash
pip install google-generativeai pillow torch numpy requests
```

## Cost Information

- **Per Image**: ~$0.039 USD
- **Batch of 4**: ~$0.156 USD
- Node displays cost estimates automatically

## License

MIT License — see [LICENSE](LICENSE) file for details.

## Credits

- **Original author**: [ShmuelRonen](https://github.com/ShmuelRonen) — [ComfyUI-NanoBanano](https://github.com/ShmuelRonen/ComfyUI-NanoBanano)
- **Fork maintainer**: [nb3rt](https://github.com/nb3rt) / [Publicis Groupe Poland Technology](https://publicisgroupe.sharepoint.com/sites/TechnologyPoland/)

---

**Note**: Unofficial implementation. Google and Gemini are trademarks of Google LLC.
