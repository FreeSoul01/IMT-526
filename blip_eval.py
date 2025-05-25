from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def generate_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption