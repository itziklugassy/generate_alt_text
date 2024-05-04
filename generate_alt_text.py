from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

# Function to load images from specified paths
def load_images(image_paths):
    images = []
    for path in image_paths:
        images.append(Image.open(path))
    return images

# Function to generate alt text using a pre-trained model
def generate_alt_text(images):
    # Load pre-trained model and tokenizer
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    alts = []
    for image in images:
        # Prepare image for the model
        inputs = feature_extractor(images=image, return_tensors="pt")
        # Generate description
        outputs = model.generate(**inputs)
        # Decode the generated ids to text description
        alt_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        alts.append(alt_text)
    return alts

# Main script
if __name__ == "__main__":
    # List of image paths
    image_paths = [
        r"C:\Users\itzik\Downloads\dt - test.jpg", 
        # Add more image paths as needed
    ]

    # Load images
    images = load_images(image_paths)

    # Generate alt texts
    alt_texts = generate_alt_text(images)

    # Print generated alt texts
    for alt in alt_texts:
        print(alt)
