import clip
import torch
from torchvision import transforms

import os
from PIL import Image

def load_and_convert_images(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Get a list of all files in the directory
    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Check if there are any image files in the directory
    if not image_files:
        print(f"No image files found in {directory_path}.")
        return

    # Define a transformation to convert the images to tensors
    transform = transforms.ToTensor()

    # Load and convert each image to a torch tensor
    image_tensors = []
    
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        
        try:
            # Open the image using PIL
            img = Image.open(image_path)

            # Apply the transformation to convert the image to a torch tensor
            
            img_tensor = transform(img).unsqueeze(0)

            # Print information about the converted image
            #print(f"Image: {image_file}, Size: {img.size}, Tensor Shape: {img_tensor.shape}")
            image_tensors.append(img_tensor)
            
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            
    all_samples = torch.cat(image_tensors, axis=0)
    
    return all_samples

class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0],  std=[2.0, 2.0, 2.0])] + clip_preprocess.transforms[:2] + clip_preprocess.transforms[4:])                                      

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()



class ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, gen_samples, src_images, target_text):
        
        gen_images_pts = load_and_convert_images(gen_samples)
        src_images_pts = load_and_convert_images(gen_samples)
        
        print(gen_images_pts.shape)
        print(src_images_pts.shape)
        
        sim_samples_to_img  = self.img_to_img_similarity(src_images_pts, gen_images_pts)

        sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), gen_images_pts)

        return sim_samples_to_img, sim_samples_to_text
