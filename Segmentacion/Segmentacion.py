import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def cancer_palette():
    return [
        [0, 0, 0],       
        [255, 0, 0],   
    ]


input_folder = "/home/anna/TFM/FINETUNE/data/CasosCáncer_241026"
output_root = "/home/anna/TFM/FINETUNE/segmentacion_resultados"
image_size = 512
palette = np.array(cancer_palette())

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

id2label = {0: "background", 1: "tumor"}
label2id = {v: k for k, v in id2label.items()}

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image_processor = SegformerImageProcessor(reduce_labels=False)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(input_folder, filename)
    image = Image.open(image_path).convert("RGB")

    inputs = image_processor(image, return_tensors="pt", size=(image_size, image_size)).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted = image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0].cpu().numpy()


    color_seg = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[predicted == label] = color


    img_np = np.array(image)
    blended = (img_np * 0.5 + color_seg * 0.5).astype(np.uint8)


    base_name = os.path.splitext(filename)[0]
    image_output_dir = os.path.join(output_root, base_name)
    os.makedirs(image_output_dir, exist_ok=True)


    image.save(os.path.join(image_output_dir, "original.jpg"))
    Image.fromarray(color_seg).save(os.path.join(image_output_dir, "mask.jpg"))
    Image.fromarray(blended).save(os.path.join(image_output_dir, "overlay.jpg"))

    print(f"✅ Procesado: {filename}")