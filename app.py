import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms

class MSSM_SCDNet(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(MSSM_SCDNet, self).__init__()
        from transformers import SegformerForSemanticSegmentation
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/segformer-b0-finetuned-ade-512-512',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.cam = torch.nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sam = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, num_classes // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num_classes // 2, num_classes, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )
   
    def forward(self, x):
        outputs = self.segformer(x)
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam_out = self.cam(logits)
        sam_out = self.sam(logits)
        logits = logits * sam_out
        return logits

# Set device to CPU explicitly
device = torch.device('cpu')
segmenter = MSSM_SCDNet(num_classes=7).to(device)
segmenter.load_state_dict(torch.load("D:/WildfireProject/segmenter_retrained.pth", map_location=torch.device('cpu')))
segmenter.eval()

st.title("Wildfire Change Detection")
st.write("Upload a satellite image to generate a change map.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
   
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
   
    with torch.no_grad():
        seg_output = segmenter(image_tensor)
        seg_pred = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()
   
    class_dict = pd.read_csv("D:/WildfireProject/class_dict.csv")
    change_map = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
    for idx, row in class_dict.iterrows():
        color = tuple(row[['r', 'g', 'b']].values)
        if row['name'] == 'barren_land':
            change_map[seg_pred == idx] = [255, 255, 255]  # Burned: White
        elif row['name'] == 'forest_land':
            change_map[seg_pred == idx] = [0, 255, 0]      # Vegetation/Recovery: Green
        else:
            change_map[seg_pred == idx] = [0, 0, 0]        # Unburned/Other: Black
   
    st.image(change_map, caption='Change Detection Map', use_column_width=True)
   
    # Calculate percentages
    total_pixels = seg_pred.size
    burned_pixels = np.sum(seg_pred == class_dict[class_dict['name'] == 'barren_land'].index[0]) if 'barren_land' in class_dict['name'].values else 0
    vegetation_pixels = np.sum(seg_pred == class_dict[class_dict['name'] == 'forest_land'].index[0]) if 'forest_land' in class_dict['name'].values else 0
    unburned_pixels = total_pixels - (burned_pixels + vegetation_pixels)
    
    burned_percentage = (burned_pixels / total_pixels) * 100
    vegetation_percentage = (vegetation_pixels / total_pixels) * 100
    unburned_percentage = (unburned_pixels / total_pixels) * 100
    
    st.write(f"**Percentage Analysis:**")
    st.write(f"- Burned (White): {burned_percentage:.2f}%")
    st.write(f"- Vegetation/Recovery (Green): {vegetation_percentage:.2f}%")
    st.write(f"- Unburned/Other (Black): {unburned_percentage:.2f}%")
    
    # Color Legend
    st.write("**Color Legend:**")
    st.write("- White: Burned (Barren Land)")
    st.write("- Green: Vegetation/Recovery (Forest Land)")
    st.write("- Black: Unburned/Other (All other classes)")
   
    plt.imsave('change_map.png', change_map)
    with open('change_map.png', 'rb') as f:
        st.download_button('Download Change Map (PNG)', f, file_name='change_map.png')
