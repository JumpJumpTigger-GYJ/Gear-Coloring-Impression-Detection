# Gear-Coloring-Impression-Detection
The entire pipeline of our model to detect the defects on gear by coloring impression
The entire algorithm consists of two parts:
1. A model called conditional FastGAN with a SimSiam Network (cFastGAN-SN) diversifying the generation to enlarge the small and imbalanced datasets
2. A new Vision Transformer called Dynamic Channel Token Vision Transformer (DCT-ViT) with linear computation complexity as a backbone for detection
