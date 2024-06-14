# DiffDisEval
This is the repository of DiffDisEval, which benchmarks text-to-image diffusion models from a discriminative perspective. Specifically, benchmark models are applied to address the classic image segmentation problem by exploiting their attention maps produced by the denoising UNet.

## Dataset
The benmark dataset is develped based on existing image segmentation datasets: VOC 2012, Kavasir-SEG, Vaihingen and Cityscapes. Two versions are provided as follows:
1) [Large version](https://drive.google.com/file/d/1ItbFqsbLNhdlHS-nNAyC9wmYHj7vtbnA/view?usp=drive_link): 500 images from VOC 2012, 300 images from Kavasir-SEG, 150 images from Vaihingen and 50 images from Cityscapes (1000 images in total).
2) [Small version](https://drive.google.com/file/d/1yMMaUsKzOkP8mztDDqzBJLDNkxvpxIN_/view?usp=drive_link): 200 images from VOC 2012, 30 images from Kavasir-SEG, 20 images from Vaihingen and 10 images from Cityscapes (260 images in total).
