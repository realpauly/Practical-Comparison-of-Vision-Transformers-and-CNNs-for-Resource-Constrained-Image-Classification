# Practical Comparison of Vision Transformers and CNNs for Resource Constrained Image Classification


## Introduction

Vision Transformers (ViTs) have emerged as a powerful alternative to traditional
Convolutional Neural Networks (CNNs) for image classification tasks. Most
studies on ViTs have explored their performance on large-scale datasets
requiring significant computational resources. This project aimed to compare the
practical implementation of Vision Transformers with one of the widely used CNN
models, ResNet-18, under resource-constrained conditions. The investigation was
inspired by the work of Dosovitskiy et al. (2021), which introduced a
transformer-based approach for image recognition, showcasing that a non-CNN
architecture could achieve competitive performance by treating image patches as
sequences.

The focus of this project was on the application of ViTs and CNNs to the
CIFAR-10 dataset, a widely used benchmark for image classification. The
comparison considered two key aspects: efficiency and accuracy. By analyzing
their performance in resource-limited environments, the study provided insights
into the practicality of these models in real-world applications where
computational and data resources are limited.

# Tools and Algorithm Description

## Vision Transformers (ViTs)

Vision Transformers (ViTs) are a novel deep learning architecture inspired by
the success of transformers in Natural Language Processing (NLP). Unlike
Convolutional Neural Networks (CNNs), ViTs split an image into fixed-size
patches, flatten each patch, and project them into a higher-dimensional space.
These patches are then processed as sequences, similar to words in NLP tasks,
using the self-attention mechanism. This architecture allows ViTs to model
long-range dependencies across an entire image, making them well-suited for
tasks requiring global context.

The architecture of ViTs consists of three main components: 1. **Patch
Embedding**: The image is divided into fixed-size patches, and each patch is
linearly projected into an embedding vector. Positional embeddings are added to
retain spatial information. 2. **Transformer Encoder**: A series of transformer
encoder layers applies self-attention and feed-forward operations to capture
relationships between image patches. 3. **Classification Head**: A learnable
`[CLS]` token is appended to the sequence, and its final representation is used
for classification tasks through a Multi-Layer Perceptron (MLP) head.

![Figure 1: Example of the architecture of ViTs (Source: Maurício et al.,
2023)](VIT.png)

This process is illustrated in Figure 1, showing how ViTs process images as
sequences of patches, enabling a global understanding of visual information.

## Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a class of deep learning models that
have been the backbone of computer vision tasks for over a decade. CNNs utilize
convolutional layers to extract spatial features from images by applying
learnable filters. These filters capture local patterns, such as edges and
textures, and hierarchically build up to more abstract features in deeper
layers.

The core components of CNNs include: 1. **Convolutional Layers**: Apply filters
to input images to extract local spatial features. 2. **Pooling Layers**: Reduce
the spatial dimensions of feature maps, typically using max-pooling or
average-pooling, to make computations efficient and capture essential patterns.
3. **Fully Connected Layers**: After flattening the feature maps, fully
connected layers are used to combine the extracted features and perform
classification tasks.

![Figure 2: Example of the architecture of CNNs (Source:Maurício et al.,
2023)](CNN.png)

This hierarchical approach allows CNNs to capture fine-grained details in images
while maintaining computational efficiency. Figure 2 illustrates a typical CNN
pipeline, highlighting the convolutional layers, pooling operations, and fully
connected layers for classification.

## Dataset and Preprocessing

The **CIFAR-10 dataset** was divided into training (50,000 images) and testing
(10,000 images) sets. Since the Vision Transformer requires fixed-size 224x224
image inputs, the images were resized during preprocessing. Data preprocessing
included:

### **Resizing**

All images were resized to 224x224 pixels for compatibility with ViTs and to maintain consistency across models.

### **Normalization** 

The images were normalized to have a mean of 0.5 and a standard deviation of 0.5 to stabilize the training process.

### **Augmentation** 

Techniques such as random cropping and flipping were applied to enhance model generalization.

The **torchvision.transforms** module was used for these preprocessing steps.
Data was loaded into memory using the PyTorch **DataLoader** with optimizations
such as `pin_memory` for faster data transfer to the GPU.

## Model Architectures

Two models were implemented: 1. **Vision Transformers (ViTs)**: - Utilized the
`vit_base_patch16_224` model from the `timm` library. - Modified the
classification head to output predictions for 10 classes. - Pre-trained weights
were leveraged to initialize the model for faster convergence.

2.  **ResNet-18**:
    -   A convolutional neural network with residual connections to enable
        deeper network training.
    -   The fully connected layer was replaced with a layer tailored for
        10-class output.
    -   Pre-trained weights were used for initialization.

These models represent two contrasting deep learning paradigms: ViTs excel at
modeling global relationships in images, while ResNet-18 relies on local feature
extraction through convolutions.

## Training Procedure

The training process was designed to ensure fairness and efficiency: 1.
**Hyperparameters**: - Batch size: 32 (chosen to accommodate GPU memory
constraints). - Learning rate: 0.001 (used for both models). - Number of epochs:
10. 2. **Loss Function**: - Cross-entropy loss was used for classification
tasks. 3. **Optimization**: - The Adam optimizer was employed for its ability to
adapt learning rates. 4. **Mixed Precision Training**: - Automatic Mixed
Precision (AMP) was enabled using `torch.cuda.amp` to reduce memory usage and
speed up computations. 5. **Device**: - Models were trained on a GPU when
available, falling back to a CPU otherwise.

During training, GPU memory utilization was logged to monitor resource usage.

## **ResNet-18 Performance**

-   **Loss**:
    -   Consistently decreased over the epochs, reaching **0.062** by the 10th
        epoch.
-   **Accuracy**:
    -   Started at **80.43%** in epoch 1.
    -   Peaked at **90.37%** before stabilizing at **89.61%**.

## **Vision Transformer (ViT) Performance**

-   **Loss**:
    -   Remained nearly constant throughout training, starting at **2.05** and
        ending at **1.95**.
-   **Accuracy**:
    -   Began at **25.75%** in epoch 1 and fluctuated slightly, concluding at
        **27.9%**.
-   These results highlight ViT's inefficiency in learning from small-scale
    datasets like CIFAR-10 without extensive pretraining.
# **Conclusion**

-   **ResNet-18**:
    -   Ideal for small datasets and resource-constrained environments due to
        its efficiency, simplicity, and inductive biases.
-   **ViT**:
    -   Highly dependent on large-scale data and pretraining, making it less
        suitable for tasks with limited data or compute resources.

For small datasets like CIFAR-10, **ResNet-18 remains the gold standard**, while
ViTs are better suited for large-scale tasks with access to extensive data and
computational resources.

# References

Maurício, J., Domingues, I., & Bernardino, J. (2023). Comparing Vision
Transformers and Convolutional Neural Networks for Image Classification: A
Literature Review. *Applied Sciences, 13*(9), 5521.
https://doi.org/10.3390/app13095521

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit,
J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image
recognition at scale. In Proceedings of the International Conference on Learning
Representations (ICLR). Google Research, Brain Team.
https://arxiv.org/abs/2010.11929
