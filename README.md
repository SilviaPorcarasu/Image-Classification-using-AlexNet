# Image Classification using AlexNet

This project explores the application of **AlexNet**, a Convolutional Neural Network (CNN), for classifying dog breeds from the **Stanford Dogs Dataset**. https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

## Dataset

A reduced version of the Stanford Dogs Dataset was used, limited to 8 classes:
- Golden Retriever
- Collie
- Doberman
- Saint Bernard
- Samoyed
- Pomeranian
- Chow Chow
- African Hunting Dog

Images were resized to `227x227` and normalized to match AlexNet's input requirements.

## Tools & Framework

- **Language**: MATLAB
- **Library**: Deep Learning Toolbox
- **Model**: Pre-trained AlexNet (transfer learning)

## Experiments

Three training configurations were tested:
1. **Clean dataset + SGDM optimizer**  
   - Train accuracy: 98.85%  
   - Validation accuracy: 95.91%  
   - Test accuracy: 97.96%  

2. **Noisy dataset + SGDM optimizer**  
   - Gaussian noise (σ = 0.05) added to images  
   - Train accuracy: 55.31%  
   - Validation accuracy: 39.88%  
   - Test accuracy: 40.21%  

3. **Clean dataset + Adam optimizer**  
   - Training stopped early due to time constraints  
   - Accuracy range: ~52-54% across all splits  

## Insights

- **SGDM** performed best on clean data.
- **Dataset quality** heavily influences model accuracy.
- **Adam optimizer** underperformed, likely due to incomplete training.

## Future Work

- Dynamic noise adjustment based on pixel histograms
- MSE-based performance tuning
- Spectrogram-based filtering to preserve high-salience features

## Author

**Silvia-Teodora Porcărașu**  
Faculty of Automatic Control and Computer Engineering "Gheorghe Asachi" Technical University of Iași

