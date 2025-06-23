# Food Image Retrieval using CNN
This project implements a **Content-Based Image Retrieval (CBIR)** system for food images using **Convolutional Neural Networks (CNN)** for feature extraction.

## Project Structure
- `data_weights_inference/`: Contains links to the dataset, extracted features files (.h5), and inference images.
- `source/`: Contains the main pipeline Jupyter Notebook and supporting Python scripts.
- `report/`: Project report in PDF and DOCX format and a poster.

## Methodology
- Images are passed through a pretrained CNN (VGG16 and ResNet50) to extract deep features.
- Features are stored and compared using a similarity.
- Given a query image, the top-k most visually similar food images are retrieved from the dataset.

## Dataset
- Food11 (a smaller version of Food101).
- Contains 11 classes, each with approximately 900 images.
- Link: 

## Evaluation
- Similarity metric: Cosine Similarity.
- Retrieval tested with top-k values (3, 5, 11, 21).
- Evaluation metric: Mean Average Precision (MAP) is used to measure retrieval quality.
- Procedure:
  - For each query image, the top-k most similar images are retrieved.
  - Precision is computed at each correct retrieval position.
  - Average Precision (AP) is calculated for each query.
  - The final MAP is the mean of all APs across the dataset.
- Results:
  - Check the `report/` folder.
 
## How to run
- You can run all steps inside the Jupyter Notebook (FoodImageRetrieval.ipynb) on Google Colab, which mounts your Google Drive and executes each script in the correct order.
- Before that, you need to install the source folder and upload it to Google Drive. Change the directory in the Jupyter Notebook file if necessary.
