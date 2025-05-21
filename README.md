# Urdu News Article Classification

This project focuses on the classification of Urdu news articles into five predefined categories: Entertainment, Business, Sports, Science-Technology, and International. It demonstrates a complete machine learning workflow, from web scraping and data preprocessing to model implementation and evaluation.

The articles were scraped from three prominent Urdu news websites: Geo Urdu, Jang, and Express using Python's Beautiful Soup and requests libraries. The initial dataset underwent a cleaning process to handle missing values in titles and content, remove duplicate articles, and filter out less informative pieces to ensure a high-quality dataset of 1139 articles. Feature extraction was performed using the Bag of Words technique.

Three machine learning models were implemented and compared:
* **Multinomial Naive Bayes:** Utilized bigram features and Lidstone Smoothing (alpha=0.01).
* **Logistic Regression:** Implemented a One-vs-All approach for multi-class classification, optimized using Cross-Entropy Loss. The input features were a combined Bag of Words from titles and content.
* **Neural Network:** Featured an input layer corresponding to the combined feature vectors, a single hidden layer with 256 neurons using ReLU activation, and a softmax output layer. It was trained using the Adam optimizer and Cross-Entropy Loss.

## Table of Contents

- [Urdu News Article Classification](#urdu-news-article-classification)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Structure](#project-structure)
  - [Dataset](#dataset)
  - [Models](#models)
    - [Multinomial Naive Bayes](#multinomial-naive-bayes)
    - [Logistic Regression](#logistic-regression)
    - [Neural Network](#neural-network)
  - [Results](#results)
  - [Usage](#usage)
  - [Limitations](#limitations)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

Urdu, a morphologically rich language, presents unique challenges in text classification. This project showcases the application of machine learning to categorize Urdu news articles. The primary goal is to demonstrate manually constructed machine learning models and evaluate their accuracy in predicting the correct news categories.

The project includes:
- A `scraper.ipynb` notebook for data collection from news websites.
- Jupyter notebooks detailing the implementation of Logistic Regression, Multinomial Naive Bayes, and a Neural Network.
- A preprocessed dataset of 1139 Urdu news articles (`urdu_articles.csv`).

## Project Structure

The repository is organized as follows:
```
Urdu-Article-Classification/
├── data/
│   └── urdu_articles.csv    # Dataset containing Urdu news articles
├── notebooks/
│   ├── logistic_regression.ipynb    # Logistic Regression model
│   ├── multinomial_bayes.ipynb      # Multinomial Naive Bayes model
│   ├── neural_network.ipynb         # Neural Network model
│   └── scraper.ipynb                # Web scraping implementation
├── LICENSE
└── README.md
```

## Dataset

The dataset comprises 1139 Urdu news articles, scraped from Geo Urdu, Jang, and Express. These articles are categorized into:
- Entertainment
- Business
- Sports
- Science-Technology
- International

The data, stored in `data/urdu_articles.csv`, has been preprocessed by removing duplicates and articles with insufficient content, and handling missing values.

## Models

### Multinomial Naive Bayes
- Employs bigram features for enhanced contextual understanding in text classification.
- Implements Lidstone Smoothing (alpha=0.01) for improved performance.

### Logistic Regression
- Uses a One-vs-All strategy for multi-class classification.
- Optimized using Cross-Entropy Loss.
- Features derived from a combined Bag of Words of article titles and content.

### Neural Network
- Architecture: Input layer, one hidden layer (256 neurons, ReLU activation), and a softmax output layer.
- Trained using Adam optimizer and Cross-Entropy Loss.

## Results

On the dataset of 1139 preprocessed articles:
- **Neural Network:** Achieved the highest accuracy at 97.8%. This is attributed to its ability to learn complex non-linear patterns.
- **Logistic Regression:** 95.61% accuracy.
- **Multinomial Naive Bayes:** 95.59% accuracy.

The linear classifiers (Logistic Regression and Multinomial Naive Bayes) performed similarly, likely due to their comparable approaches to finding linear decision boundaries and the nature of the feature sets.

## Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/ibrahim-murtaza/urdu-article-classification.git](https://github.com/ibrahim-murtaza/urdu-article-classification.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd urdu-article-classification
    ```
3.  Run the notebooks:
    - Open `scraper.ipynb` to understand the data scraping process.
    - Explore `logistic_regression.ipynb`, `multinomial_bayes.ipynb`, or `neural_network.ipynb` for model implementations.

## Limitations

-   **Dataset Size:** While 1139 articles provide a good foundation, a larger and more diverse dataset could improve generalization.
-   **Morphological Richness of Urdu:** Current preprocessing and Bag of Words feature extraction might not fully capture the linguistic complexities of Urdu. More advanced NLP techniques like word embeddings or transformer-based models could offer improvements.
-   **Computational Complexity:** Neural Networks, despite higher accuracy, are more computationally intensive than Logistic Regression or Multinomial Naive Bayes.
-   **Domain Generalization:** The models are trained on news articles; performance might vary on other Urdu text domains like social media or literature.

## Contributing
Contributions are welcome! If you have improvements or suggestions, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.