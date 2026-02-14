# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
In supervised learning, the algorithm learns from labeled training data. The goal is to map inputs to known outputs.
- **Classification**: Predicting categorical outcomes (spam vs. not spam)
- **Regression**: Predicting continuous values (house prices)
- Common algorithms: Linear Regression, Decision Trees, Random Forest, SVM, Neural Networks

### Unsupervised Learning
In unsupervised learning, the algorithm finds patterns in unlabeled data.
- **Clustering**: Grouping similar data points (K-Means, DBSCAN)
- **Dimensionality Reduction**: Reducing features while preserving information (PCA, t-SNE)
- **Association**: Finding rules in data (Apriori algorithm)

### Reinforcement Learning
An agent learns to make decisions by performing actions in an environment and receiving rewards or penalties. Used in game playing, robotics, and autonomous vehicles.

## Key Concepts

### Training and Testing
Data is typically split into training (70-80%), validation (10-15%), and test (10-15%) sets. The model learns from training data and is evaluated on test data it has never seen.

### Overfitting and Underfitting
- **Overfitting**: The model memorizes training data but fails on new data. Solutions: regularization, dropout, more data, simpler model.
- **Underfitting**: The model is too simple to capture patterns. Solutions: more features, complex model, less regularization.

### Evaluation Metrics
- **Accuracy**: Fraction of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **MSE/RMSE**: For regression tasks

## Neural Networks

Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons).

### Deep Learning
Deep learning uses neural networks with many layers (deep networks). Key architectures:
- **CNN** (Convolutional Neural Networks): Image recognition
- **RNN** (Recurrent Neural Networks): Sequential data
- **Transformer**: Natural language processing, the architecture behind modern LLMs
- **GAN** (Generative Adversarial Networks): Generating synthetic data

### The Transformer Architecture
The Transformer, introduced in "Attention Is All You Need" (2017), uses self-attention mechanisms to process sequences in parallel. It forms the basis of models like BERT, GPT, and Gemini.
