# Suicide Intent Detection Using Convolutional Neural Networks

## Project Overview

This project implements a deep learning solution for detecting suicidal intent in Twitter messages using Convolutional Neural Networks (CNNs) and also processing the messages with NLP techniques. The system analyzes textual content from tweets to classify them as either indicating suicidal intent or not, serving as a potential tool for early intervention and mental health support.

## 🎯 Problem Statement

Mental health awareness and suicide prevention are critical issues in today's digital age. With millions of people expressing their thoughts and feelings on social media platforms like Twitter, there's an opportunity to identify individuals who may be experiencing suicidal thoughts. This project aims to develop an automated system that can detect potential suicide-related content in tweets, enabling timely intervention and support.

## 🗂️ Dataset

The project uses a Twitter dataset containing approximately **9,120 tweets** with binary classification labels:
- **Class 0:** Non-suicidal tweets
- **Class 1:** Tweets indicating suicidal intent

The dataset is well-balanced with nearly equal distribution between the two classes, making it suitable for binary classification tasks.

### Sample Data
```
Tweet: "my life is meaningless i just want to end my life so badly..."
Label: 1 (Suicidal Intent)

Tweet: "everything is okay but nothing feels okay..."
Label: 1 (Suicidal Intent)
```

## 🔧 Technical Architecture

### Data Preprocessing Pipeline

1. **Text Normalization:**
   - Convert text to lowercase
   - Remove punctuation and numbers
   - Remove URLs and usernames (@mentions)
   - Convert emojis to text representations

2. **Natural Language Processing:**
   - Tokenization using NLTK
   - Part-of-speech tagging
   - Lemmatization for word normalization
   - Stop word removal

3. **Feature Engineering:**
   - Sequence padding/truncation to fixed length (64 or 196 tokens)
   - Word2Vec embeddings (300-dimensional vectors)
   - Handling out-of-vocabulary (OOV) tokens with zero vectors

### Model Architecture

The CNN model employs a multi-kernel approach for comprehensive feature extraction:

#### Core Architecture:
- **Input Layer:** 300-dimensional Word2Vec embeddings
- **Convolutional Layers:** 
  - First layer: 3 parallel convolutions (kernel sizes 3, 5, 7) with 64 filters each
  - Second layer: 3 parallel convolutions (kernel sizes 3, 5, 7) with 128 filters each
- **Pooling:** Max pooling with kernel size 2
- **Fully Connected Layers:** 
  - Hidden layer: 128 neurons with ReLU activation
  - Output layer: 2 neurons (binary classification)

#### Model Variants Implemented:
1. **Base CNN Model** (Sequence length: 64)
2. **Extended Context Window CNN** (Sequence length: 196)
3. **Regularized CNN** (with Dropout and Batch Normalization)

### Training Configuration

```python
LEARNING_RATE = 4e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 64
EPOCHS = 400
SEQUENCE_LEN = 64 (or 196)
CNN_FILTERS = 64
```

## 🚀 Key Features

### Advanced Text Processing
- **Emoji Handling:** Converts emojis to textual representations
- **Hashtag Preservation:** Maintains hashtag content for context
- **Lemmatization:** Reduces words to their base forms
- **Comprehensive Cleaning:** Removes noise while preserving meaning

### Multi-Kernel CNN Design
- **Diverse Feature Extraction:** Uses kernels of different sizes (3, 5, 7) to capture various n-gram patterns
- **Hierarchical Learning:** Two-layer architecture for complex pattern recognition
- **Parallel Processing:** Concatenates features from multiple convolutions

### Regularization Techniques
- **Dropout:** 50% dropout rate to prevent overfitting
- **Batch Normalization:** Stabilizes training and improves convergence
- **Weight Decay:** L2 regularization for better generalization

## 📊 Experimental Results

### Model Performance Comparison

| Model Configuration | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|---------|----------|
| Base CNN (Adam) | ~85% | ~0.85 | ~0.85 | ~0.85 |
| Base CNN (SGD) | ~83% | ~0.83 | ~0.83 | ~0.83 |
| Extended Context (196) | ~87% | ~0.87 | ~0.87 | ~0.87 |
| Regularized CNN | ~85% | ~0.85 | ~0.85 | ~0.85 |

### Key Findings

1. **Optimizer Comparison:** Adam optimizer outperformed SGD with faster convergence and better final accuracy
2. **Context Window Impact:** Increasing sequence length from 64 to 196 tokens improved performance but increased computational cost
3. **Regularization Effects:** Dropout and batch normalization reduced overfitting and improved generalization

## 🛠️ Implementation Details

### Dependencies
```
torch>=1.9.0
pandas>=1.3.0
scikit-learn>=1.0.0
gensim>=4.0.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
numpy>=1.21.0
tqdm>=4.62.0
emoji>=1.6.0
```

### Project Structure
```
├── CA5.ipynb                    # Main Jupyter notebook
├── twitter-suicidal-data.csv    # Dataset
├── Description/
│   └── AI-CA5-Description.pdf   # Project requirements
└── README.md                    # This file
```

## 📈 Training Process

### Data Split
- **Training Set:** 90% (8,208 tweets)
- **Validation Set:** 10% (912 tweets)
- **Stratified Sampling:** Maintains class distribution

### Optimization Strategy
- **Loss Function:** CrossEntropyLoss for binary classification
- **Optimizer:** Adam with adaptive learning rates
- **Learning Schedule:** Fixed learning rate of 4e-3
- **Early Stopping:** Monitored validation loss for convergence

## 🔍 Detailed Analysis

### Text Preprocessing Insights

1. **Lowercasing Benefits:**
   - Reduces vocabulary size and ensures consistency
   - May lose information about proper nouns and acronyms
   - Justified for sentiment analysis tasks

2. **Number Removal:**
   - Eliminates noise from dates, IDs, and irrelevant numeric data
   - May lose contextual information in some cases
   - Appropriate for suicide detection task

3. **Hashtag Preservation:**
   - Maintains important contextual and emotional information
   - Hashtags often contain key sentiment indicators
   - Crucial for social media text analysis

### Word Embedding Strategy

**Word2Vec Approach:**
- Uses pre-trained Google News Word2Vec (300 dimensions)
- Handles OOV tokens with zero vectors
- Provides semantic understanding of words

**Alternative Approaches Considered:**
- Random vectors for OOV tokens
- Special OOV tokens
- Zero vectors (implemented choice)

### CNN Architecture Rationale

**Multi-Kernel Design:**
- **Small Kernels (3):** Capture local patterns and short phrases
- **Medium Kernels (5):** Detect mid-range dependencies
- **Large Kernels (7):** Identify longer contextual patterns

**Hierarchical Feature Learning:**
- First layer: Basic pattern detection
- Second layer: Complex feature combination
- Fully connected: Final classification decision

## 📚 Research Questions Addressed

### Q1: Text Preprocessing Effects
**Lowercasing Analysis:** Explored advantages (normalization, consistency) vs. disadvantages (information loss, acronym distortion)

### Q2: Number Removal Impact
**Numerical Data Handling:** Analyzed benefits (clarity, reduced dimensionality) vs. drawbacks (semantic loss, mixed data issues)

### Q3: Hashtag Retention Strategy
**Social Media Context:** Justified preserving hashtags for contextual information and sentiment indicators

### Q4: Out-of-Vocabulary Handling
**OOV Token Strategies:** Compared random vectors, fixed vectors, and special tokens approaches

### Q5: Optimizer Comparison
**Adam vs. SGD:** Detailed analysis of adaptive learning rates, gradient accumulation, and convergence properties

### Q6: Loss Function Selection
**CrossEntropy Rationale:** Explained probability distribution handling and confidence penalty mechanisms

### Q7: Train-Test Split Strategy
**90-10 Division:** Justified ratio based on dataset size and balance requirements

### Q8: Kernel Size Analysis
**Convolution Effects:** Examined feature extraction capabilities and spatial information processing

### Q9: Dimensionality Reduction
**Architecture Design:** Analyzed placement of reduction operations and feedforward layer advantages

### Q10: Context Window Impact
**Sequence Length Effects:** Evaluated computational vs. performance trade-offs

### Q11: Extended Context Results
**Performance Comparison:** Documented improvements and computational overhead

### Q12: Regularization Benefits
**Overfitting Prevention:** Assessed dropout and batch normalization effectiveness

## 🌟 Innovation and Contributions

1. **Multi-Scale Feature Extraction:** Novel combination of multiple kernel sizes for comprehensive pattern recognition
2. **Social Media Adaptation:** Specialized preprocessing pipeline for Twitter data
3. **Comparative Analysis:** Thorough evaluation of different architectural choices
4. **Practical Application:** Real-world mental health application with social impact

## 🚀 Future Enhancements

1. **Advanced Embeddings:** Implement BERT or GPT-based contextual embeddings
2. **Attention Mechanisms:** Add attention layers for better feature selection
3. **Ensemble Methods:** Combine multiple models for improved accuracy
4. **Real-time Deployment:** Develop API for live tweet monitoring
5. **Multi-class Extension:** Expand to detect different mental health conditions
6. **Temporal Analysis:** Incorporate time-series analysis for user behavior patterns

## 📖 Usage Instructions

### Setup Environment
```bash
pip install torch pandas scikit-learn gensim nltk matplotlib seaborn numpy tqdm emoji
```

### Download NLTK Data
```python
import nltk
nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])
```

### Run the Analysis
1. Open `CA5.ipynb` in Jupyter Notebook
2. Execute cells sequentially to reproduce results
3. Monitor training progress and validation metrics
4. Analyze confusion matrices and performance reports

### Model Training
```python
# Configure model parameters
model = CNN(input_dim=300, sequence_length=64, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=4e-3)
history = train_model(model, batch_size=64, epochs=400, ...)
```

## 🏆 Impact and Applications

### Mental Health Support
- **Early Detection:** Identifies at-risk individuals for timely intervention
- **Automated Screening:** Reduces manual monitoring burden on mental health professionals
- **Social Media Integration:** Leverages existing platforms for health monitoring

### Technical Contributions
- **NLP Pipeline:** Demonstrates effective text preprocessing for social media data
- **CNN Architecture:** Shows multi-kernel approach effectiveness for text classification
- **Comparative Study:** Provides insights into various deep learning design choices
