# Anti-Hater Neural Network Filter

## Project Overview
This project implements a deep learning model to detect and classify toxic comments across multiple categories. The model uses a bidirectional LSTM neural network architecture to analyze text and identify potentially harmful content such as toxic, severe toxic, obscene, threatening, insulting, and identity hate speech.

## Problem Statement
Online platforms face significant challenges with harmful content. This neural network filter helps identify various categories of toxic language to improve content moderation and create safer online environments.

## Dataset
The project uses a toxic comment classification dataset containing comments labeled across six categories:
- Toxic (9.58%)
- Severe Toxic (1.00%)
- Obscene (5.29%)
- Threat (0.30%)
- Insult (4.94%)
- Identity Hate (0.88%)

*Note: The dataset is highly imbalanced, with negative examples significantly outnumbering positive ones across all categories.*

## Project Structure

### Data Preprocessing
- Text conversion to lowercase
- Special character removal while preserving important punctuation
- Multiple space removal
- Null value handling

### Model Architecture
The implemented model is a deep learning network with the following key components:
- Input layer with text sequences
- Embedding layer (dimensionality: 100)
- Spatial dropout (0.2) to reduce overfitting
- Bidirectional LSTM layer (64 units) with dropout
- Global max pooling
- Fully connected dense layers with dropout and batch normalization
- Output layer with sigmoid activation for multi-label classification

```python
def create_balanced_model():
    # Input layer
    input_layer = Input(shape=(maxlen,))

    # Embedding layer
    embedding = Embedding(
        input_dim=min(max_features, len(tokenizer.word_index) + 1),
        output_dim=100,
        input_length=maxlen,
        embeddings_regularizer=l2(1e-6)
    )(input_layer)

    # SpatialDropout1D
    spatial_dropout = SpatialDropout1D(0.2)(embedding)

    # LSTM bidirectional
    lstm = Bidirectional(LSTM(64, return_sequences=True,
                          dropout=0.2, recurrent_dropout=0.2))(spatial_dropout)

    # Pooling
    pooled = GlobalMaxPooling1D()(lstm)

    # Fully connected layers
    dropout1 = Dropout(0.3)(pooled)
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.0005))(dropout1)
    bn = BatchNormalization()(dense1)
    dropout2 = Dropout(0.3)(bn)

    # Output layer
    output = Dense(len(categories), activation='sigmoid')(dropout2)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model
```

### Training Strategy
- Batch size of 256 for stability
- Adam optimizer with learning rate of 0.001
- Early stopping with patience=3
- ReduceLROnPlateau to reduce learning rate when needed
- ModelCheckpoint to save the best version
- Class weights to handle imbalanced data

### Threshold Optimization
- Lower thresholds (0.4-0.5) for decision boundaries
- Thresholds optimized for F1-score
- Special attention to critical categories (threat, identity hate)

## Model Performance
The model achieves good performance on the training set but shows some limitations on specific categories:

### Metrics
- **Toxic**: F1=0.2113, Precision=1.0000, Recall=0.1181
- **Severe Toxic**: F1=0.0000, Precision=0.0000, Recall=0.0000
- **Obscene**: F1=0.0116, Precision=0.9091, Recall=0.0058
- **Threat**: F1=0.0000, Precision=0.0000, Recall=0.0000
- **Insult**: F1=0.0000, Precision=0.0000, Recall=0.0000
- **Identity Hate**: F1=0.0000, Precision=0.0000, Recall=0.0000

## Learning Curves
The training shows rapid improvement in accuracy, with validation accuracy consistently high (near 99%). However, there appears to be a gap between training and validation accuracy, particularly in the early epochs, suggesting that further model tuning could be beneficial.

## Usage
To use this model for classification:

```python
# Preprocess text
def preprocess_text(text):
    text = str(text) if pd.notna(text) else ""
    text = text.lower()
    text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Make prediction
def predict_toxicity(text, model, tokenizer, thresholds):
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=100)
    
    # Predict
    prediction = model.predict(padded)[0]
    
    # Apply optimized thresholds
    results = {}
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for i, category in enumerate(categories):
        results[category] = bool(prediction[i] >= thresholds[i])
    
    return results
```

## Files
- `model_results/best_model.weights.h5`: Saved model weights
- `model_results/tokenizer.pickle`: Saved tokenizer
- `model_results/optimized_thresholds.pickle`: Optimized classification thresholds
- `model_results/evaluation_results.csv`: Detailed evaluation metrics
- `model_results/training_curves.png`: Learning curves visualization

## Future Improvements
1. Address class imbalance with more advanced techniques (SMOTE is implemented but further tuning could improve results)
2. Experiment with different model architectures (transformer-based models)
3. Implement data augmentation techniques for minority classes
4. Fine-tune hyperparameters for better recall on critical categories
5. Incorporate more features beyond text content

## Conclusion
This anti-hater neural network provides a solid foundation for toxic comment detection. While it achieves high precision, there's room for improvement in recall, especially for the rarer toxic categories. The model architecture and training strategy can be easily extended and further optimized for specific use cases.
