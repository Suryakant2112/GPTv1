# Bigram Language Model & GPT-1 Implementation

## **1. Understanding the Bigram Language Model**
Before implementing a transformer-based model like GPT-1, I first learned about the **Bigram Language Model**, which captures simple token relationships based on probabilities.

### **What is a Bigram Model?**
A Bigram Language Model predicts the next token based **only on the previous token**. It learns the transition probabilities between character pairs and generates sequences based on these learned probabilities.

### **Key Concepts in the Bigram Model**
- **Token Embeddings:** Converts tokens (characters/words) into vector representations.
- **Logits:** Raw predictions for the next token.
- **Cross-Entropy Loss:** Measures how well predictions match actual data.
- **Sampling for Generation:** Predicts new tokens iteratively based on probabilities.

### **Limitations of the Bigram Model**
- **No Long-Term Memory:** It only considers one previous token.
- **Lack of Context:** It cannot capture relationships between distant tokens.
- **Not Scalable:** It fails to generate coherent and meaningful sequences for complex texts.

## **2. Moving to GPT-1**
To improve upon the Bigram Model, I implemented **GPT-1**, which utilizes the **Transformer architecture** with **self-attention** to handle long-range dependencies in text.

#### **Data Preprocessing**
- Loads a text dataset (Tiny Shakespeare corpus).
- Creates character-level token mappings (`stoi`, `itos` for encoding & decoding text).
- Splits data into **training (90%)** and **validation (10%)** sets.

#### **Model Components**
- **Token Embeddings:** Converts input characters into vector representations (`nn.Embedding`).
- **Positional Embeddings:** Adds position information since transformers do not have built-in sequence order.
- **Self-Attention Mechanism:**
  - **Key (K), Query (Q), Value (V):** Helps tokens attend to relevant parts of the sequence.
  - **Masked Attention:** Ensures autoregressive generation (future tokens can't be seen during training).
- **Multi-Head Attention:** Uses multiple attention heads to capture different aspects of context.
- **Feed-Forward Network:** A simple MLP applied after attention to transform embeddings.
- **Layer Normalization:** Stabilizes training and speeds up convergence.
- **Final Linear Layer:** Maps the final embeddings back to vocabulary space for token prediction.

#### **Training**
- Uses `AdamW` optimizer with learning rate `3e-4`.
- Trains the model for `5000` iterations.
- Evaluates loss every `500` steps for training & validation.
- Uses **cross-entropy loss** to measure prediction accuracy.

#### **Text Generation**
- **Autoregressive generation**: Predicts one token at a time, then feeds it back into the model.
- Uses **softmax & multinomial sampling** to pick the next token based on probabilities.
- Generates a sequence of 500 tokens from a given start token (`context`).

## **3. Why GPT-1 is Better Than the Bigram Model**
| Feature              | Bigram Model | GPT-1 |
|----------------------|-------------|-------|
| Context Length      | 1 Token     | 256 Tokens (block size) |
| Long-Term Dependency | ❌ No       | ✅ Yes (via Self-Attention) |
| Model Complexity    | Simple Lookup Table | Deep Neural Network |
| Generation Quality  | Random & Unstructured | More Coherent & Meaningful |

## **4. Running the GPT-1 Model**
### **Steps to Execute the Code:**
1. **Prepare the Dataset**
   - Download `input.txt` (Tiny Shakespeare dataset).
   - Encode the text into numerical tokens.
2. **Train the Model**
   - Run `GPTLanguageModel()` to initialize the transformer.
   - Train using `optimizer.step()` for `5000` iterations.
3. **Generate Text**
   - Use `generate()` function to produce new text based on learned patterns.
   - Decoded output is saved as a text file or printed to the console.

## **5. Future Improvements**
- **Use Larger Datasets** (e.g., books, Wikipedia) for better training.
- **Increase Model Size** (`n_embd`, `n_head`, `n_layer`) for richer representations.
- **Train on GPUs** to speed up computation.
- **Implement GPT-2 or GPT-3-like Features** (Layer Normalization, Bigger Context Window, Pretraining on Diverse Data).
