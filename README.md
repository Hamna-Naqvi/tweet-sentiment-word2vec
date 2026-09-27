# Tweet Sentiment Analysis (LSTM + Word Embeddings)

A sentiment classification pipeline for tweets: text cleaning, tokenization, stemming, and a Keras `Embedding + LSTM` model trained to classify tweets as negative, neutral, or positive.

## Pipeline

1. **Load data** — reads `tweet_sentiment.csv` (columns: `cleaned_tweets`, `sentiment`).
2. **Handle missing values** — fills missing `cleaned_tweets` with the mode.
3. **Tokenize & stem** — uses NLTK's `word_tokenize` and `PorterStemmer`.
4. **Vectorize** — Keras `Tokenizer` converts text to integer sequences, padded to length 100.
5. **Label encoding** — remaps sentiment labels (`-1 → 2`) for 3-class classification (`0`, `1`, `2`).
6. **Model** — `Embedding(64) → LSTM(64, dropout=0.3) → Dense(3, softmax)`, trained with Adam + sparse categorical crossentropy, with early stopping on validation loss.

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

You'll also need the NLTK `punkt_tab` tokenizer data (downloaded automatically in the notebook via `nltk.download('punkt_tab')`).

## Usage

1. Place your `tweet_sentiment.csv` file in the project root (or update the path in the notebook).
2. Open and run `sentiment_lstm.ipynb` in Jupyter, Colab, or VS Code.

## Data

The dataset is not included in this repository. It should have at least two columns:

| column | description |
|---|---|
| `cleaned_tweets` | preprocessed tweet text |
| `sentiment` | sentiment label (`-1`, `0`, `1`) |

## Notes

- Originally developed in Google Colab (GPU: T4).
- Model architecture and hyperparameters are a starting point — feel free to tune embedding size, LSTM units, dropout, or swap in pretrained embeddings (e.g. real Word2Vec/GloVe vectors via `gensim`).
