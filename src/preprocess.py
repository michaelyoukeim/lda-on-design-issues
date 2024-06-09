import os
import pandas as pd
import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import contractions
from config import ACCELERATOR_LIB_PATH
from collections import Counter

# Ensure NLTK data is available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Attempt to import the accelerator module
accelerator = None
if os.path.exists(ACCELERATOR_LIB_PATH):
    try:
        import sys
        sys.path.append(os.path.dirname(ACCELERATOR_LIB_PATH))
        import accelerator
        print("Accelerator module loaded successfully.")
    except ImportError as e:
        print(f"Failed to load accelerator module: {e}")
else:
    print(f"Accelerator module not found at {ACCELERATOR_LIB_PATH}.")


def concatenate_texts(summary: str, description: str) -> str:
    if pd.isna(summary):
        summary = ""
    if pd.isna(description):
        description = ""
    return summary + " " + description


def clean_text_with_rust(texts: pd.Series) -> pd.Series:
    if accelerator is None:
        print("Rust accelerator module not available. Returning empty text.")
        return pd.Series(["" for _ in texts])

    try:
        cleaned_texts = accelerator.bulk_clean_text_parallel(
            texts.tolist(), "remove", os.cpu_count())
        print(f"Cleaned {len(cleaned_texts)} texts using Rust accelerator.")
    except Exception as e:
        print(f"Error using Rust accelerator: {e}")
        return pd.Series(["" for _ in texts])

    # Replace dashes with spaces to keep words separate
    cleaned_texts = [re.sub(r'-', ' ', text) for text in cleaned_texts]

    return pd.Series(cleaned_texts)


def normalize_text(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def expand_contractions(text: str) -> str:
    return contractions.fix(text)


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(
        word, get_wordnet_pos(word)) for word in words]
    return ' '.join(lemmatized_words)


def remove_punctuation(text: str) -> str:
    # Tokenize the text to handle punctuation correctly
    words = nltk.word_tokenize(text)
    # Remove punctuation from each word
    words = [word for word in words if word not in string.punctuation]
    # Reassemble text with spaces between words
    return ' '.join(words)


def remove_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_issues(df: pd.DataFrame) -> pd.DataFrame:
    concatenated_texts = df.apply(lambda row: concatenate_texts(
        row['summary'], row['description']), axis=1)
    print("Step 1 - Concatenated Texts:")
    print(concatenated_texts.head())

    cleaned_texts = clean_text_with_rust(concatenated_texts)
    print("Step 2 - Cleaned Texts:")
    print(cleaned_texts.head())

    normalized_texts = cleaned_texts.apply(normalize_text)
    print("Step 3 - Normalized Texts:")
    print(normalized_texts.head())

    expanded_texts = normalized_texts.apply(expand_contractions)
    print("Step 4 - Expanded Contractions:")
    print(expanded_texts.head())

    lowercase_texts = expanded_texts.apply(to_lowercase)
    print("Step 5 - Lowercase Texts:")
    print(lowercase_texts.head())

    no_numbers_texts = lowercase_texts.apply(remove_numbers)
    print("Step 6 - No Numbers Texts:")
    print(no_numbers_texts.head())

    no_punctuation_texts = no_numbers_texts.apply(remove_punctuation)
    print("Step 7 - No Punctuation Texts:")
    print(no_punctuation_texts.head())

    no_stop_words_texts = no_punctuation_texts.apply(remove_stop_words)
    print("Step 8 - No Stop Words Texts:")
    print(no_stop_words_texts.head())

    lemmatized_texts = no_stop_words_texts.apply(lemmatize_text)
    print("Step 9 - Lemmatized Texts:")
    print(lemmatized_texts.head())

    final_texts = lemmatized_texts.apply(remove_extra_whitespace)
    print("Step 10 - Final Cleaned Texts:")
    print(final_texts.head())

    df['cleaned_text'] = final_texts
    return df


def create_vocabulary(df: pd.DataFrame) -> pd.DataFrame:
    all_words = ' '.join(df['cleaned_text']).split()
    vocab = Counter(all_words)

    vocab_df = pd.DataFrame(vocab.items(), columns=['token', 'count'])
    vocab_df = vocab_df.sort_values(
        by='count', ascending=False).reset_index(drop=True)
    return vocab_df
