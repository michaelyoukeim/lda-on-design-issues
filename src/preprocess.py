import os
import sys
import pandas as pd
import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import contractions
from config import ACCELERATOR_LIB_PATH, DEFAULT_DATA_PATH
from collections import Counter

# Download necessary NLTK resources at the start of your script
nltk_resources = ['stopwords', 'wordnet',
                  'punkt', 'averaged_perceptron_tagger']

for resource in nltk_resources:
    nltk.download(resource, quiet=True)


class TextAccelerator:
    def __init__(self):
        self.accelerator = self.initialize_accelerator()

    def initialize_accelerator(self):
        if os.path.exists(ACCELERATOR_LIB_PATH):
            try:
                sys.path.append(os.path.dirname(ACCELERATOR_LIB_PATH))
                import accelerator
                print("Accelerator module loaded successfully.")
                return accelerator
            except ImportError as e:
                print(f"Failed to load accelerator module: {e}")
        else:
            print(f"Accelerator module not found at {ACCELERATOR_LIB_PATH}.")
        return None

    def clean_text_with_rust(self, texts: pd.Series) -> pd.Series:
        if self.accelerator is None:
            print("Rust accelerator module not available. Returning empty text.")
            return pd.Series(["" for _ in texts])

        try:
            cleaned_texts = self.accelerator.bulk_clean_text_parallel(
                texts.tolist(), "remove", os.cpu_count())
            print(
                f"Cleaned {len(cleaned_texts)} texts using Rust accelerator.")
        except Exception as e:
            print(f"Error using Rust accelerator: {e}")
            return pd.Series(["" for _ in texts])

        return pd.Series(cleaned_texts)


def concatenate_texts(summary: str, description: str) -> str:
    if pd.isna(summary):
        summary = ""
    if pd.isna(description):
        description = ""
    return summary + " " + description


def normalize_text(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def expand_contractions(text: str) -> str:
    return contractions.fix(text)


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_special_characters(text: str) -> str:
    # Remove dashes at the start and end of each token
    text = text.strip('-')
    # Remove special characters except alphanumeric, spaces, and dashes
    return re.sub(r'[^a-zA-Z0-9\s-]', '', text)


def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                "V": wordnet.VERB, "R": wordnet.ADV}
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


def replace_ontology_classes(text: str) -> str:
    ontology_classes = pd.read_excel(os.path.join(
        DEFAULT_DATA_PATH, 'ontology_sheet_1.xlsx'))
    replaced_text = text
    for column in ontology_classes.columns:
        for index, value in ontology_classes[column].dropna().items():
            replaced_text = re.sub(r'\b{}\b'.format(
                re.escape(value.strip())), column, replaced_text)
    return replaced_text


def preprocess_issues(df: pd.DataFrame, accelerator: TextAccelerator) -> pd.DataFrame:
    concatenated_texts = df.apply(lambda row: concatenate_texts(
        row['summary'], row['description']), axis=1)
    print("Step 1 - Concatenated Texts:")
    print(concatenated_texts.head())

    cleaned_texts = accelerator.clean_text_with_rust(concatenated_texts)
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

    no_special_chars_texts = lowercase_texts.apply(remove_special_characters)
    print("Step 6 - No Special Characters Texts:")
    print(no_special_chars_texts.head())

    no_numbers_texts = no_special_chars_texts.apply(remove_numbers)
    print("Step 7 - No Numbers Texts:")
    print(no_numbers_texts.head())

    no_punctuation_texts = no_numbers_texts.apply(remove_punctuation)
    print("Step 8 - No Punctuation Texts:")
    print(no_punctuation_texts.head())

    no_stop_words_texts = no_punctuation_texts.apply(remove_stop_words)
    print("Step 9 - No Stop Words Texts:")
    print(no_stop_words_texts.head())

    lemmatized_texts = no_stop_words_texts.apply(lemmatize_text)
    print("Step 10 - Lemmatized Texts:")
    print(lemmatized_texts.head())

    final_texts = lemmatized_texts.apply(remove_extra_whitespace)
    print("Step 11 - Final Cleaned Texts:")
    print(final_texts.head())

    final_texts = final_texts.apply(replace_ontology_classes)
    print("Step 12 - Replace Ontology Classes:")
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
