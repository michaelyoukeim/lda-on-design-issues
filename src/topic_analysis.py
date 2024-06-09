import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import os
from config import DEFAULT_DATA_PATH
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import scipy.sparse
import math

# Configuration Section
NUM_TOPICS = 9
ALPHA = 0.01
BETA = 0.01

def create_document_term_matrix(df: pd.DataFrame, save_path=None):
    # Ensure that cleaned_text is a string
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    texts_joined = df['cleaned_text'].astype(str)
    dtm = vectorizer.fit_transform(texts_joined)
    
    if save_path:
        scipy.sparse.save_npz(save_path, dtm)
        with open(save_path.replace('.npz', '_feature_names.txt'), 'w') as f:
            for feature in vectorizer.get_feature_names_out():
                f.write(f"{feature}\n")
    
    return dtm, vectorizer

def load_document_term_matrix(load_path):
    dtm = scipy.sparse.load_npz(load_path)
    with open(load_path.replace('.npz', '_feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f]
    vectorizer = CountVectorizer(vocabulary=feature_names)
    return dtm, vectorizer

def run_lda(dtm, num_topics=NUM_TOPICS, alpha=ALPHA, beta=BETA):
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, learning_method='batch', max_iter=100, doc_topic_prior=alpha, topic_word_prior=beta)
    lda.fit(dtm)
    return lda

def display_topics(model, feature_names, num_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        important_words = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        topics[f"Topic {topic_idx}"] = important_words
        print(f"Topic {topic_idx}:")
        print(" ".join(important_words))
    return topics

def calculate_coherence(df, num_topics_range):
    dictionary = corpora.Dictionary(df['cleaned_text'].apply(str.split))
    corpus = [dictionary.doc2bow(text.split()) for text in df['cleaned_text']]
    coherence_scores = []
    for num_topics in num_topics_range:
        lda_gensim = models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2, random_state=42)
        coherence_model_lda = CoherenceModel(model=lda_gensim, texts=df['cleaned_text'].apply(str.split), dictionary=dictionary, coherence='c_v')
        coherence_scores.append(coherence_model_lda.get_coherence())
    return coherence_scores

def plot_coherence_scores(num_topics_range, coherence_scores):
    plt.figure(figsize=(12, 6))
    plt.plot(num_topics_range, coherence_scores, marker='o', linestyle='-', color='royalblue', markerfacecolor='orange')
    plt.xlabel("Number of Topics", fontsize=14)
    plt.ylabel("Coherence Score", fontsize=14)
    plt.title("Coherence Score vs. Number of Topics", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(num_topics_range)
    plt.savefig(os.path.join(DEFAULT_DATA_PATH, 'coherence_scores.pdf'))
    plt.show()

def plot_top_words(model, feature_names, n_top_words, title):
    num_topics = model.components_.shape[0]
    num_cols = 4
    num_rows = math.ceil(num_topics / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 7 * num_rows), sharex=True)
    axes = axes.flatten()
    
    colors = plt.cm.Paired(range(num_topics))
    
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, color=colors[topic_idx % len(colors)])
        ax.set_title(f'Topic {topic_idx + 1}', fontdict={'fontsize': 12})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=10)
        for i in range(len(top_features)):
            ax.text(weights[i], i, f'{weights[i]:.2f}', ha='left', va='center', fontsize=8, color='black')
    
    for i in range(topic_idx + 1, len(axes)):  # Hide any unused subplots
        fig.delaxes(axes[i])

    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.90, hspace=0.4)
    plt.savefig(os.path.join(DEFAULT_DATA_PATH, 'top_words_per_topic.pdf'))
    plt.show()

def plot_topic_dominance(df):
    topic_counts = df['dominant_topic'].value_counts()
    plt.figure(figsize=(12, 6))
    topic_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel("Topics", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Dominance of Topics Across Issues", fontsize=16)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(DEFAULT_DATA_PATH, 'topic_dominance.pdf'))
    plt.show()

def analyze_topics(save_dtm=False, num_topics=NUM_TOPICS, alpha=ALPHA, beta=BETA):
    df = pd.read_excel(os.path.join(DEFAULT_DATA_PATH, 'preprocessed_issue_details.xlsx'))

    dtm, vectorizer = create_document_term_matrix(df, save_path=os.path.join(DEFAULT_DATA_PATH, 'dtm.npz') if save_dtm else None)
    num_topics_range = range(3, 11)
    coherence_scores = calculate_coherence(df, num_topics_range)
    plot_coherence_scores(num_topics_range, coherence_scores)

    optimal_num_topics = num_topics_range[coherence_scores.index(max(coherence_scores))]
    lda = run_lda(dtm, num_topics=optimal_num_topics, alpha=alpha, beta=beta)
    topics = display_topics(lda, vectorizer.get_feature_names_out(), 10)

    topic_proportions = lda.transform(dtm)
    dominant_topic = topic_proportions.argmax(axis=1)
    df['dominant_topic'] = dominant_topic

    plot_topic_dominance(df)
    plot_top_words(lda, vectorizer.get_feature_names_out(), 10, 'Top words per topic')

    return df, topics

if __name__ == '__main__':
    analyze_topics(save_dtm=True)
