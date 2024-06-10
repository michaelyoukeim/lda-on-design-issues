# new
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
from config import DEFAULT_DATA_PATH
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import scipy.sparse
import joblib
import math
import time

# Configuration Section
NUM_TOPICS = 20
ALPHA = 0.01
BETA = 0.01
TOPIC_RANGE_START = 3
TOPIC_RANGE_END = 20
NUM_TOP_WORDS = 20


def create_document_term_matrix(df: pd.DataFrame, save_path=None):
    print("Creating Document-Term Matrix...")
    start_time = time.time()
    # Ensure that cleaned_text is a string
    df['cleaned_text'] = df['cleaned_text'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else str(x))

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    texts_joined = df['cleaned_text'].astype(str)
    dtm = vectorizer.fit_transform(texts_joined)

    if save_path:
        scipy.sparse.save_npz(save_path, dtm)
        with open(save_path.replace('.npz', '_feature_names.txt'), 'w') as f:
            for feature in vectorizer.get_feature_names_out():
                f.write(f"{feature}\n")
    print(
        f"Document-Term Matrix created in {time.time() - start_time:.2f} seconds.")
    return dtm, vectorizer


def load_document_term_matrix(load_path):
    print("Loading Document-Term Matrix...")
    start_time = time.time()
    dtm = scipy.sparse.load_npz(load_path)
    with open(load_path.replace('.npz', '_feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f]
    vectorizer = CountVectorizer(vocabulary=feature_names)
    print(
        f"Document-Term Matrix loaded in {time.time() - start_time:.2f} seconds.")
    return dtm, vectorizer


def run_lda(dtm, num_topics=NUM_TOPICS, alpha=ALPHA, beta=BETA):
    print(f"Running LDA with {num_topics} topics...")
    start_time = time.time()
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42,
                                    learning_method='batch', max_iter=100, doc_topic_prior=alpha, topic_word_prior=beta)
    lda.fit(dtm)
    print(f"LDA completed in {time.time() - start_time:.2f} seconds.")
    return lda


def display_topics(model, feature_names, num_top_words=NUM_TOP_WORDS):
    print(f"Displaying top {num_top_words} words per topic...")
    topics = {}
    all_topic_terms = {}
    for topic_idx, topic in enumerate(model.components_):
        important_words = [feature_names[i]
                           for i in topic.argsort()[:-num_top_words - 1:-1]]
        all_words = {feature_names[i]: topic[i] for i in topic.argsort()}
        topics[f"Topic {topic_idx + 1}"] = important_words  # 1-indexed
        all_topic_terms[f"Topic {topic_idx + 1}"] = all_words  # 1-indexed
        print(f"Topic {topic_idx + 1}:")  # 1-indexed
        print(" ".join(important_words))
    return topics, all_topic_terms


def save_topics_to_csv(topics, filepath):
    print(f"Saving topics to {filepath}...")
    topics_df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in topics.items()]))
    topics_df.to_csv(filepath, index=False)
    print(f"Topics saved to {filepath}.")


def save_all_topic_terms_to_csv(all_topic_terms, filepath):
    print(f"Saving all topic terms to {filepath}...")
    all_topic_terms_df = pd.DataFrame.from_dict(
        all_topic_terms, orient='index').transpose()
    all_topic_terms_df.to_csv(filepath, index=True)
    print(f"All topic terms saved to {filepath}.")


def save_most_occurring_topics(df, filepath):
    print(f"Saving most occurring topics to {filepath}...")
    topic_counts = df['dominant_topic'].value_counts()
    most_occurring_topics_df = pd.DataFrame(
        {'Topic': topic_counts.index, 'Count': topic_counts.values})
    most_occurring_topics_df.to_csv(filepath, index=False)
    print(f"Most occurring topics saved to {filepath}.")


def save_issue_details_per_topic(df, filepath):
    print(f"Saving issue details per topic to {filepath}...")
    topics_detail = df.groupby('dominant_topic').apply(
        lambda x: list(x['key']))
    topics_detail = topics_detail.reset_index()
    topics_detail.columns = ['Topic', 'Issues']
    topics_detail.to_csv(filepath, index=False)
    print(f"Issue details per topic saved to {filepath}.")


def calculate_coherence(df, num_topics_range):
    print("Calculating coherence scores...")
    start_time = time.time()
    dictionary = corpora.Dictionary(df['cleaned_text'].apply(str.split))
    corpus = [dictionary.doc2bow(text.split()) for text in df['cleaned_text']]
    coherence_scores = []
    for num_topics in num_topics_range:
        print(f"Calculating coherence for {num_topics} topics...")
        lda_gensim = models.LdaMulticore(
            corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2, random_state=42)
        coherence_model_lda = CoherenceModel(model=lda_gensim, texts=df['cleaned_text'].apply(
            str.split), dictionary=dictionary, coherence='c_v')
        coherence_scores.append(coherence_model_lda.get_coherence())
    print(
        f"Coherence scores calculated in {time.time() - start_time:.2f} seconds.")
    return coherence_scores


def plot_coherence_scores(num_topics_range, coherence_scores, show_plot=False):
    print("Plotting coherence scores...")
    plt.figure(figsize=(12, 6))
    plt.plot(num_topics_range, coherence_scores, marker='o',
             linestyle='-', color='royalblue', markerfacecolor='orange')
    plt.xlabel("Number of Topics", fontsize=14)
    plt.ylabel("Coherence Score", fontsize=14)
    plt.title("Coherence Score vs. Number of Topics", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(num_topics_range)

    if show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(DEFAULT_DATA_PATH, 'coherence_scores.pdf'))
    print("Coherence scores plotted.")


def plot_top_words(model, feature_names, num_top_words, title, show_plot=False):
    print("Plotting top words per topic...")
    num_topics = model.components_.shape[0]
    num_cols = 4
    num_rows = math.ceil(num_topics / num_cols)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(15, 7 * num_rows), sharex=True)
    axes = axes.flatten()

    colors = plt.cm.Paired(range(num_topics))

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-num_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, color=colors[topic_idx % len(colors)])
        ax.set_title(f'Topic {topic_idx + 1}',
                     fontdict={'fontsize': 12})  # 1-indexed
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=10)
        for i in range(len(top_features)):
            ax.text(weights[i], i, f'{weights[i]:.2f}',
                    ha='left', va='center', fontsize=8, color='black')

    for i in range(topic_idx + 1, len(axes)):  # Hide any unused subplots
        fig.delaxes(axes[i])

    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.90, hspace=0.4)

    if show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(DEFAULT_DATA_PATH, 'top_words_per_topic.pdf'))
    print("Top words per topic plotted.")


def plot_topic_dominance(df, show_plot=False):
    print("Plotting topic dominance...")
    topic_counts = df['dominant_topic'].value_counts()
    plt.figure(figsize=(12, 6))
    topic_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel("Topics", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Dominance of Topics Across Issues", fontsize=16)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(DEFAULT_DATA_PATH, 'topic_dominance.pdf'))
    print("Topic dominance plotted.")


def analyze_topics(save_dtm=False, num_topics=NUM_TOPICS, alpha=ALPHA, beta=BETA, topic_range_start=TOPIC_RANGE_START, topic_range_end=TOPIC_RANGE_END, num_top_words=NUM_TOP_WORDS):
    print("Starting topic analysis...")
    df = pd.read_excel(os.path.join(DEFAULT_DATA_PATH,
                       'preprocessed_issue_details.xlsx'))

    dtm, vectorizer = create_document_term_matrix(
        df, save_path=os.path.join(DEFAULT_DATA_PATH, 'dtm.npz') if save_dtm else None)
    num_topics_range = range(topic_range_start, topic_range_end + 1)
    coherence_scores = calculate_coherence(df, num_topics_range)
    plot_coherence_scores(num_topics_range, coherence_scores)

    optimal_num_topics = num_topics_range[coherence_scores.index(
        max(coherence_scores))]
    print(f"Optimal number of topics: {optimal_num_topics}")
    lda = run_lda(dtm, num_topics=optimal_num_topics, alpha=alpha, beta=beta)
    topics, all_topic_terms = display_topics(
        lda, vectorizer.get_feature_names_out(), num_top_words)

    # Save topics to CSV
    save_topics_to_csv(topics, os.path.join(
        DEFAULT_DATA_PATH, 'all_topic_data.csv'))

    # Save all topic terms to CSV
    save_all_topic_terms_to_csv(all_topic_terms, os.path.join(
        DEFAULT_DATA_PATH, 'all_topic_terms.csv'))

    # Save LDA model and coherence scores
    joblib.dump(lda, os.path.join(DEFAULT_DATA_PATH, 'lda_model.pkl'))
    pd.DataFrame({'num_topics': list(num_topics_range), 'coherence_score': coherence_scores}).to_csv(
        os.path.join(DEFAULT_DATA_PATH, 'coherence_scores.csv'), index=False)

    topic_proportions = lda.transform(dtm)
    dominant_topic = topic_proportions.argmax(axis=1) + 1  # 1-indexed
    df['dominant_topic'] = dominant_topic

    plot_topic_dominance(df)
    plot_top_words(lda, vectorizer.get_feature_names_out(),
                   num_top_words, 'Top words per topic')

    # Save the most occurring topics
    save_most_occurring_topics(df, os.path.join(
        DEFAULT_DATA_PATH, 'most_occurring_topics.csv'))

    # Save detailed information of issues per topic
    save_issue_details_per_topic(df, os.path.join(
        DEFAULT_DATA_PATH, 'issue_details_per_topic.csv'))

    print("Topic analysis completed.")
    return df, topics


if __name__ == '__main__':
    analyze_topics(save_dtm=True)
