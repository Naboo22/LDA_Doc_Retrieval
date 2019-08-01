# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

### IMPORTS ###

# Ignore deprecation warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# General
import pandas as pd
import numpy as np

# NLP
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#stop_words.append([[' '], ['article'], ['ext']]) # add stopwords
import spacy
from gensim.models.phrases import Phrases, Phraser
#from sklearn.feature_extraction.text import TfidfVectorizer

# Topic Modelling
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel

# HTML Parsing
from bs4 import BeautifulSoup
import urllib.request
import re

# Plotting
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

########################## PLOTTING SETTINGS ##################################
sns.set()
# Set some general parameters for plots
sns.set(style="dark", palette="muted")
plt.rcParams.update({'font.size': 12,
                     "font.family": 'Times New Roman',
                     'axes.labelsize': 14,
                     'axes.titlepad' : 10,
                     'axes.titlesize' : 15,
                     'axes.labelpad' : 10,
                     'xtick.labelsize': 14,
                     'ytick.labelsize': 14,
                     'font.weight':'normal',
                    }
                   )
###############################################################################

######################### TOPIC DISTRIBUTION #################################

def topic_dist_across_docs(lda_model, corpus_bow, corpus_str, num_dom_topics=1):
    """
    Returns a dataframe containing:
    - the dominant topic found in each comment in corpus_bow;
    - the prior applied to each comment's dominant topic;
    - top ten most probable words associated with dominant topic;
    """
    # Init output
    sent_topics_df = pd.DataFrame()

    results = [] # store results
    for ix, doc in enumerate(corpus_bow):
        # Print progress
        if ix%1000==0:
            print(round(ix/len(corpus_bow), 2))
            
        # Get main topic(s) in each document
        doc_lda = lda_model.get_document_topics(doc)
        doc_lda = sorted(
            # Sort dominant topics in doc by prior
            doc_lda, key=lambda x: (x[1]), reverse=True
        )
        
        # Get the dominant topics, prop contribution of those topics
        # and top keywords associated with topic for each comment
        top_n_topics = tuple(int(x[0]) for x in doc_lda[:num_dom_topics])
        topic_props = tuple(round(x[1],2) for x in doc_lda[:num_dom_topics])
        results.append([top_n_topics, topic_props])
    
    # Rename columns
    sent_topics_df = pd.DataFrame(
            {'Dominant_Topic(s)': [x[0] for x in results], 
             'Prop_Contribution(s)': [x[1] for x in results]}
            )

    # Add processed comment text to rows, 
    # preserve index values as new column
    contents = corpus_str.reset_index()
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    
    return(sent_topics_df)
    
def normalise_and_sort(a_series, threshold=3):
    a_series = a_series / a_series.sum()
    return a_series.sort_values(ascending=False)[:threshold]

def get_top_topics_vs_time(topic_distribution, time_step='10A'):
    
    # Groupby timestep and get counts of each topic
    grouped_topic_counts = topic_distribution.groupby(
        [pd.Grouper(key='Year', freq=time_step), '0']
    )['0'].count()
    
    # Normalise counts (relative to timestep) and return top n
    top_n_per_step = grouped_topic_counts.groupby('Year').apply(normalise_and_sort)
    
    # Drop extra index
    top_n_per_step.index = top_n_per_step.index.droplevel(level=0) 
    
    # Get as plotting df
    # Prep data for plotting
    plot_df = top_n_per_step.to_frame(name='Proportion').reset_index()
    plot_df.Year = plot_df.Year.apply(lambda x: int(x.year))
    plot_df = plot_df.rename({'0':'Topic ID'}, axis=1)
    
    return plot_df
    
###############################################################################
    
#########################    PREPROCESSING    #################################


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(
                # deacc=True removes punctuations
                str(sentence), deacc=True
                )
        )  

# Define functions for stopwords, bigrams, and lemmatization
def remove_stopwords(texts):
    return [[word for word in doc if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(
                [token.lemma_ for token in doc if token.pos_ in allowed_postags]
                )
    return texts_out

def remove_infrequent(corpus, threshold=25):
    bow = [tkn for doc in corpus for tkn in doc]
    bow = pd.Series(bow).value_counts()
    infreq_vocab = set(bow[bow < threshold].index)
    
    corpus = [[tkn for tkn in doc if tkn not in infreq_vocab] for doc in corpus]
    return corpus, infreq_vocab

def pre_process_pipeline(corpus, bigram_min_count=25, bigram_threshold=10,
                         infreq_threshold=25):
    
    # Tokenize corpus
    corpus = list(sent_to_words(corpus))
    
    # Remove Stop Words
    corpus = remove_stopwords(corpus)
        
    # Find and replace empirically modelled bigrams
    bigram = Phrases(
            corpus, min_count=bigram_min_count, threshold=bigram_threshold
            )
    # More efficient method to find and replace bigrams
    bigram_mod = Phraser(bigram)
    corpus = make_bigrams(corpus, bigram_mod)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv
    corpus = lemmatization(corpus, nlp,
                           allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    # Remove infrequent tokens
    corpus, infreq_vocab = remove_infrequent(corpus, threshold=infreq_threshold)
    
    # Remove Stop Words - 2nd time
    corpus = remove_stopwords(corpus)
    
    return corpus

###############################################################################


######################## PMI SCORING USING TO WIKI ############################
    
def get_raw_wiki(url):
    page = urllib.request.urlopen(url) # conntect to website
    soup = BeautifulSoup(page, 'html.parser')
    wiki_corpus = soup.getText().split('\n')
    return wiki_corpus

def clean_wiki(wiki_corpus, meta_vocab=set(['article', 'ext'])):
    
    # Remove meta-vocab
    wiki_corpus = [[tkn for tkn in doc if tkn not in meta_vocab] for doc in wiki_corpus]
    
    # Remove sentences without tokens
    wiki_corpus = [doc for doc in wiki_corpus if len(doc) != 0]
    
    # Remove doc boundaries
    wiki_corpus = [tkn for doc in wiki_corpus for tkn in doc]
    
    return wiki_corpus

def get_word_pairs(top_words):
    
    # Get all possible word pairs (ignoring order)
    word_pairs = {}
    seen_words = []
    for word_A in top_words:
        seen_words.append(word_A) # so that we don't get repeat pairs
        for word_B in [word for word in top_words if word not in seen_words]:
            word_pair = (word_A, word_B)
            word_pairs[word_pair] = 0
            
    return word_pairs

# Create function that gets co-occurrence estimatest from Wiki
def estimate_probs(wiki, top_words=[], window_size=10):
    
    # Get dict of word pairs for results
    word_pairs = get_word_pairs(top_words)
    
    # Get dict for marginal probs
    marginal_probs = {word:0 for word in top_words}
    
    # Iterate sliding window through wiki doc
    for window_num in range(len(wiki) - 10):
    
        # Get next window
        window = wiki[window_num : window_num + window_size]
        
        # Iterate over all word pairs of interest
        for word_pair in word_pairs.keys():
            word_A, word_B = word_pair
            
            # Check if word pair appears in window
            if (word_A in window) and (word_B in window):
                # Add word-pair coccurrence to count
                word_pairs[word_pair] = word_pairs.get(word_pair, 0) + 1
                
            # Check if individual words appear in window
            if (word_A in window):
                marginal_probs[word_A] = marginal_probs.get(word_A, 0) + 1
            elif (word_B in window):
                marginal_probs[word_B] = marginal_probs.get(word_B, 0) + 1
                
    # Convert count to prob estimate
    total_windows = len(wiki) - 10
    word_pairs_probs = {
            word_pair : (count + 1) / total_windows  # add 1 smoothing!
            for word_pair, count in word_pairs.items()
            }
    marginal_probs = {
        word : (count + 1) / total_windows  # add 1 smoothing!
        for word, count in marginal_probs.items()
        }
    
    return word_pairs_probs, marginal_probs

def PMI(joint_prob, marginal_A, marginal_B):
    return np.log2(joint_prob / (marginal_A * marginal_B))

# Create function to calculate the PMI of a word pair
def calc_pairs_PMI(top_words, wiki, window_size=10):
    
    # Get probs for the word pairs of interest
    joint_probs, marginal_probs = estimate_probs(
        wiki, top_words, window_size=window_size
    )
    
    # Calcuate PMI for the word pairs of interest
    pairs_PMI = {}
    for word_pair in joint_probs.keys():
        word_A, word_B = word_pair
        joint = joint_probs[word_pair]
        marg_A = marginal_probs[word_A]
        marg_B = marginal_probs[word_B]
        pairs_PMI[word_pair] = PMI(joint, marg_A, marg_B)
    
    return pairs_PMI

def get_topic_PMI_score(top_words, wiki, window_size=10, agg=np.median):
    
    # Get dict of pairs PMI scores
    pairs_PMI = calc_pairs_PMI(
        top_words, wiki, window_size=window_size
    )
    
    # Agg PMI scores and return value
    scores = [score for score in pairs_PMI.values()]
    return agg(scores)

# Get PMI scores for each topic
def get_topic_PMIs(lda_model, dictionary, wiki_corpus, num_topics = 100, window=10):
    
    topic_PMIs = {}
    for topic_ID in range(num_topics):
        
        # Get top words for topic
        top_words = lda_model.get_topic_terms(topic_ID, 10)
        top_words = [dictionary[word[0]] for word in top_words]
        
        # Get PMI aggregate for topic
        topic_PMI = get_topic_PMI_score(
            top_words, wiki_corpus, window_size=window, agg=np.median
        )
        
        topic_PMIs[topic_ID] = topic_PMI
        
    return topic_PMIs

def filter_PMIs(topic_PMIs):
    """
    Remove topics whose top ten terms do not appear in collocations in the wiki
    and therefore have identical PMI scores (due to add-one smoothing)
    
    Returns a pandas Series with index of topic ID and values of PMI score
    """
    
    topic_PMIs = pd.Series(topic_PMIs)
    # Get value of duplicate (PMI produced by add-one smoothign alone)
    duplicate_PMI = topic_PMIs[topic_PMIs.duplicated()].tolist()[0]
    
    topic_PMIs = topic_PMIs[topic_PMIs != duplicate_PMI]
    
    return topic_PMIs

def get_topic_history(data, bow_corpus, topic_ID, lda_model,
                      timestep_code='1A', agg=np.mean):
    """
    Returns the probability assigned to given topic aggregated over each time
    step as a pd.Series
    """

    agg_results = {}
    for time_step, data in data.resample(timestep_code, on='year'):

        # Get indices for data in this range
        indices = data.index.tolist()
        
        # Get list of topic probs for docs in this timestep
        topic_probs = [
            get_topic_prob(bow_corpus[ix], topic_ID, lda_model) 
            for ix in indices
        ]

        # Aggregate and add to results list
        agg_results[time_step.year] = agg(topic_probs)
                
        # Show Progress
        print(time_step.year)
        if time_step.year % 20 == 0:
            print(f'Mean prob: {agg(topic_probs)}')
        
    return pd.Series(agg_results)

def get_topic_prob(bow_doc, topic_ID, lda_model):
    """
    Returns topic prob of topic of interest for a single bow rep of a doc
    """
    topic_dist = lda_model.get_document_topics(
        bow_doc, 
        minimum_probability=1e-15
    )
    return topic_dist[topic_ID][1]

###############################################################################

############################## PLOTTING #######################################

def plot_topic_dist(topic_distributions, ax=None):
    # Prepare acis
    if ax==None:
        fig, ax = plt.subplots()
    else:
        pass
    # Plot
    ax = sns.lineplot( data=topic_distributions, ax=ax)
    return ax

def display_top_topics(top_5, lda_model, ax=None):
    """
    top_5: p.Series with idx topic ID, values PMI score
    lda_model: gensim lda_model from which to retrieve topical words
    """
    
    # X ticks
    #time_steps = top_5.tolist()
    time_steps = [
            f'Topic {top_5.index[ix]}, ' + str(PMI) for ix, PMI in enumerate(top_5)]
    
    # # For smaller time range
    # time_steps = time_steps[:len(time_steps) // 2]
    # print(time_steps)
    
    # Prepare x axis
    if ax==None:
        fig, ax = plt.subplots()
    else:
        pass
    
    plt.xticks(
        [0.5] + list(range(1, len(time_steps) + 1)) + [len(time_steps) + 0.5],
        [''] + time_steps + [''],
    )
    
    # Move through groups, displaying text in group
    bbox = dict(boxstyle="round", fc='cyan', alpha=0.1)
    
    for i, grp in enumerate(top_5.index):
        top_n = list(reversed([x[0] for x in lda_model.show_topic(grp)]))
    
        for j in range(len(top_n)):
            # show each word in each group in a column
            ax.text(
                (1 + 1*i), (0.1 + 0.09*j), top_n[j],
                size=15, horizontalalignment='center',
                bbox=bbox
            )
            
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False) # labels along the bottom edge are off
    
    ax.set_xlabel('PMI Score')
    ax.set_ylabel('Probability in Topic')
    ax.set_xlim(left=0.5)
    ax.grid(False)
    return ax
    
#"""MAKE FUNCTION"""
#top_topics = topics_per_decade.groupby('Year').apply(lambda x: x.iloc[0, :])
#second_topics = topics_per_decade.groupby('Year').apply(lambda x: x.iloc[1, :])
#thrid_topics = topics_per_decade.groupby('Year').apply(lambda x: x.iloc[2, :])
#top_topics.Year = top_topics.Year.astype(int)
#second_topics.Year = second_topics.Year.astype(int)
#thrid_topics.Year = thrid_topics.Year.astype(int)
#
#
#def get_bars(ax):
#    childrenLS = ax.get_children()
#    barlist = filter(
#        lambda x: isinstance(x, matplotlib.patches.Rectangle), 
#        childrenLS
#    )
#    return barlist
#
#def grouped_bar_topics_over_time(plot_df, width_bars=.9, colour_map='tab20',
#                                 top_n='auto', ax=ax, show_every=1):
#    
#    # Group the data by the time step (chosen prior to handing)
#    grouped = plot_df.groupby('Year')
#    
#    # Get the number of topics to model per time step
#    if top_n == 'auto':
#        top_n = grouped['Proportion'].count().iloc[0]
#    else:
#        pass
#    
#    # Plot prep
#    years = plot_df.Year.unique()
#    x = np.arange(len(years))  # the label locations
#    
#    # get color dictionary for topic ID --> color
#    topics = sorted(plot_df['Topic ID'].unique())
#    #my_cmap = cm.get_cmap(colour_map)
#    my_cmap = ListedColormap(sns.color_palette("Spectral", 22))
#    cmapper = {topic_id: ix for ix, topic_id in enumerate(topics)}
#    #my_norm = BoundaryNorm(topics, ncolors=my_cmap.N)
#    
#    # Iterate over top_n topics and plot in grouped barchart
#    for rank in range(top_n):
#    
#        # Get list of topic proportions
#        topic_props = grouped.apply(lambda x: x.iloc[rank, 2]).tolist()
#
#        # Get list of topic keys (for colour mapping)
#        topic_keys = grouped.apply(lambda x: x.iloc[rank, 1]).tolist()
#        
#        # Get colors for bars
#        color_ids = [my_cmap(cmapper[topic_id]) for topic_id in topic_keys]
#        
#        # Plot the bars
#        rects = ax.bar(
#            x - width_bars + rank*(width_bars), 
#            topic_props, 
#            width_bars, 
#            color=color_ids,
#            #alpha=0.7,
#            tick_label=sorted(years)
#        )
#        
#        # Legend
#        patches = []
#        for topic in pd.Series(topic_keys).unique():
#            patch = mpatches.Patch(
#                color=my_cmap(cmapper[topic]), label=f'{int(topic)}'
#            )
#            patches.append(patch)
#        ax.legend(handles=patches, fontsize=12, ncol=5, title='Topic IDs')
#    
#    # Aesthetics
#    ax.grid(False)
#    
#    for ix, tick in enumerate(ax.get_xticklabels()):
#        tick.set_rotation(45) # set rotation of tick labels
#        if ix % show_every != 0: # choose whether to show tick
#            tick.set_visible(False)
#        
#    return ax

###############################################################################
import multiprocessing
def get_topic_prob_in_timestep(bow_subset, lda_model, topic_ID):
    """
    Returns the probability assigned to a topic in a timestep
    """
    # Progress
    print(multiprocessing.current_process().name)
    
    # Get list of topic probs for docs in this timestep
    topic_probs = [
        get_topic_prob(bow_doc, topic_ID, lda_model) 
        for bow_doc in bow_subset
    ]
        
    return pd.Series(topic_probs)















