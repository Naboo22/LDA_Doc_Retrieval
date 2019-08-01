# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:20:00 2019

@author: md522
"""

# Custom utils for tasks such as preprocessing
import utils
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

class Query_Corpus:
    """
    Query class is a basic framework for a tool which, given a 
    raw_corpus and a search topic (phrase), can return possible 
    documents of interest, as well as some additional information.
    
    This is done by modelling the corpus using LDA - see Blei (2003), 
    ranking the resultant topics based upon the PMI score of the word 
    pairs created from the topics' most probable words, using the 
    parsed wiki page provided (or relating to the search phrase) 
    - see Newman (2010).
    
    These probability distribution of these topics of interest (TOIs)
    is then plotted over the corpus, and an example of a document 
    corressponding to each topic is shown.
    """
    
    def __init__(self, data, lda_model, lda_bow, lda_dict, 
                 url=None, wiki_processed=None,
                 PMI_scores=None, top_topics=None):
        """
        raw_corpus: list of lists containing string documents (untokenized)
        data: pd.DataFrame with cols for: corpus, year (datetime)
        """
        
        self.data = data
        self.lda_model = lda_model
        self.url = url
        self.wiki_processed=wiki_processed
        self.PMI_scores = PMI_scores
        self.lda_dict = lda_dict
        self.lda_bow = lda_bow
        self.top_topics = top_topics
        
    ######################## MASTER FUNCTIONS ################################
    
    def show_matching_topics(self, top_n=5, plot_type='words', time_step='1A',
                             ax=None):
        """
        Display the top n topics' top ten most probable words on a plot
        or the mean probabilities of each topic per timestep
        """
        
        # We need top_n PMI scoring topics first
        if self.lda_model == None:
            self.lda_model = self.get_lda_model()
        if self.top_topics == None:
            self.top_topics = self.get_top_n_topics(top_n=5)
        else:
            pass
        
        # Plot
        if ax == None:
            fig, ax = plt.subplots()
        
        if plot_type == 'words':
            ax = utils.display_top_topics(self.top_topics, self.lda_model,
                                          ax=ax)
            return ax
        elif plot_type == 'prob_dist':
            # Get topic IDs of interest as list of ints
            top_topic_IDs = self.top_topics.index.tolist()
            
            # Get prob dists of each topic over corpus (as DataFrame)
            self.topic_prob_distributions = self.get_topic_prob_dists(
                    topic_IDs=top_topic_IDs, time_step=time_step)
            
            # Plot probs distribuions
            ax = utils.plot_topic_dist(
                    self.top_topics, self.lda_model, ax=ax)
            return ax
    
    def process_fit_and_match(self):
        """
        Process raw corpus, fit LDA model, and return PMI score for each topic
        ID as a pd.Series.
        """
        # Pre-process corpus
        self.processed_corpus = self.get_processed_corpus()

        # Fit LDA model
        self.lda_model = self.get_lda_model()

        # Get PMI scores for topics
        self.PMI_scores = self.get_PMI_scores()
        
        return self.PMI_scores
    
    ##########################################################################
    
    def get_topic_history_parallel(self, topic_ID, time_step='1A', agg=np.mean,
                                   num_processors=None):
        """
        Returns the probability assigned to given topic aggregated over each time
        step as a pd.Series. Topic probabilities are calculated for each timestep
        (index range) concurrently.
        """
        
        # Get list of lists containing index ranges
        index_ranges = [
                (time_range, subset.index.tolist()) for time_range, subset in 
                self.data.resample('1A', on='year')
                ]
    
#        if __name__ ==  '__main__': 
        # Prepare for prallelising 
        if num_processors == None:
            num_processors = mp.cpu_count() - 1
        else:
            pass
        pool = mp.Pool(processes=num_processors)
        
        # Get results in parallel
        topic_probs = {x[0].year: pool.apply_async(
            utils.get_topic_prob_in_timestep, 
            args=(self.lda_bow[x[1]], self.lda_model, topic_ID))
            for x in index_ranges}
        
        topic_probs = {year: val.get() for year, val in topic_probs.items()}
    
        # Aggregate and add to results list
        agg_results = {year : agg(probs) for year, probs in topic_probs.items()}
            
        return pd.Series(agg_results)
    
    def get_topic_prob_dists(self, time_step='1A', topic_IDs=[0], 
                             num_processors=None):
        """
        Return a pd.DataFrame containing the mean probabilities of each topic 
        passed for each timestep.
        """
        
        results_dict = {}
        for topic_ID in topic_IDs:
            # Progess
            print(f'Working on topic {topic_ID}')
            print(type(self.data))
            
            # Get prob dist over docs for current topic ID
            prob_dist = self.get_topic_history_parallel(
                    topic_ID, num_processors=num_processors)
            
            # Add to results dict
            results_dict[f'Topic {topic_ID}'] = prob_dist
        
        # Place all results in DataFrame
        self.topic_prob_distributions = pd.DataFrame(results_dict)
        
        return self.topic_prob_distributions
    
    def get_top_n_topics(self, top_n=5):
        """
        Return the top n topics (by PMI score)
        """
        if self.PMI_scores == None:
            self.PMI_scores = self.process_fit_and_match()
        else:
            pass
        
        # Get top n topics (lowest to highest PMI score)
        self.top_topics = self.PMI_scores.sort_values(ascending=True)[-top_n:]
        return self.top_topics
        
        
    def get_processed_corpus(self):
        """
        Return processed corpus (tokenized, lemmatized, collocated,...)
        """
        if self.processed_corpus == None:
            # Do the processing with raw_corpus
            pass
        else:
            return self.processed_corpus
        
    def get_lda_model(self):
        """
        Return gensim LDA model for corpus
        """
        if self.lda_model == None:
            # Fit the model
            pass
        else:
            return self.lda_model
        
    
    def get_PMI_scores(self):
        
        if self.PMI_scores == None:
            if self.raw_wiki == None:
                pass
            else:
                # Parse text of wiki page
                self.raw_wiki = utils.get_raw_wiki(self.url)
                
                # Preprocess raw wiki text
                self.wiki_processed = utils.pre_process_pipeline(
                        self.raw_wiki, bigram_min_count=10, bigram_threshold=1)
                self.wiki_processed = utils.clean_wiki(self.wiki_processed)
            
            # Calculate topic PMI scores
            self.PMI_scores = utils.get_topic_PMIs(
                    self.lda_model, self.lda_dict, self.wiki_processed)
            
            # Remove PMI scores that are produced by add-one smoothing alone
            self.PMI_scores = utils.filter_PMIs(self.PMI_scores)
            
            return self.PMI_scores
        
        else:
            return self.PMI_scores
            
    
    def process_text(self):
        return self
       
###############################################################################























