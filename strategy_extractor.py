
import pandas as pd
import numpy as np
import re
import keywords
import en_core_web_sm
import time
import json
nlp = en_core_web_sm.load()

        
# main_features = ['Acknowledgement', 'Agreement', 'Hedges', 'Negation', 'Positive_Emotion', 'Subjectivity', 'Adverb_Limiter', 'Disagreement', 'Negative_Emotion']
# main_features_pos = ['Acknowledgement', 'Agreement', 'Hedges', 'Positive_Emotion', 'Subjectivity']
# main_features_neg = ['Negation', 'Negative_Emotion', 'Adverb_Limiter', 'Disagreement']
kw = keywords.kw


def prep_data(text):

    # text cleaning
    
    text = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)
    text = text.lstrip()
    text = text.lower()
    # t = re.sub(r"[.?!]+\ *", "", t)  # spcifially replace punctuations with nothing
    # t = re.sub('[^A-Za-z,]', ' ', t)  # all other special chracters are replaced with blanks
    
    orig = ["let's", "i'm", "won't", "can't", "shan't", "'d",
            "'ve", "'s", "'ll", "'re", "n't", "u.s.a.", "u.s.", "e.g.", "i.e.",
            "‘", "’", "“", "”", "100%", "  ", "mr.", "mrs.", "dont", "wont"]

    new = ["let us", "i am", "will not", "cannot", "shall not", " would",
        " have", " is", " will", " are", " not", "usa", "usa", "eg", "ie",
        "'", "'", '"', '"', "definitely", " ", "mr", "mrs", "do not", "would not"]

    for i in range(len(orig)):
        text = text.replace(orig[i], new[i])

    return text

def count_matches(keywords, spacy_df, inversion = False):
    
    """
    For a given series of tokens, searches for occurrences of keywords from a predefined list 
    and returns a DataFrame where each column corresponds to a category, and each row 
    represents a token. The cells contain either 1 (if the token matches a keyword in the 
    category) or 0 (if not).

    Inputs:
    keywords (dict): A dictionary with keys as categories and values as lists of phrases to search.
    tokens (pd.Series): A pandas Series where each element is a token.

    Outputs:
    DataFrame: A pandas DataFrame where each column corresponds to a keyword category, and each 
    cell contains either 1 (token matches a keyword) or 0 (no match).
    """

    if inversion == True:
        tokens = spacy_df['TOKEN'][spacy_df['Negations'] == 1].to_list()
    else:
        tokens = spacy_df['TOKEN']
        
    tokens = [' ' + token + ' ' for token in tokens] # add spaces around each word. Ensures match with dictionary AND are seperate words
       
    for key in keywords:
        if key not in spacy_df.columns:
            spacy_df[key] = 0 

        for word in keywords[key]:
            if word in tokens:
                for token in tokens:
                    if word == token:
                        spacy_df.loc[spacy_df['TOKEN'].apply(lambda x: x == token.strip()), key] += 1

    return spacy_df


def get_dep_pairs(spacy_df):
    """
    Uses spaCy to find list of dependency pairs from text.
    Performs negation handling where by any dependency pairs related to a negated term is removed

    Input: Text

    Outputs: Dependency pairs from text that do not have ROOT as the head token or is a negated term
    """

    spacy_df['Negations'] = np.where(spacy_df['DEP'] == 'neg', 1, 0)
    exclusions = set(spacy_df['HEAD_INDEX'][spacy_df['DEP'] == 'neg'].to_list())
    spacy_df['DEP_PAIRS_POS'] = np.where(spacy_df['HEAD_INDEX'].isin(exclusions) | (spacy_df['DEP'] == 'neg'), 0, 1)

    return spacy_df


def count_spacy_matches(keywords, spacy_df, negations = True):
    """
    When searching for key words are not sufficient, we may search for dependency pairs.
    Finds any-prespecified dependency pairs from text string and outputs the counts

    Inputs:
            Dependency pairs from text
            Predefined tokens for search in dependency heads

    Output:
            Count of dependency pair matches
    """
    if negations == True:
        dep_pairs = spacy_df['DEP_PAIRS'][spacy_df['DEP_PAIRS_POS'] == 1].to_list()
    else:
        dep_pairs = spacy_df['DEP_PAIRS'].to_list()

    for key in keywords:
        if key not in spacy_df.columns:
            spacy_df[key] = 0 
        check = any(item in dep_pairs for item in keywords[key])
        
        if check == True:
            for phrase in keywords[key]:
                if phrase in dep_pairs:
                    for dep in dep_pairs:
                        if phrase == dep:
                            spacy_df.loc[spacy_df['DEP_PAIRS'].apply(lambda x: x == dep), key] += 1

    return spacy_df


def word_start(keywords, spacy_df):
    """
    Find first words in text such as conjunctions and affirmations
    """
    
    first_words = spacy_df['TOKEN'][spacy_df['WORD_NUM'] == 1].to_list()
    first_words = [' ' + word + ' ' for word in first_words]
    token_index = spacy_df['TOKEN_INDEX'][spacy_df['WORD_NUM'] == 1].to_list()

    for key in keywords:
        
        if key not in spacy_df.columns:
            spacy_df[key] = 0 
            
        for i in range(len(first_words)):
            if first_words[i] in keywords[key]:
                spacy_df.loc[spacy_df['TOKEN_INDEX'].apply(lambda x: x == token_index[i]), key] += 1

    return spacy_df

def bare_command(spacy_df):
    """
    Check the first word of each sentence is a verb AND is NOT contained in list of key words
    In other words, all other verbs not in the list of keywords are considered bare commands

    Output: Count of matches
    """

    keywords = set([' be ', ' do ', ' please ', ' have ', ' thank ', ' hang ', ' let '])
    
    first_words = spacy_df['TOKEN'][spacy_df['WORD_NUM'] == 1].to_list()
    first_words = [' ' + word + ' ' for word in first_words]
    token_index = spacy_df['TOKEN_INDEX'][spacy_df['WORD_NUM'] == 1].to_list()
    tags = spacy_df['TAG'][spacy_df['WORD_NUM'] == 1].to_list()
    bc = [c for a, b, c in zip(tags, first_words, token_index) if a == 'VB' and b not in keywords]
    
    spacy_df['Bare_Command'] = 0
    
    for i in range(len(bc)):
        spacy_df.loc[spacy_df['TOKEN_INDEX'].apply(lambda x: x == bc[i]), 'Bare_Command'] += 1
    
    return spacy_df

def Question(spacy_df):
    """
    Counts number of prespecified question words
    """

    # question_keywords = set([' who ', ' what ', ' where ', ' when ', ' why ', ' how ', ' which '])
    search_tags = set(['WRB', 'WP', 'WDT'])
    
    # subset sentence numbers and get the first word from that sentence if the last word is a question mark.
    sent_idx = spacy_df['SENT_NUM'][(spacy_df['TOKEN'] == '?') & (spacy_df['POS'] == 'PUNCT')]
    
    # create new column wh question. If tag is in list of search_tags, return TOKEN_INDEX
    # Use TOKEN_INDEX to add 1 to correct row
    spacy_df['WH_Questions'] = 0
    spacy_df['YesNo_Questions'] = 0
    
    for sent_num in sent_idx:
        idx = spacy_df['TOKEN_INDEX'][(spacy_df['SENT_NUM'] == sent_num) & (spacy_df['WORD_NUM'] == 1)].item()
        if spacy_df['TAG'][(spacy_df['SENT_NUM'] == sent_num) & (spacy_df['WORD_NUM'] == 1)].item() in search_tags:
           spacy_df.loc[spacy_df['TOKEN_INDEX'].apply(lambda x: x == idx), 'WH_Questions'] += 1
        else:
            spacy_df.loc[spacy_df['TOKEN_INDEX'].apply(lambda x: x == idx), 'YesNo_Questions'] += 1
        
    return spacy_df

def adverb_limiter(keywords, spacy_df):
    """
    Search for tokens that are advmod and in the prespecifid list of words
    """
    
    al_tokens = spacy_df['TOKEN'][spacy_df['DEP'] == 'advmod'].to_list()
    al_idx = spacy_df['TOKEN_INDEX'][spacy_df['DEP'] == 'advmod'].to_list()
    
    spacy_df['Adverb_Limiters'] = 0
    for i in range(len(al_tokens)):
        if str(' ' + str(al_tokens[i]) + ' ') in keywords['Adverb_Limiter']:
            spacy_df.loc[spacy_df['TOKEN_INDEX'].apply(lambda x: x == al_idx[i]), 'Adverb_Limiters'] += 1

    return spacy_df

def spacy_table(doc):
    
    """
    Analyzes the given text and produces a DataFrame containing various linguistic features for each token.
    
    Parameters:
    text (str): The text to analyze.
    
    Outputs:
    None, but prints a DataFrame that includes the sentence number, word number, text of each token,
    lemma, part of speech, detailed tag, dependency relation, shape, alpha status, stop word status,
    head token's text, index of the head token, and the index of the token itself.
    
    The function processes the text through the spaCy NLP pipeline to extract these features,
    organizing the data by sentences and tokens to provide detailed linguistic insights.
    """

    # Initialize data list
    data = []

    # Initialize sentence number
    sentence_number = 1
    
    for sent in doc.sents:
        # Initialize word number within each sentence
        word_number = 1

        for token in sent:
            data.append({
                'TOKEN_INDEX': token.i,   # Word number across all sentences
                'SENT_NUM': sentence_number,
                'WORD_NUM': word_number, # Word number within a sentence
                'TOKEN': token.text,
                'POS': token.pos_,
                'TAG': token.tag_,
                'DEP': token.dep_,
                'HEAD_TEXT': token.head.text,  # Head token's text
                'HEAD_INDEX': token.head.i    # Index of head token in the document 
            })
            word_number += 1  # Increment word number
        
        sentence_number += 1  # Increment sentence number
        
    data = pd.DataFrame(data)
    data['DEP_PAIRS_INPUT'] = data.apply(lambda row: [row['DEP'], row['HEAD_TEXT'], row['HEAD_INDEX'], row['TOKEN'], row['TOKEN_INDEX']], axis=1)
    data['DEP_PAIRS'] = data.apply(lambda row: [row['DEP'], row['HEAD_TEXT'], row['TOKEN']], axis=1)
    
    return data

def feat_counts(text, kw):
    """
    Main function for getting the features from text input.
    Calls other functions to load dataset, clean text, counts features,
    removes negation phrases.

    Input:
            Text string
            Saved data of keywords and dependency pairs from pickle files

    Output:
            Feature counts
    """

    doc_clean_text = nlp(prep_data(text))
    spacy_df = spacy_table(doc_clean_text)
    # print(spacy_df)

    spacy_df = count_matches(kw['word_matches'], spacy_df) # simple dictionary lookups for 30 features
    spacy_df = get_dep_pairs(spacy_df) # Create 2 new columns - Negation linked and non-negation linked dependency pairs (5 features)
    spacy_df = count_spacy_matches(kw['spacy_pos'], spacy_df) # Dependency pairs not linked to negation (e.g. Acknowledgment, Agreement, Apology, Gratitude, Truth intensifier)
    spacy_df = count_spacy_matches(kw['spacy_noneg'], spacy_df, negations = False) # no negations required for disagreement and subjectivity
    spacy_df = count_matches(kw['spacy_neg_only'], spacy_df, inversion = True) # inverted emotional polarity for positive and negative emotions
    spacy_df = word_start(kw['word_start'], spacy_df) # count start word matches like conjunctions and affirmations
    spacy_df = bare_command(spacy_df) # Bare commands
    spacy_df = Question(spacy_df) # YesNo and Wh questions
    spacy_df = adverb_limiter(kw['spacy_tokentag'], spacy_df) # Adverb Limiters
    spacy_df = spacy_df.drop('DEP_PAIRS_POS', axis=1)
    spacy_df = spacy_df.drop('Negations', axis=1)
    
    # spacy_df.to_csv('output.csv', index = False)

    return spacy_df

def get_2024_politeness_strategy_features(text):

    df = feat_counts(text, kw)

    # Named columns to start and end
    start_column = "Agreement"
    end_column = "Adverb_Limiters"

    # Find column indices
    start_idx = df.columns.get_loc(start_column)
    end_idx = df.columns.get_loc(end_column)
    
    # Select columns within the range
    prevalence = df.iloc[:, start_idx:end_idx+1]  # Include the end column
    prevalence = prevalence.sum(axis=0)

    # Define the columns that will be used for politeness markers
    politeness_columns = df.columns[11:]  # Starting from the 12th column (index 11) REPLACE WITH COLUMN NAMES

    # Initialize dictionaries for politeness_markers and politeness_strategies
    politeness_markers = {}
    politeness_strategies = {}

    # Iterate over each of the politeness columns and construct the dictionaries
    for col in politeness_columns:
        # Filter rows where the politeness marker is 1
        # print(df[df[col] > 0])
        marker_rows = df[df[col] > 0][['TOKEN', 'HEAD_TEXT','SENT_NUM', 'WORD_NUM']].values.tolist()
        
        # Store the list of dictionaries for politeness markers
        politeness_markers[col] = marker_rows
        
        # Store the count of occurrences for politeness strategies
        if len(marker_rows) != 0:
            politeness_strategies[col] = int(df[col].sum())
        else:
            politeness_strategies[col] = 0
        
    politeness_strategies = {f"feature_politeness_=={key}==": value for key, value in politeness_strategies.items()}

    # Combine the two dictionaries into one
    meta = {
        'politeness_markers': politeness_markers,
        'politeness_strategies': politeness_strategies
    }

    return prevalence, meta

if __name__ == "__main__":
    
    start_time = time.process_time()
    
    text = 'Hello! I understand your perspective but I do not agree, I do concur. I believe. Just negate. Sorry. What was the question? Is it random? But I do not understand why this is an issue. I fucking hate this! I disagree so much. Not great. This is terribly annoying!'

    prevalence, meta = get_2024_politeness_strategy_features(text)
    
    print(prevalence)
    print(meta)
    # Pretty print the dictionary with indentation
    print(json.dumps(meta, indent=4))
    
    delta = round(time.process_time() - start_time, 3)
    print('Runtime: ', delta)
    