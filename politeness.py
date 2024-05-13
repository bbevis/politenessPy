import os
import pandas as pd
import regex
import re
import keywords
import en_core_web_sm
import time
nlp = en_core_web_sm.load()


class feature_extraction:
    
    def __init__(self, text):
        
        self.main_features = ['Acknowledgement', 'Agreement', 'Hedges', 'Negation', 'Positive_Emotion', 'Subjectivity', 'Adverb_Limiter', 'Disagreement', 'Negative_Emotion']
        self.main_features_pos = ['Acknowledgement', 'Agreement', 'Hedges', 'Positive_Emotion', 'Subjectivity']
        self.main_features_neg = ['Negation', 'Negative_Emotion', 'Adverb_Limiter', 'Disagreement']
        self.kw = keywords.kw

    ########################################
    # prepping & cleaning data
    ########################################

    def clean_text(self, text):

        orig = ["let's", "i'm", "won't", "can't", "shan't", "'d",
                "'ve", "'s", "'ll", "'re", "n't", "u.s.a.", "u.s.", "e.g.", "i.e.",
                "‘", "’", "“", "”", "100%", "  ", "mr.", "mrs.", "dont", "wont"]

        new = ["let us", "i am", "will not", "cannot", "shall not", " would",
            " have", " is", " will", " are", " not", "usa", "usa", "eg", "ie",
            "'", "'", '"', '"', "definitely", " ", "mr", "mrs", "do not", "would not"]

        for i in range(len(orig)):
            text = text.replace(orig[i], new[i])

        return text

    def prep_simple(self, text):

        # text cleaning

        t = text.lower()
        t = self.clean_text(t)
        t = re.sub(r"[.?!]+\ *", "", t)  # spcifially replace punctuations with nothing
        t = re.sub('[^A-Za-z,]', ' ', t)  # all other special chracters are replaced with blanks

        return t

    def sentence_split(self, doc):

        # doc = nlp(text)
        sentences = [str(sent) for sent in doc.sents]
        sentences = [' ' + self.prep_simple(str(s)) + ' ' for s in sentences]

        return sentences

    def sentence_pad(self, doc):

        sentences = self.sentence_split(doc)

        return ''.join(sentences)

    ########################################
    # extracting features
    ########################################

    def count_matches(self, keywords, doc):
        """
        For a given piece of text, search for the number if keywords from a prespecified list

        Inputs: Prespecified list (keywords), text

        Outputs: Counts of keyword matches
        """

        text = self.sentence_pad(doc)

        # print(text)

        key_res = []
        phrase2_count = []

        for key in keywords:

            key_res.append(key)
            counter = 0

            check = any(item in text for item in keywords[key])

            if check == True:

                for phrase in keywords[key]:

                    phrase_count = text.count(phrase)

                    if phrase_count > 0:

                        counter = counter + phrase_count

            phrase2_count.append(counter)

        res = pd.DataFrame([key_res, phrase2_count], index=['Features', 'Counts']).T

        return res


    def get_dep_pairs(self, doc):
        """
        Uses spaCy to find list of dependency pairs from text.
        Performs negation handling where by any dependency pairs related to a negated term is removed

        Input: Text

        Outputs: Dependency pairs from text that do not have ROOT as the head token or is a negated term
        """

        dep_pairs = [[token.dep_, token.head.text, token.head.i, token.text, token.i] for token in doc]


        negations = [dep_pairs[i] for i in range(len(dep_pairs)) if dep_pairs[i][0] == 'neg']
        token_place = [dep_pairs[i][2] for i in range(len(dep_pairs)) if dep_pairs[i][0] == 'neg']

        dep_pairs2 = []

        if len(negations) > 0:

            for j in range(len(dep_pairs)):

                if dep_pairs[j][2] not in token_place and dep_pairs[j] not in dep_pairs2:
                    dep_pairs2.append(dep_pairs[j])

        else:
            dep_pairs2 = dep_pairs.copy()

        dep_pairs2 = [[dep_pairs2[i][0], dep_pairs2[i][1], dep_pairs2[i][3]] for i in range(len(dep_pairs2))]

        return dep_pairs2, negations


    def get_dep_pairs_noneg(self, doc):
        """
        No negation is done as we are only searching 'hits'
        """
        return [[token.dep_, token.head.text, token.text] for token in doc]


    def count_spacy_matches(self, keywords, dep_pairs):
        """
        When searching for key words are not sufficient, we may search for dependency pairs.
        Finds any-prespecified dependency pairs from text string and outputs the counts

        Inputs:
                Dependency pairs from text
                Predefined tokens for search in dependency heads

        Output:
                Count of dependency pair matches
        """

        key_res = []
        phrase2_count = []

        for key in keywords:
            key_res.append(key)
            counter = 0
            check = any(item in dep_pairs for item in keywords[key])

            if check == True:

                for phrase in keywords[key]:
                    if phrase in dep_pairs:
                        for dep in dep_pairs:
                            if phrase == dep:
                                counter = counter + 1

            phrase2_count.append(counter)

        res = pd.DataFrame([key_res, phrase2_count], index=['Features', 'Counts']).T

        return res


    def token_count(self, doc):

        # Counts number of words in a text string
        return len([token for token in doc])


    def bare_command(self, doc):
        """
        Check the first word of each sentence is a verb AND is contained in list of key words

        Output: Count of matches
        """

        keywords = set([' be ', ' do ', ' please ', ' have ', ' thank ', ' hang ', ' let '])

        # Returns first word of every sentence along with the corresponding POS
        first_words = [' ' + self.prep_simple(str(sent[0])) + ' ' for sent in doc.sents]

        POS_fw = [sent[0].tag_ for sent in doc.sents]

        # returns word if word is a verb and in list of keywords
        bc = [b for a, b in zip(POS_fw, first_words) if a == 'VB' and b not in keywords]

        return len(bc)

    def Question(self, doc):
        """
        Counts number of prespecified question words
        """

        keywords = set([' who ', ' what ', ' where ', ' when ', ' why ', ' how ', ' which '])
        tags = set(['WRB', 'WP', 'WDT'])

        sentences = [str(sent) for sent in doc.sents if '?' in str(sent)]

        all_qs = len(sentences)

        n = 0
        for i in range(len(sentences)):
            whq = [token.tag_ for token in nlp(sentences[i]) if token.tag_ in tags]

            if len(whq) > 0:
                n += 1

        return all_qs - n, n


    def word_start(self, keywords, doc):
        """
        Find first words in text such as conjunctions and affirmations
        """

        key_res = []
        phrase2_count = []

        for key in keywords:

            first_words = [' ' + self.prep_simple(str(sent[0])) + ' ' for sent in doc.sents]
            #first_words = [prep.prep_simple(str(fw)) for fw in first_words]
            cs = [w for w in first_words if w in keywords[key]]

            phrase2_count.append(len(cs))
            key_res.append(key)

        res = pd.DataFrame([key_res, phrase2_count], index=['Features', 'Counts']).T
        return res


    def adverb_limiter(self, keywords, doc):
        """
        Search for tokens that are advmod and in the prespecifid list of words
        """

        tags = [token.dep_ for token in doc if token.dep_ == 'advmod' and
                str(' ' + str(token) + ' ') in keywords['Adverb_Limiter']]

        return len(tags)


    ########################################
    # feature scores
    ########################################


    def feat_counts(self, text, kw):
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

        text = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)
        text = text.lstrip()

        clean_text = self.prep_simple(text)

        doc_text = nlp(text)
        doc_clean_text = nlp(clean_text)

        # Count key words and dependency pairs with negation
        kw_matches = self.count_matches(kw['word_matches'], doc_text)

        dep_pairs, negations = self.get_dep_pairs(doc_clean_text)
        dep_pair_matches = self.count_spacy_matches(kw['spacy_pos'], dep_pairs)

        dep_pairs_noneg = self.get_dep_pairs_noneg(doc_clean_text)
        disagreement = self.count_spacy_matches(kw['spacy_noneg'], dep_pairs_noneg)

        neg_dp = set([' ' + i[1] + ' ' for i in negations])
        neg_only = self.count_spacy_matches(kw['spacy_neg_only'], neg_dp)

        # count start word matches like conjunctions and affirmations
        start_matches = self.word_start(kw['word_start'], doc_text)

        scores = pd.concat([kw_matches, dep_pair_matches, disagreement, start_matches, neg_only])
        scores = scores.groupby('Features').sum().sort_values(by='Counts', ascending=False)
        scores = scores.reset_index()

        # add remaining features
        bc = self.bare_command(doc_text)
        scores.loc[len(scores)] = ['Bare_Command', bc]

        ynq, whq = self.Question(doc_text)

        scores.loc[len(scores)] = ['YesNo_Questions', ynq]
        scores.loc[len(scores)] = ['WH_Questions', whq]

        adl = self.adverb_limiter(kw['spacy_tokentag'], doc_text)
        scores.loc[len(scores)] = ['Adverb_Limiter', adl]

        scores = scores.sort_values(by='Counts', ascending=False)

        tokens = self.token_count(doc_text)
        scores.loc[len(scores)] = ['Token_count', tokens]

        return scores

    def normalise_scores(self, scores):
        """
        Divides feature counts by 100 words/tokens
        """

        token_count = list(scores['Counts'][scores['Features'] == 'Token_count'])[0]
        scores['Counts_norm'] = scores['Counts'] / token_count * 100

        return scores

    def extract_features(self, text):

        start_time = time.process_time()
        
        scores = self.feat_counts(text, self.kw)
        scores = self.normalise_scores(scores)
        scores = scores.rename(columns={"Features": "features", "Counts": "feature_counts", "Counts_norms": "features_per_wordcount"})
        
        delta = round(time.process_time() - start_time, 3)
        print('Runtime: ', delta)

        return scores

if __name__ == "__main__":
    
    text = 'Hello! I understand your perspective but you are so dumb.'
    recp = feature_extraction(text)

    features = recp.extract_features(text)
    print(features)
    