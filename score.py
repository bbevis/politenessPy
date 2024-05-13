
import pandas as pd
import numpy as np
import politeness
import pretrained_weights


class receptiveness_score:
    
    def __init__(self, text):
        
        self.recept_feats = ['Negation', 'Reasoning', 'Positive_Emotion', 'Negative_Emotion', 'Hedges',
            'Agreement', 'Disagreement', 'Acknowledgement', 'Subjectivity', 'Adverb_Limiter']
        
        self.text = text

    def predict_score(self):
        
        # Use politeness script to extract linguistic features
        recp = politeness.feature_extraction(self.text)
        features = recp.extract_features(self.text).set_index('features').loc[self.recept_feats]
        X = features['feature_counts'].to_list()
        X.insert(0, 1)

        # Extract the weights as a numpy array
        weights = pd.DataFrame(pretrained_weights.weights, index = ['value']).T
        Intercept = weights.loc['Intercept']['value']
        weights = weights.loc[self.recept_feats]
   
        weights = weights.values.flatten().tolist()
        weights.insert(0, Intercept)

        # Predict the score using the trained model
        return round(sum(x * w for x, w in zip(X, weights)),3)

if __name__ == "__main__":
    
    text = 'Hello! I understand your perspective but I still think it is rediculous.'
    recp_score = receptiveness_score(text)

    score = recp_score.predict_score()
    print(score)