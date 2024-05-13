
import pandas as pd
import numpy as np
import politeness
import score

text = 'Hello! I understand your perspective but I still think it is rediculous.'
recp_score = score.receptiveness_score(text)

# score = recp_score.predict_score()
print(recp_score.predict_score())