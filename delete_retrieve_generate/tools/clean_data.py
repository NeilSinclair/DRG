import pandas as pd
import re

files = ['yelp_train_15.0', 'yelp_train_15.1', 'yelp_test_15_short.1', 'yelp_test_15_short.0']

for f in files:
    x = pd.read_csv(f'./data/yelp/{f}')
    x = x.apply(lambda x: re.sub(r'"', '', str(x)))
    x = x.apply(lambda x: re.sub(r'\.', ' .', str(x)))
    x = x.apply(lambda x: re.sub(r'!', ' !', str(x)))
    x = x.apply(lambda x: re.sub(r'\?', ' ?', str(x)))
    x = x.apply(lambda x: re.sub(r' {2,}', ' ', str(x)))

    x.to_csv(f'./data/yelp/{f}', index=False)

