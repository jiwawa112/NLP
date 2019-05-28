# Build Search Engine
# Input:Words
# Output:Documents

import numpy as np
from functools import reduce
from operator import and_

def naive_search(keywords):
    news_ids = [i for i in enumerate(news_content) if all(w in n for w in keywords)]
    # O(D * w)

len(news_content)

### Input word ->the documents which contain this word
set(np.where(X[0].toarray()[0][0]))

X.shape

X.transpose()

def search_engine(query):
    """
    @query is the searched words,splited by space
    @ return is the related documents which ranked by tfidf similarity
    """
    words = query.split()

    query_vec = vectorized.transform([' '.join(words)]).toarray()[0]

    candidates_ids = [word_2_id[w] for w in words]

    documents_ids = {
        set(np.where(transposed_x[_id])[0]) for _id in candidates_ids
    }

    merged_documents = reduce(and_,documents_ids)
    # we could know the documents which contain these words

    sorted_documents_id = sorted(merged_documents,key=lambda i:distance(query_vec,x[i].toarray()))

    return sorted_documents_id


