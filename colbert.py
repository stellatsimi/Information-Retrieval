!git -C ColBERT/ pull || git clone https://github.com/stanford-futuredata/ColBERT.git
import sys; sys.path.insert(0, 'ColBERT/')

try: # When on google Colab, let's install all dependencies with pip.
    import google.colab
    !pip install -U pip
    !pip install -e ColBERT/['faiss-gpu','torch']
except Exception:
  import sys; sys.path.insert(0, 'ColBERT/')
  try:
    from colbert import Indexer, Searcher
  except Exception:
    print("If you're running outside Colab, please make sure you install ColBERT in conda following the instructions in our README. You can also install (as above) with pip but it may install slower or less stable faiss or torch dependencies. Conda is recommended.")
    assert False

import colbert

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection


import os
import glob

collection = []
for filename in glob.glob("/content/drive/MyDrive/docs/*"):
  with open(os.path.join(os.getcwd(), filename), "r") as f:
    text=f.read().replace("\n", " ")
    line = text
    collection.append(line)

queries = []
with open(os.path.join(os.getcwd(), "/content/drive/MyDrive/Queries_20"), "r") as f:
  text = f.read().split("\n")
  for i in text:
    queries.append(i)


f'Loaded {len(queries)} queries and {len(collection):,} passages'


nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300 # truncate passages at 300 tokens
max_id = 1000
dataset = "drive"
datasplit = "MyDrive"

index_name = f'{dataset}.{datasplit}.{nbits}bits'

checkpoint = 'colbert-ir/colbertv2.0'

with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                # Consider larger numbers for small datasets.

    indexer = Indexer(checkpoint=checkpoint, config=config)
    indexer.index(name=index_name, collection=collection[:max_id], overwrite=True)

indexer.get_index() # You can get the absolute path of the index, if needed.


with Run().context(RunConfig(experiment='notebook')):
    searcher = Searcher(index=index_name, collection=collection)


listOfDocNames = []
for filename in glob.glob("/content/drive/MyDrive/docs/*"):
  listOfDocNames.append(os.path.basename(filename))

query = queries[0] # try with an in-range query or supply your own
print(f"#> {query}")

# Find the top-3 passages for this query
results = searcher.search(query, k=10)

# Print out the top-k retrieved passages
for passage_id, passage_rank, passage_score in zip(*results):
    print(f"\t [{passage_rank}] \t\t {passage_score:.1f} {listOfDocNames[passage_id]} \t\t {searcher.collection[passage_id]}")