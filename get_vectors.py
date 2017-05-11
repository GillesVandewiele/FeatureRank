import pandas as pd
import gensim, logging, os, sys

df = pd.read_csv('zoo_metadata.csv')

model = gensim.models.Word2Vec.load('db2vec_sg_200_5_25_5')

tag_dict = []
for tag in df['rdf2vec_tag']:
	tag_dict.append({'tag': tag, 'vector': model.wv[tag]})

output = pd.DataFrame.from_dict(tag_dict)
output.to_csv('vectors.csv')