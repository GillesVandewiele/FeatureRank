import pandas as pd
import gensim, logging, os, sys

df = pd.read_csv('animals_dbpedia_uris.csv')

model = gensim.models.Word2Vec.load('db2vec_sg_200_5_25_5')

tag_dict = []
for _class, tag in zip(df['class'], df['rdf2vec_tag']):
	try:
		tag_dict.append({'class': _class, 'tag': tag, 'vector': list(model.wv[tag])})
	except:
		print(tag, 'failed')

output = pd.DataFrame.from_dict(tag_dict)
output.to_csv('vectors.csv')