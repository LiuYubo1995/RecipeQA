import gensim
from gensim.test.utils import get_tmpfile
fname = get_tmpfile('my_doc2vec_model') 
model = gensim.models.doc2vec.Doc2Vec.load(fname)
print(model.infer_vector(['i', 'am', 'a', 'student']))