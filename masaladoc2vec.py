import gensim
import json

with open('data.json') as outfile:
    data = json.load(outfile)

print 'Loading JSON'
sentences = []
for point in data:
    sentences.append(point['title'])
docs = []
for sent in sentences:
    docs.append(sent.split('|')[0])

print 'Normalizing and Cleaning Data'



#ADD FUNCTION FOR NORMALIZING HERE
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text



fina_doc = []
for doc in docs:
    fina_doc.append(normalize_text(doc))

from gensim.models.doc2vec import LabeledSentence
labsen = []
uid = 0
for sen in fina_doc:
    labsen.append(LabeledSentence(sen.split(), ['%s' %uid]))
    uid+=1

print 'Converting to LabeledSentence Objects'

model = gensim.models.Doc2Vec(size=300, window=7, min_count=3, workers=11,alpha=0.025, min_alpha=0.025)

print 'Building Vocabulary'

model.build_vocab(labsen)

print 'Training Model'

for epoch in range(10):
                print 'Training EPOC:{}, Current Alpha value: {}'.format(epoch,model.alpha)
                model.train(labsen)
                model.alpha -= 0.002 # decrease the learning rate
                model.min_alpha = model.alpha # fix the learning rate, no deca
                model.train(labsen)


print 'Model Ready'
sent = int(input('Which sentence number to see for similars? '))
sims = model.docvecs.most_similar(sent)
print 'Similar Sentences:'
print sims
save = int(input('Enter 1 to save model:'))
if save==1:
	model.save('masalaherb.doc2vec')



