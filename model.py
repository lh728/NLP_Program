stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
names = ['talk.politics.mideast', 'sci.electronics', 'soc.religion.christian', 'rec.motorcycles','comp.sys.mac.hardware','rec.sport.hockey']
docs_to_train = fetch_20newsgroups(subset = 'train', categories= names, shuffle = True, random_state = 1)
docs_to_test = fetch_20newsgroups(subset = 'test', categories= names, shuffle = True, random_state = 2)
df_train = pd.DataFrame(docs_to_train.data)
df_train['target'] = pd.Series(data=docs_to_test.target) 
df_test = pd.DataFrame(docs_to_test.data)
df_test['target'] = pd.Series(data=docs_to_test.target)
df_train.rename(columns = {0:'content'},inplace=True) 
df_test.rename(columns = {0:'content'},inplace=True)

# set a function to remove emails, newline chars and single quotes
def remove_sth(sentences):
      '''
      gensim.utils simple_preprocess method will do these steps:
      Step1: Separate each text with a space as a delimiter
      Step2: lowercase the words in the text
      Step3: Remove stop words
      Step4: Count the word frequency of each word and remove those words that appear only once in the text
      '''
      for i in sentences:
        i = re.sub('\S*@\S*\s?', '', i)  
        i = re.sub('\s+', ' ', i)  
        i = re.sub("\'", "", i) 
        i = gensim.utils.simple_preprocess(str(i), deacc=True) # deacc will remove punctuation
        yield(i)
data = df_train['content'].values.tolist()
words = list(remove_sth(data))
# Use models to form bigrams, trigrams. 
'''
gensim.models.Phrases method Phrase (collocation) detection. 
Automatically detect common phrases – aka multi-word expressions, word n-gram collocations – from a stream of sentences.
Detect phrases, based on collected collocation counts. 
Adjacent words that appear together more frequently than expected are joined together with the _ character.
It can be used to generate phrases on the fly, using the phrases[sentence] and phrases[corpus] syntax.
Phrases extract bigram phrases based on co-occurrence frequencies. Statistics based on co-occurrence: 
Affected by min_count and threshold, the larger the parameter setting, the more difficult it is to combine words into bigrams.
min_count ignores all collected words and bigrams with a total count below this value. Default is 5
threshold represents the threshold for forming phrases (higher means fewer phrases). 
If (cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold, then accept phrases of words a and b, where N is the total vocabulary size. Default is 10.0
'''
bigram = gensim.models.Phrases(words, threshold=100)  
trigram = gensim.models.Phrases(bigram[words], threshold=100)
'''
The purpose of Phraser is to reduce the memory consumption of phrases by discarding some Phasers model (phrases_model) states. 
You can use Phraser instead of Phrases if you don't need to update bigram statistics with new documents later.
After a one-time initialization, Phraser takes up less memory and is faster than using the Phrases model.
'''  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
# Next, lemmatize each word form to its root form, (but keep nouns, adjectives, verbs, and adverbs)
def process(words, stop_words, tags):
    words = [[i for i in simple_preprocess(str(j)) if i not in stop_words] for j in words]
    words = [bigram_mod[i] for i in words]
    words = [trigram_mod[bigram_mod[i]] for i in words]
    text = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # Use spacy.load() function to load language model, The disable parameter is a list of disabled processing pipelines
    for i in words:
        doc = nlp(" ".join(i)) 
        text.append([i.lemma_ for i in doc if i.pos_ in tags])
    '''
    Only keep pos as it contributes the most
    token pos_ tag is Coarse-grained part-of-speech from the Universal POS tag set(https://universaldependencies.org/u/pos/),
    These tags mark the core part-of-speech categories. To distinguish additional lexical and grammatical properties of words
    token lemma_ attribute means 'Base form of the token, with no inflectional suffixes'.
    '''
    text = [[j for j in simple_preprocess(str(i)) if j not in stop_words] for i in text]    
    return text
df_train3 = process(words,stop_words,['NOUN', 'ADJ', 'VERB', 'ADV']) 
# Now, create the LDA model
id2word = corpora.Dictionary(df_train3) # corpora dictionary: Key is the word in the dictionary, and its Val is the unique numeric ID corresponding to the word
'''
Doc2Bow is a method encapsulated in Gensim, mainly used to implement the Bow model
The Bag-of-words model (BoW model) first appeared in the fields of Natural Language Processing and Information Retrieval. 
The model ignores elements such as grammar and word order of the text, and regards it as a collection of several words, 
and the occurrence of each word in the document is independent. BoW uses an unordered set of words to express a piece of text or a document
'''
corpus = [id2word.doc2bow(i) for i in df_train3] 
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=6,random_state=3,update_every=1,
                                            chunksize=10,passes=10,alpha='symmetric',iterations=300,per_word_topics=True)
