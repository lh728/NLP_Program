# Next we figure out a way to extract the main topic of each sentence
def topics(ldamodel, corpus, texts):
  df = pd.DataFrame()
  for _, j in enumerate(ldamodel[corpus]):
    a = j[0] if ldamodel.per_word_topics else j # When per_word_topics is True, also get a list of topics to which each word in the document belongs           
    a = sorted(a, key=lambda x: (x[1]), reverse=True)
  # Dominant topic, Perc of Contribution (Percentage contribution of each topic to documentation) and Keywords 
    for i, (topic_num, prop) in enumerate(a):
      if i == 0:  
        topic_nums = ldamodel.show_topic(topic_num)
        topic_keywords = ", ".join([i for i, _ in topic_nums])
        df = df.append(pd.Series([int(topic_num), round(prop,4), topic_keywords]), ignore_index=True)
      else:
        break
  df.columns = ['Main_Topic', 'Contribution', 'Keywords']
  # Add original text to the end of the output
  contents = pd.Series(texts)
  df = pd.concat([df, contents], axis=1)
  return(df)
df_train4 = topics(lda_model, corpus, df_train3)
df_topics = df_train4.reset_index()
df_topics.columns = ['Num_Doc', 'Main_Topic', 'Contribution', 'Keywords', 'Texts']
df_topics
# Now I want to get the most typical sentences for each topic
df_topics_sorted = pd.DataFrame()
df_topics_out = df_train4.groupby('Main_Topic')
for i, j in df_topics_out:
  df_topics_sorted = pd.concat([df_topics_sorted,j.sort_values(['Contribution'], ascending=False).head(1)],axis=0)   
df_topics_sorted.reset_index(drop=True, inplace=True)
df_topics_sorted.columns = ['Num_Doc', "Contribution", "Keywords", "Representative Text"]
df_topics_sorted

# show how many words in each document
lens = [len(d) for d in df_topics['Texts']]
plt.figure(figsize=(14,7), dpi=160,facecolor='white')
plt.hist(lens, bins = 500)
plt.title('Words Distribution in each Document')
plt.text(125, 350, "Mean   : " + str(round(np.mean(lens))))
plt.text(125, 340, "Median : " + str(round(np.median(lens))))
plt.text(125, 330, "Std   : " + str(round(np.std(lens))))
plt.text(125, 320, "Min    : " + str(round(np.min(lens))))
plt.text(125, 310, "Max  : " + str(round(np.max(lens))))
plt.gca().set(xlim=(0, 500), ylabel='Number of the Documents', xlabel='Words Count')
plt.xticks(np.linspace(0,1000,9))
plt.show()

col = ['red','blue','black','green','purple','orange']
fig, ax = plt.subplots(2,3,figsize=(12,8), dpi=160, sharex=True, sharey=True,facecolor = 'white')
for i, ax in enumerate(ax.flatten()):    
  df_topics2 = df_topics.loc[df_topics['Main_Topic'] == i , :]
  doc_lens = [len(d) for d in df_topics2['Texts']]
  ax.hist(doc_lens, bins = 500, color=col[i])
  ax.tick_params(axis='y', labelcolor=col[i], color=col[i])
  sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
  ax.set(xlim=(0, 500), xlabel='Words Count')
  ax.set_ylabel('Number of Documents', color=col[i])
  ax.set_title('Topic: '+str(i), color=col[i])
fig.tight_layout()
fig.subplots_adjust(top=0.90)
plt.xticks(np.linspace(0,1000,9))
fig.suptitle('Words Counts by Main Topic Distribution', fontsize=22)
plt.show()

# wordcloud
cloud = WordCloud(stopwords=stop_words,
                  background_color='white',color_func=lambda *args, **kwargs: col[i])
some_topics = lda_model.show_topics(formatted=False)
fig, ax = plt.subplots(2, 3, figsize=(10,5), sharex=True, sharey=True,facecolor = 'white',dpi = 600)
for i, ax in enumerate(ax.flatten()):
  fig.add_subplot(ax)
  topic_words = dict(some_topics[i][1])
  cloud.generate_from_frequencies(topic_words)
  plt.gca().imshow(cloud)
  plt.gca().set_title('Topic ' + str(i))
  plt.gca().axis('off')
plt.tight_layout()
plt.show()

#  the importance (weights) of the keywords matters
topicss = lda_model.show_topics(formatted=False)
flat = [j for i in df_train3 for j in i]
counter = Counter(flat)
c = []
for i, j in topicss:
  for w, h in j:
    c.append([w, i , h, counter[w]])
df_train5 = pd.DataFrame(c, columns=['word', 'topic', 'importance', 'word_count']) 
fig, ax = plt.subplots(2, 3, figsize=(12,10), sharey=True, dpi=600,facecolor = 'white')
cols = [color for _, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(ax.flatten()):
  ax.bar(x='word', height="word_count", data=df_train5.loc[df_train5['topic']==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
  ax_twinx = ax.twinx()
  ax_twinx.bar(x='word', height="importance", data=df_train5.loc[df_train5['topic']==i, :], color=cols[i], width=0.2, label='Weights')
  ax.set_ylabel('Word Count', color=cols[i])
  ax_twinx.set_ylim(0, 0.030); 
  ax.set_ylim(0, 3000)
  ax.set_title('Topic: ' + str(i), color=cols[i])
  ax.tick_params(axis='y', left=False)
  ax.set_xticklabels(df_train5.loc[df_train5['topic']==i, 'word'], rotation=30, horizontalalignment= 'right')
  ax.legend(loc='upper left'); 
  ax_twinx.legend(loc='upper right')
fig.suptitle('Keywords Word Count and Importance')  
fig.tight_layout(w_pad=2)    
plt.show()

# t-SNE clustering
weights = []
for _, i in enumerate(lda_model[corpus]):
  weights.append([w for _, w in i[0]])   
df_train6 = pd.DataFrame(weights).fillna(0).values
df_train6 = df_train6[np.amax(df_train6, axis=1) > 0.4]
d = np.argmax(df_train6, axis=1)
tsne_model = TSNE(verbose=1, random_state=4, init='pca')
tsne_lda = tsne_model.fit_transform(df_train6)
colors = np.array(cols)
e = figure(title="t-SNE Clustering", 
              plot_width=1000, plot_height=600)
e.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=colors[d])
show(e)
# pyLDAVis plot
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary=lda_model.id2word)
pyLDAvis.save_html(vis,'lda.html')
