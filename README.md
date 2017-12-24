# Wine-Points
# Loading required package: NLP
install.packages(c('tm', 'SnowballC', 'wordcloud', 'topicmodels'))

library(tm);

library(SnowballC)

library(wordcloud)

# Acquiring the data set

wine =  read.csv("winemag-data-130k-v2.csv",stringsAsFactors = F, row.names = 1);

str(wine)

nrow(wine)

NCOL(wine)

# Initializing the target variable and Corpus
rating_bound <- 90

wine$rating_above_90 <- as.factor(wine$points > rating_bound)

wine$variety <- as.factor(wine$variety)

head(wine, 5)

wine_corpus = Corpus(VectorSource(wine$description))

# Cleanig of the corpus
wine_corpus = tm_map(wine_corpus, content_transformer(tolower))

wine_corpus = tm_map(wine_corpus, removeNumbers)

wine_corpus = tm_map(wine_corpus, removePunctuation)

wine_corpus = tm_map(wine_corpus, removeWords, c("the", "and", stopwords("english")))

wine_corpus = tm_map(wine_corpus, stemDocument)

wine_corpus = tm_map(wine_corpus, stripWhitespace)

inspect(wine_corpus[1])

# Creating the Document term matrix
wine_dtm_tfidf = DocumentTermMatrix(wine_corpus, control = list(weighting = weightTfIdf))

wine_dtm_tfidf = removeSparseTerms(wine_dtm_tfidf, 0.98)


# Creating the worldcloud of most frequent words

freq = data.frame(sort(colSums(as.matrix(wine_dtm_tfidf)), decreasing=TRUE))

wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(1, "Dark2"))

wine = cbind(wine, as.matrix(wine_dtm_tfidf))


# Following command will display the words appearing at least two thousand times in the sms_dtm_train matrix:
findFreqTerms(wine_dtm_tfidf, 2000)

wine_freq_words <- findFreqTerms(wine_dtm_tfidf, 2000)

wine_dtm_tfidf = wine_dtm_tfidf[ , wine_freq_words]

typeof(wine_dtm_tfidf)

ncol(wine_dtm_tfidf)

# Combining the DTM and data set
wine = cbind(wine, as.matrix(wine_dtm_tfidf))

set.seed(123)

id_train <- sample(nrow(wine),nrow(wine)*0.70)

wine_train <- wine[id_train,]

wine_test <- wine[-id_train,]

wine$rating_above_90 = as.factor(wine$rating_above_90)

# Training the algorithm
install.packages(c('rpart', 'rpart.plot'))

library(rpart)

library(rpart.plot)

wine.tree = rpart(rating_above_90 ~ acid + appl + aroma + dri + fruit + palat + berri + fruiti + red + ripe +tannin +
                    wine + crisp + flavor + finish + note + fresh + show + soft + spice + textur + drink
                  + black + cherri + oak + rich + light + sweet +  price + variety , data = wine_train)

wine.pred = predict(wine.tree, wine_test, type = "class")

wine.pred


# Package for confusion matrix
library(caret)

confusionMatrix(wine.pred, wine_test$rating_above_90)









