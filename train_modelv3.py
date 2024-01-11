import wikipediaapi
import gensim
from gensim.models import Word2Vec
import nltk
import string
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def convert_utf(text):
    text = text.replace('\u2018', "'").replace('\u2019', "'").replace('\u201C', "`").replace('\u201D', "`").replace('\u2013', '-').replace('\u2014', '-')
    return text

def preprocess_text(text):
    text = convert_utf(text)
    text = text.lower()
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    words = [word for sublist in words for word in sublist]  
    words = [word for word in words if word.isalpha() and word not in string.punctuation]
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    return words

def train_save_model(page_title, max_depth=2):
    user_agent = 'TomBCodingAssignment/1.0 (tb220110@nulondon.ac.uk)'
    wiki_wiki = wikipediaapi.Wikipedia(user_agent, 'en')

    def crawl_page(page, depth):
        if depth > max_depth:
            return []

        if page.exists():
            preprocessed_text = preprocess_text(page.text)
            linked_texts = []

            for link in page.links:
                linked_page = wiki_wiki.page(link)
                linked_texts.extend(crawl_page(linked_page, depth + 1))

            return [preprocessed_text] + linked_texts
        else:
            return []

    start_page = wiki_wiki.page(page_title)
    wikipedia_texts = crawl_page(start_page, depth=0)

    all_words = [word for sublist in wikipedia_texts for word in sublist]

    model = Word2Vec(sentences=[all_words], vector_size=100, window=5, min_count=1)
    model.save("word2vec.model")

    print(f"Word2Vec model trained on Wikipedia page: {page_title}")

page_title = "Color"
max_depth = 2
train_save_model(page_title, max_depth)
