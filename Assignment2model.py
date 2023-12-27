pip install wikipedia-api
pip install contractions
import wikipediaapi
import requests
import nltk
import unicodedata
import gensim
import string 
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import contractions
nltk.download('punkt')
nltk.download('stopwords')

user = 'WKDcodingassignment/1.0 (wk3218pgt@students.nulondon.ac.uk)'
wiki_wiki = wikipediaapi.Wikipedia(user, 'en') #initialize Wikipedia API in English
api_key = input("Please enter your Guardian API key:") #prompt for Guardian API key!!!!!!!!!!!!!

def get_category_pages_content(category_titles):
    combined_text = ""

    for category_title in category_titles:
        # Check if it's a category page
        category_page = wiki_wiki.page("Category:" + category_title)
        if category_page.exists():
            # Fetch articles from category
            for cat in category_page.categorymembers.values():
                if cat.ns == wikipediaapi.Namespace.MAIN:
                    combined_text += cat.text
        else:
            # Handle as a regular article
            article_page = wiki_wiki.page(category_title)
            if article_page.exists():
                combined_text += article_page.text

    return combined_text

category_titles = ["Computer Science", "Filtration", "Economy of Italy", "Football", "Amazons", "Ice bath", "Solstice", "Age of Aquarius", "Western culture", "Tropics", "Kinship", "Political spectrum", "Boy or girl paradox", "Hot and cold cognition", "Season", "Girls & Boys", "Temperature", "Global North and Global South", "Chinese law", "Central heating", "East End of London", "East–West dichotomy", "Short(finance)", "Body weight", "Thermostat", "Compass", "Cat–dog relationship", "State of matter", "Advertising", "Common sense", "United Nations", "Mr. Men", "Night", "Learning environment", "Rain", "Fire", "Seven deadly sins", "Emotion", "White", "Art", "Pets", "Opposite (semantics)", "Adjectives", "Family", "Money", "Work (human activity)", "Health", "Predation", "Culture", "Biology", "Charles Dickens", "A Vindication of the Rights of Woman", "Grammatical gender", "Presidency of Donald Trump", "Feminism", "Nelson Mandela", "Hip hop music", "Mexico", "Eastenders", "Porn", "Modern era", "Millenials", "Student–teacher ratio", "Cat people and dog people", "Global cuisine", "Earth", "Artificial intelligence", "Sport", "English language", "Opposite(semantics)", "Nature", "Human","Ageing", "Black"]
wiki_category_content = get_category_pages_content(category_titles)

pip install requests
import requests

def fetch_guardian_articles(api_key, queries): #define function to fetch articles from Guardian API
    Guardian_url = "https://content.guardianapis.com/search"
    articles_text = ""  #initialize variable to hold combined article text
    for query in queries: #iterate over each query term
     
        params = {                           #set up parameters for the request
            'api-key': api_key,              #API key for authentication
            'q': query,                      #current query term
            'page-size': 200,                #number of articles to fetch
            'show-fields': 'body'            #specify that only main body of articles to be returned
        }
        
        response = requests.get(Guardian_url, params=params)    #make API request
        data = response.json() #convert JSON response into Python

        for article in data['response']['results']: #Guardian safety check/extract article bodies and add to combined text
            if 'body' in article['fields']:
                articles_text += article['fields']['body']
    
    return articles_text    # Return the combined text of all articles
query_terms = ["Culture", "Sport", "Lifestyle", "Obituaries", "Corrections and clarifications", "Home", "Opinion", "Film", "Business", "Society"]
guardian_content = fetch_guardian_articles(api_key, query_terms)

def remove_html_tags(text):
    """Remove HTML tags from text and replace them with a space."""
    return re.sub(r'<.*?>', ' ', text)

cleaned_guardian_content = remove_html_tags(guardian_content)
cleaned_guardian_content = re.sub(r'\s+', ' ', cleaned_guardian_content).strip() #remove trailing and multiple white spaces

def preprocessed_text(text):
    text = contractions.fix(text)  #expand contractions
    text = text.lower()  #lowercase characters
    text = re.sub(r'\d+', '', text)  #remove any numbers
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')  #remove non-ASCII characters
    text = text.replace ("-"," ")
    text = re.sub(r'[^\w\s.!?]', '', text) #remove standard punctuation except for sentence-ending ones. 
    text = text.replace('nights','night').replace ('color', 'colour').replace('nugent', 'student').replace('womans', 'woman').replace('easts', 'east').replace('liquids','liquid').replace('solids','solid').replace('evening','').replace('mid', '').replace('dsm', 'physical').replace('cheaper', 'cheap').replace('newcomer','boy').replace('vapor','gas').replace('mans','man').replace('fathers','father').replace('bulb','dry')
    return text

cleaned_guardian_text = preprocessed_text(cleaned_guardian_content) #clean both datasets using preprocessed_text defined function
cleaned_wiki_text = preprocessed_text(wiki_category_content)

def tokenize_and_clean(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        cleaned_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        cleaned_sentences.append(cleaned_tokens)
    return cleaned_sentences

combined_content = cleaned_guardian_text + " " + cleaned_wiki_text #combine datasets

final_dataset = tokenize_and_clean(combined_content) #tokenise combined dataset using tokenize_and_clean defined function

model = gensim.models.Word2Vec(vector_size=100, min_count=18, window=15, sg=0)
model.build_vocab(final_dataset, update=False) #Build vocabulary from the dataset
model.train(final_dataset, total_examples=model.corpus_count, epochs=10) #train the model
model.save('./model_test') #save trained model