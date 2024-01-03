pip install wikipedia-api
pip install contractions
import wikipediaapi #for accessing wikipedia content
import nltk #Imports Natural Language Toolkit (NLTK) for working with human language data
import unicodedata #for processing and normalizing Unicode strings
import gensim #open-source library for unsupervised topic modeling and natural language processing
import re #Python's built-in package for regular expressions
from nltk.corpus import stopwords #imports`stopwords` function from NLTK.
from nltk.tokenize import word_tokenize #imports function for splitting string into list of words
from nltk.tokenize import sent_tokenize #imports function for splitting string into list of sentences.
import contractions #module for exapanding contractions
nltk.download('punkt') #prerequisite for NLTK's word and sentence tokenizer functions
nltk.download('stopwords') #downloads a set of common stopwords
    
user = 'WKDcodingassignment/1.0 (wk3218pgt@students.nulondon.ac.uk)' #user identifier for Wikipedia requests
wiki_wiki = wikipediaapi.Wikipedia(user, 'en') #initialize Wikipedia API in English
api_key = input("Please enter your Guardian API key:") #prompt for Guardian API key!!!!!!!!!!!!

def get_category_pages_content(category_titles): #fetches specified Wikipedia content and combines pages into single string
    combined_text = ""

    for category_title in category_titles: #check if a category page
        category_page = wiki_wiki.page("Category:" + category_title)
        if category_page.exists():  #if a category page iterate over its members and fetch articles
            for cat in category_page.categorymembers.values():
                if cat.ns == wikipediaapi.Namespace.MAIN:
                    combined_text += cat.text #concatenate text of category articles
        else:  #otherwise handle as a regular Wikipedia article
            article_page = wiki_wiki.page(category_title)
            if article_page.exists():
                combined_text += article_page.text #and add article to combined text

    return combined_text

category_titles = ["Computer Science", "Filtration", "Economy of Italy", "The Addams Family (musical)", "American Civl War", "Football", "Sadness", "Amazons", "Ice bath", "Solstice", "Age of Aquarius", "Western culture", "Tropics", "Kinship", "Political spectrum", "Boy or girl paradox", "Hot and cold cognition", "Season", "Girls & Boys", "Temperature", "Global North and Global South", "Chinese law", "Central heating", "East End of London", "East–West dichotomy", "Short(finance)", "Body weight", "Thermostat", "Compass", "Cat–dog relationship", "State of matter", "Advertising", "Common sense", "United Nations", "Mr. Men", "Night", "Learning environment", "Rain", "Fire", "Seven deadly sins", "Emotion", "White", "Art", "Pets", "Opposite (semantics)", "Adjectives", "Family", "Money", "Work (human activity)", "Health", "Predation", "Culture", "Biology", "Charles Dickens", "A Vindication of the Rights of Woman", "Grammatical gender", "Presidency of Donald Trump", "Feminism", "Nelson Mandela", "Hip hop music", "Mexico", "Eastenders", "Porn", "Modern era", "Millenials", "Student–teacher ratio", "Cat people and dog people", "Global cuisine", "Earth", "Artificial intelligence", "Sport", "English language", "Opposite(semantics)", "Nature", "Human","Ageing", "Black"]
#my list of categories
wiki_category_content = get_category_pages_content(category_titles) #compiles and stores Wikipedia content from listed categories

pip install requests
import requests #necessary for making HTTP requests

def fetch_guardian_articles(api_key, queries): #define function to fetch articles from Guardian API
    Guardian_url = "https://content.guardianapis.com/search"
    articles_text = ""  #initialise variable to hold combined article text
    for query in queries: #iterate over each query term
     
        params = {                           #sets up parameters for Guardian request
            'api-key': api_key,              #API key for authentication
            'q': query,                      #query term
            'page-size': 200,                #number of articles to fetch
            'show-fields': 'body'            #specifies that only main body of articles to be returned
        }
        
        response = requests.get(Guardian_url, params=params)    #make API request
        data = response.json() #convert JSON response into Python

        for article in data['response']['results']: #Guardian safety check/extract article bodies and add to combined text
            if 'body' in article['fields']:
                articles_text += article['fields']['body']
    
    return articles_text    #return the combined text of all articles
query_terms = ["Culture", "Sport", "Lifestyle", "Obituaries", "Corrections and clarifications", "Home", "Opinion", "Film", "Business", "Society"]
#specified query terms
guardian_content = fetch_guardian_articles(api_key, query_terms) #compiles and stores Guardian content from listed categories

def remove_html_tags(text):  #defined function for removing html in a way that does not concatenate words separated by these tags
    return re.sub(r'<.*?>', ' ', text) #replaces all html tags with single space with re module 

cleaned_guardian_content = remove_html_tags(guardian_content) #remove html tags
cleaned_guardian_content = cleaned_guardian_content.strip() #remove trailing and white spaces

def preprocessed_text(text):
    text = contractions.fix(text)  #expand contractions
    text = text.lower()  #lowercase characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')  #remove non-ASCII characters
    text = text.replace ("-"," ") #removes hyphens and adds space
    text = text.replace('nights','night').replace('africas', 'africa').replace ('color', 'colour').replace('nugent', 'student').replace('womans', 'woman').replace('easts', 'east').replace('liquids','liquid').replace('solids','solid').replace('evening','').replace('mid', '').replace('dsm', 'physical').replace('cheaper', 'cheap').replace('newcomer','boy').replace('vapor','gas').replace('mans','man').replace('fathers','father').replace('bulb','dry')
    #bespoke removal of some words including plurals and US spellings
    return text

cleaned_guardian_text = preprocessed_text(cleaned_guardian_content) #clean both datasets using preprocessed_text defined function
cleaned_wiki_text = preprocessed_text(wiki_category_content)
combined_content = cleaned_guardian_text + " " + cleaned_wiki_text #combine the two datasets

def tokenise_and_clean(text):
    stop_words = set(stopwords.words('english')) #initialises a set of stopwords from NLTK
    sentences = sent_tokenize(text) #splits text into individual sentences to process on a sentence-by-sentence basis
    cleaned_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence) #breaks each sentence into individual words.
        cleaned_tokens = [token for token in tokens if token.isalpha() and token not in stop_words] #filters out non-alphabetic characters (again, just in case)
        cleaned_sentences.append(cleaned_tokens) #appends cleaned list of word tokens to cleaned_sentences list
    return cleaned_sentences #returns list of sentence lists with cleaned word tokens
 
final_dataset = tokenise_and_clean(combined_content) #tokenise combined dataset using defined function

model = gensim.models.Word2Vec(vector_size=100, min_count=18, window=15, sg=0)
model.build_vocab(final_dataset, update=False) #Build vocabulary from the dataset
model.train(final_dataset, total_examples=model.corpus_count, epochs=10) #train the model
model.save("Assignment2_FINAL.model") #save trained model