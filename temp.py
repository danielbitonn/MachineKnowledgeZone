import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import trange, tqdm
import openai

from datetime import datetime, timedelta


def get_urls(date):
    url = 'https://techcrunch.com/' + date.strftime('%Y/%m/%d')
    content = requests.get(url).text
    return [a['href'] for a in BeautifulSoup(content, features="html.parser").find_all(
        'a',
        {'class': 'post-block__title__link'}
    )]


days_to_track = 7
urls = sum([get_urls(datetime.now() - timedelta(days=i)) for i in trange(days_to_track)], [])
print(f"Len of urls:\n{len(urls)}")


def get_article(url):
    content = requests.get(url).text
    article = BeautifulSoup(content, features="html.parser").find_all('div', {'class': 'article-content'})[0]
    return [p.text for p in article.find_all('p', recursive=False)]


articles = pd.DataFrame({
    'url': urls,
    'article': [get_article(url) for url in tqdm(urls)]
})
print(f"Articles:\n{articles}")

paragraphs = (
    articles.explode('article')
    .rename(columns={'article': 'paragraph'})
)
print(f"Paragraphs:\n{paragraphs}")

paragraphs = paragraphs[paragraphs['paragraph'].str.split().map(len) > 10]
print(f"Paragraphs:\n{paragraphs}")

with open('api_key', 'r') as f:
    openai.api_key = f.read().strip()
print(f"Key:\n{openai.api_key}")


def get_embedding(texts, model='text-embedding-ada-002'):
    texts = [text.replace('\n', ' ') for text in texts]
    return [res['embedding'] for res in openai.Embedding.create(input=texts, model=model)['data']]


batch_size = 100
embeddings = []

for i in trange(0, len(paragraphs), batch_size):
    embeddings += get_embedding(paragraphs.iloc[i:i + batch_size]['paragraph'])

paragraphs['embedding'] = embeddings
query = 'How many downlands does ChtGPT ap have?'  # keep in mind we scraped only a sample of articles from the last week
query_embedding = get_embedding([query])[0]
best_idx = paragraphs['embedding'].map(
    lambda emb: np.dot(emb, query_embedding) / (
            np.linalg.norm(emb) * np.linalg.norm(query_embedding)
    )
).argmax()

best_paragraph = paragraphs.iloc[best_idx]['paragraph']
prompt = (
        "Here's a piece of text:\n" +
        best_paragraph + '\n\n' +
        'I have a question about this text: ' + query +
        'Please answer in a concise manner'
)

print(prompt)
