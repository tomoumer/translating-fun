import streamlit as st
# import pandas as pd 
import requests
from bs4 import BeautifulSoup
# import time
from transformers import pipeline

def parse_wotd_url(
    url: str
) -> tuple[str, str, list[str], str]:
    """Function that takes in the url where the word of the day is located
        And parses out the word, definition, examples and the stand-alone
        word enty link.

        Parameters
        ----------
        url : str
            The url pointing to the word of the day location.

        Returns
        wotd : str
            Word of the day (on the given url).
        definition : str
            Definition of the wotd.
        examples : list
            Example or examples of the wotd.
        entry_link : str
            Url for the wotd entry in the dictionary.
    """

    result = requests.get(url)

    if result.status_code != 200:
        raise ValueError(f'Website did not respond as expected. Status code: {result.status_code}')

    soup = BeautifulSoup(result.text)

    # can see previous words of the day using h2 tags
    wotd = soup.find_all('h2')[0].text

    meaning_tag = soup.find_all('h2')[1]

    # get the definition and examples until we reach the "see the entry"
    sibling = meaning_tag
    definition = ''
    examples = []
    entry_link = ''
    while True:
        sibling = sibling.find_next_sibling('p')

        if definition == '':
            definition = sibling.text
        elif "See the entry" in sibling.text:
            entry_link = sibling.find(href=True).attrs.get('href', 'Not found')
            break
        else:
            examples.append(sibling.text.replace('//', '').strip())

    return wotd, definition, examples, entry_link

eng_to_ita = pipeline(
    "translation", 
    model="facebook/m2m100_418M", 
    tokenizer="facebook/m2m100_418M",
    src_lang="en", 
    tgt_lang="it"  # Italian code
)

eng_to_slo = pipeline(
    "translation", 
    model="facebook/m2m100_418M", 
    tokenizer="facebook/m2m100_418M",
    src_lang="en", 
    tgt_lang="sl"  # Slovenian code
)


st.title('Translating From English to Italian and Slovenian')

## select what word / words

wotd, definition, examples, entry_link = parse_wotd_url('https://www.merriam-webster.com/word-of-the-day/')

## =========== actual display
st.write('word:', wotd)
st.write('definition:', definition)
for i, example in enumerate(examples):
    st.write(f'example {i}: {example}')

st.divider()
st.write('slo')
st.write('beseda dneva:', eng_to_slo(wotd)[0]['translation_text'])
st.write('definicija:', eng_to_slo(definition)[0]['translation_text'])
for i, example in enumerate(examples):
    st.write(f'primer {i}: {eng_to_slo(example)[0]["translation_text"]}')

st.divider()
st.write('ita')
st.write('parola del giorno:', eng_to_ita(wotd)[0]['translation_text'])
st.write('definizione:', eng_to_ita(definition)[0]['translation_text'])
for i, example in enumerate(examples):
    st.write(f'esempio {i}: {eng_to_ita(example)[0]["translation_text"]}')




