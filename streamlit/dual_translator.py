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

    soup = BeautifulSoup(result.text, features="html.parser")

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
st.divider()
col1, col2, col3 = st.columns(3)
col1.write(f'🇺🇸 {wotd}')
col1.write('definition:')
col1.write(definition)
col1.write('Examples:')
for example in examples:
    col1.write(example)

col2.write(f"🇸🇮 {eng_to_slo(wotd)[0]['translation_text']}")
col2.write('Definicija:')
col2.write(eng_to_slo(definition)[0]['translation_text'])
col2.write('Primeri:')
for example in examples:
    col2.write(eng_to_slo(example)[0]["translation_text"])

col3.write(f"🇮🇹 {eng_to_ita(wotd)[0]['translation_text']}")
col3.write('Definizione:')
col3.write(eng_to_ita(definition)[0]['translation_text'])
col3.write('Esempi:')
for example in examples:
    col3.write(eng_to_ita(example)[0]["translation_text"])

st.divider()



