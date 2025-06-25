import streamlit as st
import pandas as pd 
import requests
from bs4 import BeautifulSoup
import random
# import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# new changes
# from functools import lru_cache

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

def translate(
    text: str ,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM
) -> tuple[str, str]:
    inputs = tokenizer(text, return_tensors="pt")
    # SLO
    bos = tokenizer.convert_tokens_to_ids("slv_Latn")
    outs = model.generate(**inputs, forced_bos_token_id=bos, max_length=512)
    slo_translation = tokenizer.batch_decode(outs, skip_special_tokens=True)[0]
    # ITA
    bos = tokenizer.convert_tokens_to_ids("ita_Latn")
    outs = model.generate(**inputs, forced_bos_token_id=bos, max_length=512)
    ita_translation = tokenizer.batch_decode(outs, skip_special_tokens=True)[0]

    return slo_translation, ita_translation


model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# @lru_cache(maxsize=1)
# def get_translator(src_lang: str, tgt_lang: str):
#     return pipeline(
#         "translation", 
#         model="facebook/m2m100_418M", 
#         tokenizer="facebook/m2m100_418M",
#         src_lang=src_lang, 
#         tgt_lang=tgt_lang
#     )

# eng_to_slo = get_translator("en", "sl")
# eng_to_ita = get_translator("en", "it")




if 'stage' not in st.session_state:
    st.session_state.stage = 0

if 'english' not in st.session_state:
    st.session_state.english = []

if 'slovenian' not in st.session_state:
    st.session_state.slovenian = []

if 'italian' not in st.session_state:
    st.session_state.italian = []

if 'available_dates' not in st.session_state:
    end_date = pd.to_datetime('today').normalize()
    start_date = end_date - pd.to_timedelta('1000d')
    dates_1001 = pd.date_range(start_date, end_date, freq='D').to_list()
    st.session_state.available_dates = dates_1001


def get_wotd(wotd_date):
    if wotd_date == 'today':
        url = 'https://www.merriam-webster.com/word-of-the-day/'
    elif wotd_date == 'random':
        random_date = random.choice(st.session_state.available_dates).date()
        url = f'https://www.merriam-webster.com/word-of-the-day/{random_date}'

    wotd, definition, examples, entry_link = parse_wotd_url(url)
    # slovenian = [res['translation_text'] for res in eng_to_slo([wotd, definition] + examples)]
    # italian = [res['translation_text'] for res in eng_to_ita([wotd, definition] + examples)]

    translated = [translate(res, tokenizer, model) for res in [wotd, definition] + examples]
    slovenian = [x[0] for x in translated]
    italian = [x[1] for x in translated]
    
    st.session_state.english = [wotd, definition, examples, entry_link]
    st.session_state.slovenian = slovenian
    st.session_state.italian = italian
    st.session_state.stage = 1


st.title('Translating From English to Slovenian and Italian')

if st.session_state.stage == 0:
    st.write('This app will (attempt to) translate Words of the day from Meriam-Webster' \
    'online dictionary into Slovenian and Italian.')


## select what word / words
elif st.session_state.stage == 1:
    
    wotd, definition, examples, entry_link = st.session_state.english
    wotd_slo = st.session_state.slovenian[0]
    definition_slo = st.session_state.slovenian[1]
    examples_slo = st.session_state.slovenian[2:]
    wotd_ita = st.session_state.italian[0]
    definition_ita = st.session_state.italian[1]
    examples_ita = st.session_state.italian[2:]

    ## =========== actual display
    st.write('Merriam-Webster word entry:', entry_link)
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.write(f'🇺🇸 {wotd}')
    col1.write('definition:')
    col1.write(definition)
    col1.write('Examples:')
    for example in examples:
        col1.write(example)

    col2.write(f"🇸🇮 {wotd_slo}")
    col2.write('Definicija:')
    col2.write(definition_slo)
    col2.write('Primeri:')
    for example in examples_slo:
        col2.write(example)

    col3.write(f"🇮🇹 {wotd_ita}")
    col3.write('Definizione:')
    col3.write(definition_ita)
    col3.write('Esempi:')
    for example in examples_ita:
        col3.write(example)

    st.divider()

st.button("Get today's word!", on_click=get_wotd, args=('today',))
st.button('Get random WOTD!', on_click=get_wotd, args=('random',))



