#!/usr/bin/python
#-*- coding: utf-8 -*-
#
# Подготовка трейнсета для векторизации. На выходе - файл с токенизированными
# предложениями (токены разделены пробелом). Если работаем с полем FORM или
# LEMMA и в поле MISC встретился признак Entity, то заменяем токен на
# '<EntityX>', где X - значение признака. Токены с недопустимыми символами
# также замеяются на спец. токены вида <Y>, где Y - тип спец. токена.
#
# Синтаксис:
#     python parse_wiki.py
#
import re

from corpuscula.conllu import Conllu
from corpuscula.wikipedia_utils import download_wikipedia
from toxine.wikipedia_utils import TokenizedWikipedia

FIELD = 'FORM'                     # какое поле нам нужно
# NB: В TokenizedWikipedia().articles() есть только поле FORM, остальные поля
#     пустые. Однако для построения хороших векторов для словоформ без
#     лемматизации одной википедии мало

#corpus = 'wiki_tagged.conllu'      # исходный корпус. если None, берём
corpus = None                      #     TokenizedWikipedia().articles()
parsed_corpus = 'wiki_{}_parsed.txt'.format(FIELD)  # куда сохранять результат

download_wikipedia(overwrite=False)

def replace_tokens (token):
    token = token.lower()

    if token.isdecimal():
        # replace all numbers longer than 3001 with <NUM>
        if int(token) > 3001:
            token = '<NUM_UINT>'

    elif re.match(r'[+-]\d+', token):
        # if the number has a decimal part or (+-) in front, leave as-is (+36,6)
        token = '<NUM_SINT>'

    elif re.match(r'[+-]?\d+[.,]\d*$', token):
        # if the number has a decimal part or (+-) in front, leave as-is (+36,6)
        token = '<NUM_FLOAT>'

    elif re.match(r'[`’\'²³°$€%&~№()/"\«»„“+.,:;!?-]+$', token) \
      or re.match(r'[a-zA-Z]+-[а-яА-ЯёЁ]+$', token):
        # special symbols is left as-is
        # cases like 'HTLV-вирус', 'HTML-страница', 'ascii-код',
        #     'ftp-аутентификация' etc.
        pass

    elif re.match(r'\d+-[а-яА-ЯёЁ]+$', token):
        # 20-градусный
        token = re.sub(r'\d+', '<NUM_UINT>', token)

    elif re.search(r'[^a-zA-Zа-яА-ЯёЁ°²³.-]', token) \
      or (re.match(r'[a-zA-Z]', token)
      and re.match(r'[а-яА-ЯёЁ]', token)):
         # Everything else that has both latin and cyrillic (and any other character)
         #     is considered as UNK
         # Non-latin, non-cyrillic tokens are UNK, too
        return '<UNK>'

    return token

def read_corpus (corpus=None, silent=False):
    if isinstance(corpus, str):
        corpus = Conllu.load(corpus, **({'log_file': None} if silent else {}))
    elif callable(corpus):
        corpus = corpus()
    else:
        corpus = TokenizedWikipedia().articles(silent=silent)
    entity_prefix = 'Entity'
    for sent in corpus:
        sent = [
            '<{}>'.format(
                '+'.join(filter(lambda x: x.startswith(entity_prefix),
                                x['MISC'].keys()))
            )
                if FIELD in ['FORM', 'LEMMA']
               and list(filter(lambda x: x.startswith(entity_prefix),
                               x['MISC'].keys())) else
            replace_tokens(x[FIELD])
                for x in sent[0] if x[FIELD] and '-' not in x['ID']
        ]
        if sent:
            yield sent

with open(parsed_corpus, 'wt', encoding='utf-8') as f:
    [print(' '.join(x), file=f) for x in read_corpus(corpus)]
