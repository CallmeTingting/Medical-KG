import math
import itertools
import string
import re
import requests

import nltk
import spacy

nltk.download('all')
spacy.cli.download("en_core_web_sm")

import numpy as np
import pandas as pd
import openpyxl

import matplotlib.pyplot as plt
import seaborn as sn
import networkx as nx
import graphviz
from graphviz import Graph

from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# extract treatment, side effect and interaction for particular drug in bnf
def bnf_search_keyword(drug):
    url = 'https://bnf.nice.org.uk/drug/' + drug + '.html'
    response = requests.get(url)
    indication_structure = re.compile('<span class="indication"><span class="indication">\s*(.*?)\s*</span></span>')
    indication = indication_structure.findall(response.text)
    indications = [((re.sub(r'\([^)]*\)', '', i)).rstrip()).lower() for i in indication]

    side_effect_structure = re.compile('<span class="sideEffect">\s*(.*?)\s*</span>')
    side_effect = side_effect_structure.findall(response.text)
    side_effects = [s.partition('(')[0].lower() for s in side_effect]

    interaction_url_structure = re.compile('<li><a href="\s*(.*?)\s*" class="interactant">')
    interaction_url = interaction_url_structure.findall(response.text)
    interaction_url = 'https://bnf.nice.org.uk' + interaction_url[0]
    interaction_response = requests.get(interaction_url)
    interaction_structure = re.compile('class="scroll-to-href"><span>\s*(.*?)\s*</span></a>')
    interaction = interaction_structure.findall(interaction_response.text)
    interactions = [i.lower() for i in interaction]

    return [indications, side_effects, interactions]


# extract titles and pmids from pubmed with particular drug
def pubmed_search_keyword(drug):
    titles = []
    pmids = []

    url = 'https://pubmed.ncbi.nlm.nih.gov/?term=' + drug + '&filter=simsearch1.fha&filter=pubt.randomizedcontrolledtrial&filter=datesearch.y_10&filter=lang.english'
    response = requests.get(url)
    result_num_structure = re.compile('<meta name="log_resultcount" content="(.*?)" />')
    result_num = int(result_num_structure.findall(response.text)[0])

    for j in range(math.ceil(int(result_num) / 10)):
        page_url = 'https://pubmed.ncbi.nlm.nih.gov/?term=' + drug + '&filter=simsearch1.fha&filter=pubt.randomizedcontrolledtrial&filter=datesearch.y_10&filter=lang.english' + '&page=' + str(
            j + 1)
        response = requests.get(page_url)
        article_title_structure = re.compile('data-article-id="\d{8}">\s*(.*?)\s*</a>')
        article_title = article_title_structure.findall(response.text)
        titles = titles + article_title

        article_pmid_structure = re.compile('<meta name="log_displayeduids" content="(.*?)" />')
        article_pmid = article_pmid_structure.findall(response.text)
        article_pmids = article_pmid[0].split(',')
        pmids = pmids + article_pmids

    return titles, pmids


# extract sentences and mesh terms with pmids in pubmed
def abstract_and_mesh(titles, pmids):
    MeSH_terms = []
    sentences = []
    exsiting_pmids = []

    for i in range(len(titles)):
        title = titles[i].lower()
        pmid = pmids[i]

        if (sum(['laser' in title, 'surgery' in title, 'surgical' in title, 'implant' in title]) == 0) and (
                pmid not in exsiting_pmids):
            exsiting_pmids.append(pmid)
            url = 'https://pubmed.ncbi.nlm.nih.gov/' + str(pmid) + '/'
            response = requests.get(url)
            MeSH_structure = re.compile('data-pinger-ignore>\s*(.*?)\s*</button><div id')
            MeSHs = MeSH_structure.findall(response.text)
            MeSH_term = []
            for MeSH in list(set(MeSHs)):
                append_term = re.search(r'(.*)/(.*)', MeSH).group(1)[:-1] if '/' in MeSH else MeSH
                MeSH_term.append(append_term.strip('*'))
            MeSH_terms = MeSH_terms + MeSH_term

            abstract_title_strcuture = re.compile('<strong class="sub-title">\s*(.*?)\s*</strong>')
            abstract_title = abstract_title_strcuture.findall(response.text)

            abstract_content_strcuture = re.compile('</strong>\s*(.*?)\s*</p>')
            abstract_content = abstract_content_strcuture.findall(response.text)
            abstract_contents = [a for a in abstract_content if (len(a) > 0) and ('figure-caption-contents' not in a)]
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            Target = ['result:', 'results:', 'conclusion:', 'conclusions:', 'findings:', 'interpretation:',
                      'discussion:']

            for j in range(len(abstract_title)):
                if abstract_title[j].lower() in Target:
                    sentence = tokenizer.tokenize(abstract_contents[j])
                    for term in sentence:
                        sentences.append(term)
    return sentences, list(set(MeSH_terms))


# extract mesh hierarchy for each mesh term 2
def mesh_feature(response):
    MeSH_hierarchy_strcuture = re.compile('</a><ul>\s*(.*?)\s*</ul>')
    MeSH_hierarchy = MeSH_hierarchy_strcuture.findall(response.text)
    structure = re.compile('>\s*(.*?)\s*<')
    if len(MeSH_hierarchy) > 0:
        return list(filter(None, structure.findall(MeSH_hierarchy[0])))


# extract mesh hierarchy for each mesh term 1
def drug_diseases_and_symptoms(MeSH_terms):
    df_MeSH = pd.DataFrame({'MeSH': MeSH_terms})
    columns = ['category1', 'category2', 'other features']
    df_MeSH[columns] = [None, None, None]

    for i in range(len(MeSH_terms)):
        url = 'https://www.ncbi.nlm.nih.gov/mesh/?term=' + MeSH_terms[i]
        response = requests.get(url)
        pagefound_strcuture = re.compile('<title>\s*(.*?)\s*</title>')
        pagefound = pagefound_strcuture.findall(response.text)

        if pagefound != ['No items found - MeSH - NCBI']:
            resultpage_strcuture = re.compile(
                '<h3 class="result_count left">\s*(.*?)\s*</h3><span id="result_sel" class="nowrap">')
            resultpage = resultpage_strcuture.findall(response.text)

            if resultpage == []:
                MeSH_features = mesh_feature(response)

            else:
                next_url_structure = re.compile('class="ui-helper-hidden-accessible">\s*(.*?)\s*</label>')
                next_url_id = next_url_structure.findall(response.text)[0]
                url_id = ''.join(list(filter(lambda num: num in '0123456789', next_url_id)))
                next_url = 'https://www.ncbi.nlm.nih.gov/mesh/' + str(url_id)
                next_response = requests.get(next_url)
                MeSH_features = mesh_feature(next_response)
            df_MeSH['category1'][i] = MeSH_features

    df_MeSH.dropna(subset=['category1'], inplace=True)
    df_MeSH.reset_index(drop=True, inplace=True)

    for j in range(df_MeSH.shape[0]):
        if len(df_MeSH['category1'][j]) >= 3:
            df_MeSH['other features'][j] = df_MeSH['category1'][j][2:]
            df_MeSH['category2'][j] = df_MeSH['category1'][j][1]
            df_MeSH['category1'][j] = df_MeSH['category1'][j][0]
        elif len(df_MeSH['category1'][j]) == 2:
            df_MeSH['category2'][j] = df_MeSH['category1'][j][1]
            df_MeSH['category1'][j] = df_MeSH['category1'][j][0]

    df_MeSH_drug = (df_MeSH[(df_MeSH['category1'] == 'Chemicals and Drugs Category')])['MeSH']
    df_MeSH_disease_and_symptom = (df_MeSH[(df_MeSH['category1'] == 'Diseases Category')])['MeSH']
    drugs = [d.lower() for d in list(df_MeSH_drug.unique())]
    attention_term = ['capsules', 'tablets']
    drugs = [d for d in drugs if d not in attention_term]
    diseases_and_symptoms = [d.lower() for d in list(df_MeSH_disease_and_symptom.unique())]

    return df_MeSH, drugs, diseases_and_symptoms


# filter out instances
def example_generation(sentences, drugs, diseases_and_symptoms, indications, side_effects, interactions, drug):
    drugs = drugs + [drug]
    sentence_list1 = []
    indications_list = []

    sentence_list2 = []
    side_effects_list = []

    sentence_list3 = []
    interact_drugs_list = []

    sentence_list4 = []
    no_interact_list = []

    translator = str.maketrans('', '', string.punctuation)

    for i in range(len(sentences)):
        one_gram = [t.lower() for t in nltk.word_tokenize(re.sub(r'\([^)]*\)', '', sentences[i]).translate(translator))]
        number_list = [x for x in one_gram if x.isdigit()]
        number_counts = len(number_list)
        two_grams = [' '.join(b) for b in ngrams(one_gram, 2)]
        three_grams = [' '.join(b) for b in ngrams(one_gram, 3)]
        tokens = one_gram + two_grams + three_grams

        target_drug_in_sentence = 0
        indications_in_sentence = []
        side_effects_in_sentence = []
        other_diseases_in_sentence = []
        interact_drugs_in_sentence = []
        no_interact_in_sentence = []

        for token in tokens:
            if token == drug:
                target_drug_in_sentence = 1
            if token in indications:
                indications_in_sentence.append(token)
            if token in side_effects:
                side_effects_in_sentence.append(token)
            if token in diseases_and_symptoms and (token not in indications) and (token not in side_effects):
                other_diseases_in_sentence.append(token)
            if (token in drugs) and (token in interactions):
                interact_drugs_in_sentence.append(token)
            if (token in drugs) and (token not in interactions + [drug]):
                no_interact_in_sentence.append(token)

        indications_in_sentence = list(set(indications_in_sentence))
        side_effects_in_sentence = list(set(side_effects_in_sentence))
        other_diseases_in_sentence = list(set(other_diseases_in_sentence))
        interact_drugs_in_sentence = list(set(interact_drugs_in_sentence))
        no_interact_in_sentence = list(set(no_interact_in_sentence))

        if (len(indications_in_sentence) >= 1) and (target_drug_in_sentence == 1) and (number_counts < 2):
            for j in range(len(indications_in_sentence)):
                indication = indications_in_sentence[j]
                indications_list.append(indication)
                sentence = re.sub(r'\([^)]*\)', '', sentences[i]).lower()
                sentence = sentence.replace(drug, 'drug')
                sentence = sentence.replace(indication, 'disease_and_symptom')
                unrelated_drugs = interact_drugs_in_sentence + no_interact_in_sentence
                unrelated_diseases = indications_in_sentence + side_effects_in_sentence + other_diseases_in_sentence
                unrelated_diseases = [u for u in unrelated_diseases if u != indication]

                if unrelated_drugs is not None:
                    for u in range(len(unrelated_drugs)):
                        sentence = sentence.replace(unrelated_drugs[u], 'other_drug')

                if unrelated_diseases is not None:
                    for v in range(len(unrelated_diseases)):
                        sentence = sentence.replace(unrelated_diseases[v], 'other_disease_and_symptom')

                sentence_list1.append(sentence)

        if (len(side_effects_in_sentence) >= 1) and (target_drug_in_sentence == 1) and (number_counts < 2):
            for j in range(len(side_effects_in_sentence)):
                side_effect = side_effects_in_sentence[j]
                side_effects_list.append(side_effect)
                sentence = re.sub(r'\([^)]*\)', '', sentences[i]).lower()
                sentence = sentence.replace(drug, 'drug')
                sentence = sentence.replace(side_effect, 'disease_and_symptom')
                unrelated_drugs = interact_drugs_in_sentence + no_interact_in_sentence
                unrelated_diseases = indications_in_sentence + side_effects_in_sentence + other_diseases_in_sentence
                unrelated_diseases = [u for u in unrelated_diseases if u != side_effect]

                if unrelated_drugs is not None:
                    for u in range(len(unrelated_drugs)):
                        sentence = sentence.replace(unrelated_drugs[u], 'other_drug')

                if unrelated_diseases is not None:
                    for v in range(len(unrelated_diseases)):
                        sentence = sentence.replace(unrelated_diseases[v], 'other_disease_and_symptom')

                sentence_list2.append(sentence)

        if (len(interact_drugs_in_sentence) >= 1) and (target_drug_in_sentence == 1) and (number_counts < 2):
            for j in range(len(interact_drugs_in_sentence)):
                interact_drug = interact_drugs_in_sentence[j]
                interact_drugs_list.append(interact_drug)
                sentence = re.sub(r'\([^)]*\)', '', sentences[i]).lower()
                sentence = sentence.replace(drug, 'drug_one')
                sentence = sentence.replace(interact_drug, 'drug_two')
                unrelated_drugs = interact_drugs_in_sentence + no_interact_in_sentence
                unrelated_drugs = [u for u in unrelated_drugs if u != interact_drug]
                unrelated_diseases = indications_in_sentence + side_effects_in_sentence + other_diseases_in_sentence

                if unrelated_drugs is not None:
                    for u in range(len(unrelated_drugs)):
                        sentence = sentence.replace(unrelated_drugs[u], 'other_drug')

                if unrelated_diseases is not None:
                    for v in range(len(unrelated_diseases)):
                        sentence = sentence.replace(unrelated_diseases[v], 'other_disease_and_symptom')

                sentence_list3.append(sentence)

        if (len(no_interact_in_sentence) >= 1) and (target_drug_in_sentence == 1) and (number_counts < 2):
            for j in range(len(no_interact_in_sentence)):
                no_interact = no_interact_in_sentence[j]
                no_interact_list.append(no_interact)
                sentence = re.sub(r'\([^)]*\)', '', sentences[i]).lower()
                sentence = sentence.replace(drug, 'drug_one')
                sentence = sentence.replace(no_interact, 'drug_two')
                unrelated_drugs = interact_drugs_in_sentence + no_interact_in_sentence
                unrelated_drugs = [u for u in unrelated_drugs if u != no_interact]
                unrelated_diseases = indications_in_sentence + side_effects_in_sentence + other_diseases_in_sentence

                if unrelated_drugs is not None:
                    for u in range(len(unrelated_drugs)):
                        sentence = sentence.replace(unrelated_drugs[u], 'other_drug')

                if unrelated_diseases is not None:
                    for v in range(len(unrelated_diseases)):
                        sentence = sentence.replace(unrelated_diseases[v], 'other_disease_and_symptom')

                sentence_list4.append(sentence)

    df_indication = pd.DataFrame(
        {'sentence': sentence_list1, 'drug': [drug] * len(sentence_list1), 'disease_and_symptom': indications_list,
         'classification': ['treatment'] * len(sentence_list1)})
    df_side_effects = pd.DataFrame(
        {'sentence': sentence_list2, 'drug': [drug] * len(sentence_list2), 'disease_and_symptom': side_effects_list,
         'classification': ['side_effect'] * len(sentence_list2)})
    df_interact_drugs = pd.DataFrame(
        {'sentence': sentence_list3, 'drug_one': [drug] * len(sentence_list3), 'drug_two': interact_drugs_list,
         'classification': ['interact'] * len(sentence_list3)})
    df_no_interact = pd.DataFrame(
        {'sentence': sentence_list4, 'drug_one': [drug] * len(sentence_list4), 'drug_two': no_interact_list,
         'classification': ['not_interact'] * len(sentence_list4)})

    return df_indication, df_side_effects, df_interact_drugs, df_no_interact


# use trigger words to filter out drug interaction instances
def interact_instance_filter(df_interacts, df_nointeracts):
    df_interacts = df_interacts.sample(frac=1).reset_index(drop=True)
    df_nointeracts = df_nointeracts.sample(frac=1).reset_index(drop=True)

    trigger_word = ['+', 'plus', 'taken', 'extended', 'with drugtwo and drugone', 'with drugone and drugtwo',
                    'adding', 'added', 'addition', 'combining', 'combined', 'combination', 'dual',
                    'of drugone and drugtwo', 'of drug_two and drug_one', 'coadministration', 'co-prescribed',
                    'concomitant', 'associated with', 'drugone with drugtwo group', 'drugtwo with drugone group']

    for i in range(df_interacts.shape[0]):
        sentence = df_interacts['sentence'][i]
        trigger_number = 0
        translator = str.maketrans('', '', string.punctuation)
        sentence = sentence.translate(translator)
        one_gram = [s.lower() for s in nltk.word_tokenize(sentence)]
        four_grams = ngrams(one_gram, 4)
        four_grams = [' '.join(f) for f in four_grams]
        token = one_gram + four_grams
        for j in range(len(trigger_word)):
            trigger_number = trigger_number + 1 if trigger_word[j] in token else trigger_number
        if trigger_number == 0:
            df_interacts['classification'][i] = 'not_interact'

    df_nointeract_filter = df_interacts[df_interacts['classification'] == 'not_interact']
    df_interact_filter = df_interacts[df_interacts['classification'] == 'interact']
    df_nointeract_overall = pd.concat([df_nointeracts, df_nointeract_filter])

    df_nointeract_filter = df_nointeract_filter.sample(frac=1).reset_index(drop=True)
    df_nointeract_overall = df_nointeract_overall.sample(frac=1).reset_index(drop=True)

    return df_interact_filter, df_nointeract_overall


# obtain tokens in the shortest path in dependency graph
def Shorest_path(sentence, nlp, target_one, target_two):
    doc = nlp(sentence)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_), '{0}'.format(child.lower_)))
    graph = nx.Graph(edges)
    try:
        feature = nx.shortest_path(graph, source=target_one, target=target_two)
    except:
        feature = []

    return feature[1:-1]


# obtain features in feature engineering
def feature_selection(df):
    translator = str.maketrans('', '', string.punctuation)
    lemmatizer = WordNetLemmatizer()
    stopword = stopwords.words('english')
    nlp = spacy.load("en_core_web_sm")

    neg_noun_tag = ['CC', 'DT', 'PDT', 'MD', 'NN', 'NNS']
    verb_tag = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    adj_tag = ['JJ', 'RB']

    df = df.sample(frac=1).reset_index(drop=True)
    df_train = df[0: math.ceil(df.shape[0] * 0.8)]
    df_test = df[math.ceil(df.shape[0] * 0.8) + 1:]
    df_test = df_test.reset_index(drop=True)

    column_one = df.columns[1].translate(translator)
    column_two = df.columns[2].translate(translator)

    token_feature_list = []
    token_tags_list = []
    shortest_path_list = []
    path_tags_list = []

    for i in range(df_train.shape[0]):
        sentence = df_train['sentence'][i].translate(translator)
        shortest_path = Shorest_path(sentence, nlp, column_one, column_two)
        path_tags = nltk.pos_tag(shortest_path)
        removed_term = [' ']
        path_candidates = []
        for u, v in path_tags:
            if v in neg_noun_tag and (u not in removed_term):
                path_candidates.append(u + '_DE')
            if v in verb_tag and (u not in removed_term):
                path_candidates.append(lemmatizer.lemmatize(u, 'v') + '_DE')
            if v in adj_tag and (u not in removed_term):
                path_candidates.append(lemmatizer.lemmatize(u, 'a') + '_DE')

        path_tags_candidates = []
        for u, v in path_tags:
            path_tags_candidates.append(v)
        path_tags_candidates = [p + '_DEGP' for p in path_tags_candidates]

        one_gram = [s.lower() for s in nltk.word_tokenize(sentence)]
        position_1 = one_gram.index(column_one)
        position_2 = one_gram.index(column_two)
        position = [position_1, position_2] if position_2 > position_1 else [position_2, position_1]
        position_list = ['BF'] * position[0] + ['position_1'] + ['BE'] * (position[1] - position[0] - 1) + [
            'position_2'] + ['AF'] * (len(one_gram) - position[1] - 1)
        token_tags = nltk.pos_tag(one_gram)
        token_candidates = []
        removed_term = stopword + [column_one, column_two, ' ']

        for u, v in token_tags:
            if (v in neg_noun_tag) and (u not in removed_term):
                token_candidates.append(u + '_' + position_list[one_gram.index(u)])
            if (v in verb_tag) and (u not in removed_term):
                token_candidates.append(lemmatizer.lemmatize(u, 'v') + '_' + position_list[one_gram.index(u)])
            if (v in adj_tag) and (u not in removed_term):
                token_candidates.append(lemmatizer.lemmatize(u, 'a') + '_' + position_list[one_gram.index(u)])

        token_tags_candidates = []
        for u, v in token_tags:
            token_tags_candidates.append(v)

        token_tags_BE = token_tags_AF = []

        if (position[0] - 3) >= 0:
            token_tags_BE = token_tags_candidates[position[0] - 3:position[0]]
            three_grams = ngrams(token_tags_BE, 3)
            three_grams = [' '.join(f) for f in three_grams]
            token_tags_BE += three_grams
        elif (position[0] - 3) < 0:
            token_tags_BE = ['None'] * int(3 - position[0])
            token_tags_BE += token_tags_candidates[0:position[0]]
        token_tags_BE = [t + '_BEGP' for t in token_tags_BE]

        if (len(one_gram) - position[1] - 4) >= 0:
            token_tags_AF = token_tags_candidates[position[1] + 1: position[1] + 4]
            three_grams = ngrams(token_tags_AF, 3)
            three_grams = [' '.join(f) for f in three_grams]
            token_tags_AF += three_grams
        elif (len(one_gram) - position[1] - 4) < 0:
            token_tags_AF = token_tags_candidates[position[1]:]
            token_tags_AF += ['None'] * int(4 + position[1] - len(one_gram))
        token_tags_AF = [t + '_AFGP' for t in token_tags_AF]

        token_feature_list += list(set(token_candidates))
        token_tags_list = token_tags_list + list(set(token_tags_BE)) + list(set(token_tags_AF))
        shortest_path_list += list(set(path_candidates))
        path_tags_list += list(set(path_tags_candidates))

    token_features = [f[0] for f in nltk.FreqDist(token_feature_list).most_common(200)]
    token_tag_features = [f[0] for f in nltk.FreqDist(token_tags_list).most_common(80)]
    path_features = [f[0] for f in nltk.FreqDist(shortest_path_list).most_common(100)]
    path_tag_features = [f[0] for f in nltk.FreqDist(path_tags_list).most_common(50)]

    return df_train, df_test, token_features, token_tag_features, path_features, path_tag_features


# extract features from sentences
def feature_extraction(df, token_features, token_tag_features, path_features, path_tag_features):
    translator = str.maketrans('', '', string.punctuation)
    lemmatizer = WordNetLemmatizer()
    stopword = stopwords.words('english')
    nlp = spacy.load("en_core_web_sm")

    neg_noun_tag = ['CC', 'DT', 'PDT', 'MD', 'NN', 'NNS']
    verb_tag = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    adj_tag = ['JJ', 'RB']

    column_one = df.columns[1].translate(translator)
    column_two = df.columns[2].translate(translator)

    df = pd.concat(
        [df, pd.DataFrame(columns=['token_feature', 'path_feature', 'token_tag_feature', 'path_tag_feature'])],
        sort=False)
    df = pd.concat([df, pd.DataFrame(columns=token_features + path_features + token_tag_features + path_tag_features)],
                   sort=False)

    for i in range(df.shape[0]):
        sentence = df['sentence'][i].translate(translator)
        shortest_path = Shorest_path(sentence, nlp, column_one, column_two)
        path_tags = nltk.pos_tag(shortest_path)
        removed_term = [' ']
        path_candidates = []
        for u, v in path_tags:
            if v in neg_noun_tag and (u not in removed_term):
                path_candidates.append(u + '_DE')
            if v in verb_tag and (u not in removed_term):
                path_candidates.append(lemmatizer.lemmatize(u, 'v') + '_DE')
            if v in adj_tag and (u not in removed_term):
                path_candidates.append(lemmatizer.lemmatize(u, 'a') + '_DE')

        path_tags_candidates = []
        for u, v in path_tags:
            path_tags_candidates.append(v)
        path_tags_candidates = [p + '_DEGP' for p in path_tags_candidates]

        one_gram = [s.lower() for s in nltk.word_tokenize(sentence)]
        position_1 = one_gram.index(column_one)
        position_2 = one_gram.index(column_two)
        position = [position_1, position_2] if position_2 > position_1 else [position_2, position_1]
        position_list = ['BF'] * position[0] + ['position_1'] + ['BE'] * (position[1] - position[0] - 1) + [
            'position_2'] + ['AF'] * (len(one_gram) - position[1] - 1)
        token_tags = nltk.pos_tag(one_gram)
        token_candidates = []
        removed_term = stopword + [column_one, column_two, ' ']

        for u, v in token_tags:
            if (v in neg_noun_tag) and (u not in removed_term):
                token_candidates.append(u + '_' + position_list[one_gram.index(u)])
            if (v in verb_tag) and (u not in removed_term):
                token_candidates.append(lemmatizer.lemmatize(u, 'v') + '_' + position_list[one_gram.index(u)])
            if (v in adj_tag) and (u not in removed_term):
                token_candidates.append(lemmatizer.lemmatize(u, 'a') + '_' + position_list[one_gram.index(u)])

        token_tags_candidates = []
        for u, v in token_tags:
            token_tags_candidates.append(v)

        token_tags_BE = token_tags_AF = []

        if (position[0] - 3) >= 0:
            token_tags_BE = token_tags_candidates[position[0] - 3:position[0]]
            three_grams = ngrams(token_tags_BE, 3)
            three_grams = [' '.join(f) for f in three_grams]
            token_tags_BE += three_grams
        elif (position[0] - 3) < 0:
            token_tags_BE = ['None'] * int(3 - position[0])
            token_tags_BE += token_tags_candidates[0:position[0]]
        token_tags_BE = [t + '_BEGP' for t in token_tags_BE]

        if (len(one_gram) - position[1] - 4) >= 0:
            token_tags_AF = token_tags_candidates[position[1] + 1: position[1] + 4]
            three_grams = ngrams(token_tags_AF, 3)
            three_grams = [' '.join(f) for f in three_grams]
            token_tags_AF += three_grams
        elif (len(one_gram) - position[1] - 4) < 0:
            token_tags_AF = token_tags_candidates[position[1]:]
            token_tags_AF += ['None'] * int(4 + position[1] - len(one_gram))
        token_tags_AF = [t + '_AFGP' for t in token_tags_AF]

        df['token_feature'][i] = list(set(token_candidates))
        df['token_tag_feature'][i] = list(set(token_tags_BE)) + list(set(token_tags_AF))
        df['path_feature'][i] = list(set(path_candidates))
        df['path_tag_feature'][i] = list(set(path_tags_candidates))

        for j in range(len(token_features)):
            df[token_features[j]][i] = 1 if token_features[j] in df['token_feature'][i] else 0

        for k in range(len(path_features)):
            df[path_features[k]][i] = 1 if (path_features[k] in df['path_feature'][i]) else 0

        for m in range(len(token_tag_features)):
            df[token_tag_features[m]][i] = 1 if (token_tag_features[m] in df['token_tag_feature'][i]) else 0

        for n in range(len(path_tag_features)):
            df[path_tag_features[n]][i] = 1 if (path_tag_features[n] in df['path_tag_feature'][i]) else 0

    return df


# extract instances from predict contents
def name_entity_recognition(sentences, drugs, diseases_and_symptoms):
    sentence_list1 = []
    drugs_list1 = []
    drugs_list2 = []

    sentence_list2 = []
    drugs_list3 = []
    diseases_and_symptoms_list = []

    translator = str.maketrans('', '', string.punctuation)

    for i in range(len(sentences)):
        one_gram = [s.lower() for s in nltk.word_tokenize(re.sub(r'\([^)]*\)', '', sentences[i]).translate(translator))]
        number_list = [x for x in one_gram if x.isdigit()]
        number_counts = len(number_list)
        two_grams = [' '.join(b) for b in ngrams(one_gram, 2)]
        three_grams = [' '.join(b) for b in ngrams(one_gram, 3)]
        tokens = one_gram + two_grams + three_grams

        drugs_in_sentence = []
        diseases_and_symptoms_in_sentence = []

        for token in tokens:
            if token in drugs:
                drugs_in_sentence.append(token)
            if token in diseases_and_symptoms:
                diseases_and_symptoms_in_sentence.append(token)

        if len(list(set(drugs_in_sentence))) >= 2 and (number_counts < 2):
            drug_candidates = list(set(drugs_in_sentence))
            diseases_and_symptoms_candidates = list(set(diseases_and_symptoms_in_sentence))
            combinations = list(itertools.combinations(drug_candidates, 2))
            for j in range(len(combinations)):
                drugs_list1.append(combinations[j][0])
                drugs_list2.append(combinations[j][1])
                sentence = re.sub(r'\([^)]*\)', '', sentences[i]).lower()
                sentence = sentence.replace(combinations[j][0], 'drug_one')
                sentence = sentence.replace(combinations[j][1], 'drug_two')
                unrelated_drugs = [u for u in drug_candidates if u != combinations[j][0] and u != combinations[j][1]]
                unrelated_diseases = diseases_and_symptoms_candidates

                if unrelated_drugs is not None:
                    for u in range(len(unrelated_drugs)):
                        sentence = sentence.replace(unrelated_drugs[u], 'other_drug')

                if unrelated_diseases is not None:
                    for v in range(len(unrelated_diseases)):
                        sentence = sentence.replace(unrelated_diseases[v], 'other_disease_and_symptom')

                sentence_list1.append(sentence)

        if (len(list(set(diseases_and_symptoms_in_sentence))) >= 1) and (len(list(set(drugs_in_sentence))) >= 1) and (
                number_counts < 2):
            drug_candidates = list(set(drugs_in_sentence))
            diseases_and_symptoms_candidates = list(set(diseases_and_symptoms_in_sentence))
            combinations = list(itertools.product(drug_candidates, diseases_and_symptoms_candidates))
            for j in range(len(combinations)):
                drugs_list3.append(combinations[j][0])
                diseases_and_symptoms_list.append(combinations[j][1])
                sentence = re.sub(r'\([^)]*\)', '', sentences[i]).lower()
                sentence = sentence.replace(combinations[j][0], 'drug')
                sentence = sentence.replace(combinations[j][1], 'disease_and_symptom')
                unrelated_drugs = [u for u in drug_candidates if u != combinations[j][0]]
                unrelated_diseases = [u for u in diseases_and_symptoms_candidates if u != combinations[j][1]]

                if unrelated_drugs is not None:
                    for u in range(len(unrelated_drugs)):
                        sentence = sentence.replace(unrelated_drugs[u], 'other_drug')

                if unrelated_diseases is not None:
                    for v in range(len(unrelated_diseases)):
                        sentence = sentence.replace(unrelated_diseases[v], 'other_disease_and_symptom')

                sentence_list2.append(sentence)

    df_drug_drug = pd.DataFrame({'sentence': sentence_list1, 'drug_one': drugs_list1, 'drug_two': drugs_list2})
    df_drug_diseases_and_symptoms = pd.DataFrame(
        {'sentence': sentence_list2, 'drug': drugs_list3, 'disease_and_symptom': diseases_and_symptoms_list})

    return df_drug_drug, df_drug_diseases_and_symptoms


# train the classifier and output predict outcome
def tree_classifier(df_train, df_test, df_predict, token_features, token_tag_features, path_features,
                    path_tag_features):
    lb = LabelBinarizer()
    lb.fit((df_train['classification'].unique()).tolist())

    Xtrain = df_train[token_features + path_features + token_tag_features].to_numpy()
    Ytrain = df_train['classification'].to_numpy()
    Ytrain = (lb.transform(Ytrain)).reshape(-1)

    Xtest = df_test[token_features + path_features + token_tag_features].to_numpy()
    Ytest = df_test['classification'].to_numpy()
    Ytest = (lb.transform(Ytest)).reshape(-1)

    clf = RandomForestClassifier(min_samples_split=5, bootstrap=True)
    clf.fit(Xtrain, Ytrain)
    accuracy = clf.score(Xtest, Ytest)
    test_result = (clf.predict(Xtest)).reshape(-1)
    F_score = f1_score(Ytest, test_result)
    cf_matrix = confusion_matrix(Ytest, test_result)
    df_cf = pd.DataFrame(cf_matrix, index=[i for i in (df_train['classification'].unique()).tolist()],
                         columns=[i for i in (df_train['classification'].unique()).tolist()])
    sn.heatmap(df_cf, annot=True, cmap='Blues')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

    # feature_name = token_features + path_features + token_tag_features + path_tag_features
    # importances = clf.feature_importances_.reshape(-1)
    # sorted_indices = np.argsort(importances)[::-1]
    # feature_importances = importances[sorted_indices]
    # feature_names = np.array(feature_name)[sorted_indices]
    # plt.figure(figsize=(15, 8))
    # sn.barplot(feature_names[:30], feature_importances[:30], palette="Blues_r")
    # plt.xticks(rotation=90)
    # plt.xlabel('feature')
    # plt.ylabel('feature importance')
    # plt.show()

    df_predict = df_predict.sample(frac=1).reset_index(drop=True)
    Xpredict = df_predict[token_features + path_features + token_tag_features].to_numpy()
    Ypredict = clf.predict(Xpredict)
    Ypredict = lb.inverse_transform(Ypredict)
    df_result = df_predict[[df_predict.columns[1], df_predict.columns[2]]]
    df_result['classification'] = Ypredict

    return accuracy, F_score, cf_matrix, df_result


# main program start from here
df_bnf = pd.read_excel(r'C:\Users\win\Desktop\bnf_drugs.xlsx')
df_indications = df_side_effects = pd.DataFrame(columns=['sentence', 'drug', 'disease_and_symptom', 'classification'])
df_interact_drugs = df_no_interacts = pd.DataFrame(columns=['sentence', 'drug_one', 'drug_two', 'classification'])


# generate datasets for 2 classifications
for l in range(df_bnf.shape[0]):
    drug = df_bnf['drugs'][l]
    indications, side_effects, interactions = bnf_search_keyword(drug)
    drug = re.search(r'(.*)-(.*)', drug).group(1) if '-' in drug else drug
    indications = df_bnf['treatments for'][l].split(',')
    side_effects = df_bnf['side effects'][l].split(',')
    titles, pmids = pubmed_search_keyword(drug)
    sentences, MeSH_terms = abstract_and_mesh(titles, pmids)
    df_MeSH, drugs, diseases_and_symptoms = drug_diseases_and_symptoms(MeSH_terms)
    df_indication, df_side_effect, df_interact_drug, df_no_interact = example_generation(sentences, drugs,
                                                                                         diseases_and_symptoms,
                                                                                         indications, side_effects,
                                                                                         interactions, drug)

    df_indications = pd.concat([df_indications, df_indication])
    df_side_effects = pd.concat([df_side_effects, df_side_effect])
    df_interact_drugs = pd.concat([df_interact_drugs, df_interact_drug])
    df_no_interacts = pd.concat([df_no_interacts, df_no_interact])


# filter out drug interaction instances
df_interact_drugs, df_no_interacts = interact_instance_filter(df_interact_drugs, df_no_interacts)


# generate predict instances
predict_titles, predict_pmids = pubmed_search_keyword('asthma')
predict_sentences, predict_MeSH_terms = abstract_and_mesh(predict_titles, predict_pmids)
predict_df_MeSH, predict_drugs, predict_diseases_and_symptoms = drug_diseases_and_symptoms(predict_MeSH_terms)
df_interact_predict, df_treatment_predict = name_entity_recognition(predict_sentences, predict_drugs,
                                                                    predict_diseases_and_symptoms)


# sample balanced datasets
df_treatment = pd.concat([df_indications.sample(n=200), df_side_effects.sample(n=200)])
df_interact = pd.concat([df_interact_drugs.sample(n=200), df_no_interacts.sample(n=200)])


# treatment-side effect classification
df_treatment_train, df_treatment_test, treatment_token_features, treatment_token_tag_features, treatment_path_features, treatment_path_tag_features = feature_selection(
    df_treatment)
df_treatment_train = feature_extraction(df_treatment_train, treatment_token_features, treatment_token_tag_features,
                                        treatment_path_features, treatment_path_tag_features)
df_treatment_test = feature_extraction(df_treatment_test, treatment_token_features, treatment_token_tag_features,
                                       treatment_path_features, treatment_path_tag_features)
# drop instance that central entities have same token,ie:
df_treatment_predict = df_treatment_predict.drop(df_treatment_predict[(df_treatment_predict['drug'] == 'vitamin d') & (
        df_treatment_predict['disease_and_symptom'] == 'vitamin d deficiency')].index)
df_treatment_predict = df_treatment_predict.sample(frac=1).reset_index(drop=True)
df_treatment_predict = feature_extraction(df_treatment_predict, treatment_token_features, treatment_token_tag_features,
                                          treatment_path_features, treatment_path_tag_features)

treatment_accuracy, treatment_F, treatment_CF, df_treatment_result = tree_classifier(df_treatment_train,
                                                                                     df_treatment_test,
                                                                                     df_treatment_predict,
                                                                                     treatment_token_features,
                                                                                     treatment_token_tag_features,
                                                                                     treatment_path_features,
                                                                                     treatment_path_tag_features)


# drug interact-no interact classification
df_interact_train, df_interact_test, interact_token_features, interact_token_tag_features, interact_path_features, interact_path_tag_features = feature_selection(
    df_interact)
df_interact_train = feature_extraction(df_interact_train, interact_token_features, interact_token_tag_features,
                                       interact_path_features, interact_path_tag_features)
df_interact_test = feature_extraction(df_interact_test, interact_token_features, interact_token_tag_features,
                                      interact_path_features, interact_path_tag_features)
# drop instance that central entities have same token, ie:
df_interact_predict = df_interact_predict.drop(df_interact_predict[(df_interact_predict['drug_one'] == 'carbon') & (
        df_interact_predict['drug_two'] == 'carbon dioxide')].index)
df_interact_predict = df_interact_predict.sample(frac=1).reset_index(drop=True)
df_interact_predict = feature_extraction(df_interact_predict, interact_token_features, interact_token_tag_features,
                                         interact_path_features, interact_path_tag_features)
interact_accuracy, interact_F, interact_CF, df_interact_result = tree_classifier(df_interact_train, df_interact_test,
                                                                                 df_interact_predict,
                                                                                 interact_token_features,
                                                                                 interact_token_tag_features,
                                                                                 interact_path_features,
                                                                                 interact_path_tag_features)


# outcomes of predict sentences for treatment-side effect
treatment_one_hot = pd.get_dummies(df_treatment_result['classification'])
df_treatment_result = df_treatment_result.drop('classification', axis=1)
df_treatment_result = df_treatment_result.join(treatment_one_hot)
treatment_table = df_treatment_result.groupby(['drug', 'disease_and_symptom']).sum()
treatment_table = treatment_table.reset_index()
treatment_table_graph = treatment_table
treatment_table_graph = treatment_table_graph.drop(
    treatment_table_graph[(treatment_table_graph['treatment'] - treatment_table_graph['side_effect'] < 3)].index)
treatment_table_graph = treatment_table_graph.reset_index(drop=True)
treatment_table_graph = treatment_table_graph.drop(
    treatment_table_graph[(treatment_table_graph['side_effect'] - treatment_table_graph['treatment'] < 3)].index)
treatment_table_graph = treatment_table_graph.reset_index(drop=True)


# outcomes of predict sentences for drug interact-no interact
interact_one_hot = pd.get_dummies(df_interact_result['classification'])
df_interact_result = df_interact_result.drop('classification', axis=1)
df_interact_result = df_interact_result.join(interact_one_hot)
interact_table = df_interact_result.groupby(['drug_one', 'drug_two']).sum()
interact_table = interact_table.reset_index()
interact_table_graph = interact_table
interact_table_graph = interact_table_graph.drop(
    interact_table_graph[(interact_table_graph['interact'] > 3)].index)
interact_table_graph = interact_table_graph.reset_index(drop=True)


# medical knowledge graph visualization
dot_drugs = list(
    set((treatment_table_graph['drug'].unique()).tolist() + (interact_table_graph['drug_one'].unique()).tolist() + (
        interact_table_graph['drug_two'].unique()).tolist()))
dot_diseases_and_symptoms = (treatment_table_graph['disease_and_symptom'].unique()).tolist()


# visualize entities
dot = Graph('KG')
dot.attr('node', shape='ellipse')
for i in range(len(dot_drugs)):
    dot.node(dot_drugs[i], style='filled', fillcolor='peachpuff4')

dot.attr('node', shape='box')
for j in range(len(dot_diseases_and_symptoms)):
    dot.node(dot_diseases_and_symptoms[j], style='filled', fillcolor='peru')


# visualize relationships
for m in range(treatment_table_graph.shape[0]):
    if treatment_table_graph['treatment'][m] >= treatment_table_graph['side_effect'][m]:
        dot.edge(treatment_table_graph['drug'][m], treatment_table_graph['disease_and_symptom'][m])
    elif treatment_table_graph['treatment'][m] < treatment_table_graph['side_effect'][m]:
        dot.edge(treatment_table_graph['drug'][m], treatment_table_graph['disease_and_symptom'][m],
                 style='dashed')

for n in range(interact_table_graph.shape[0]):
    if interact_table_graph['interact'][n] - interact_table_graph['not_interact'][n] > 3:
        dot.edge(interact_table['drug_one'][n], interact_table['drug_two'][n], color='firebrick4')
    else:
        dot.edge(interact_table['drug_one'][n], interact_table['drug_two'][n], color='firebrick4')