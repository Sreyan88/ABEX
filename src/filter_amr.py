import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
from tqdm import tqdm
import random
import argparse
import re
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import sys
sys.path.append('/fs/nexus-projects/audio-visual_dereverberation/sonal/smatchpp/smatchpp/')
from main import main_func

def get_arguments():
    parser = argparse.ArgumentParser()

    # out path
    parser.add_argument('--input', type=str, default='', help='input path')
    parser.add_argument('--output', type=str, default='', help='output path')
    parser.add_argument('--output_mixner', type=str, default='', help='output path')
    parser.add_argument('--target', type=str, default='', help='target path')
    parser.add_argument('--method', type=str, default='append', help='append/replace')
    parser.add_argument('--labels', type=str, default='', help='file with input labels')


    args = parser.parse_args()
    return args

args = get_arguments()

with open(args.target, 'r') as f:
    text = f.readlines()

text = [i.strip() for i in text]

print('calculate similar sentences')
model = SentenceTransformer("all-distilroberta-v1")
embeddings = model.encode(text, show_progress_bar=True, batch_size=1024)

text_similar = []
for i in range(len(embeddings)):
    arr = []
    for j in range(len(embeddings)):
        arr.append(1 - cosine(embeddings[i], embeddings[j]))

    top_2 = sorted(range(len(arr)), key=lambda i: arr[i])[-2]
    text_similar.append(top_2)

with open(args.input, 'r') as f:
    a = f.readlines()

a = [i.strip() for i in a]

with open(args.labels, 'r') as f:
    labels = f.readlines()

labels = [i.strip() for i in labels]


def replace_multiple_spaces(sentence):
    return re.sub(r'\s+', ' ', sentence)

def remove_invalid_parentheses(sentence):
    stack = []
    invalid_indices = set()

    # Find indices of invalid parentheses
    for i, char in enumerate(sentence):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if not stack:
                invalid_indices.add(i)
            else:
                stack.pop()

    # Add remaining unclosed parentheses to invalid indices
    invalid_indices.update(stack)

    # Remove invalid parentheses from the sentence
    result = ''
    for i, char in enumerate(sentence):
        if i not in invalid_indices:
            result += char

    return result

def calculate_depth(sent):
    max_count = -1
    open_count = 0

    for i in sent:
        # print(i, open_count)
        if i=='(':
            open_count += 1
            max_count = max(max_count, open_count)
        elif i==')':
            open_count -= 1

    return max_count

def remove_fact(sent, depth, label):
    sent = sent.split(' ')
    indexes = [i for i, x in enumerate(sent) if x[:4] == ':ARG' and sent[i+1]=='(']

    new_indexed = []

    for i in indexes:
        tempCount = 1
        idx = i+2
        sub_depth = 1
        while tempCount>0:
            if sent[idx] == '(':
                tempCount += 1
                sub_depth = max(sub_depth, tempCount)
            elif sent[idx]== ')':
                tempCount -= 1
            idx += 1

        # print(sub_depth, sub_depth/depth)
        if sub_depth/depth < 0.34:
            new_indexed.append(i)

    # print()
    # print(new_indexed)

    temp = random.gauss(0.5, 0.05)

    temp = max(int(temp*len(new_indexed)), 1)

    text_indexed = [sent[i] for i in new_indexed]
    embeddings = model.encode(text_indexed, show_progress_bar=True, batch_size=1024)
    label_embedding = model.encode([label])[0]
    similarities_indexed = []
    for embedding in embeddings:
        simi = 1 - cosine(embedding, label_embedding)
        similarities_indexed.append(simi)

    items_with_similarities = list(zip(new_indexed, similarities_indexed))
    sorted_items = sorted(items_with_similarities, key=lambda x: x[1])
    new_indexed = [item[0] for item in sorted_items]
    new_indexed = new_indexed[:temp]

    for i in indexes:
        sent[i] = 'xxyyzz'

    sent = ' '.join(sent)

    tag = 'xxyyzz ('
    count = sent.count(tag)  # number of occurrences of the tag
    start_index = -1

    for i in range(count):
        start_index = sent.find(tag)

        if start_index==-1:
            break

        idx = start_index + len(tag)
        open_bracket_count = 1

        while open_bracket_count>0:
            if sent[idx]=='(':
                open_bracket_count+=1
            elif sent[idx]==')':
                open_bracket_count-=1
            idx+=1

        sent = sent[:start_index] + sent[idx:]

    return sent



def remove_wiki(sent):
    # print(sent)
    sent = sent.split(' ')
    indexes = [i for i, x in enumerate(sent) if x == ':wiki' and sent[i+2] == ':name']

    # print(indexes)

    for i in indexes:
        sent[i+1] = 'xxyyzz'

    sent = ' '.join(sent)

    tag = ':wiki xxyyzz :name ('
    count = sent.count(tag)  # number of occurrences of the tag
    start_index = -1

    for i in range(count):
        start_index = sent.find(tag)

        if start_index==-1:
            break

        idx = start_index + len(tag)
        open_bracket_count = 1

        while open_bracket_count>0:
            if sent[idx]=='(':
                open_bracket_count+=1
            elif sent[idx]==')':
                open_bracket_count-=1
            idx+=1

        sent = sent[:start_index] + sent[idx:]

    # print(sent)

    return sent

def remove_value(sent):
    sent = sent.split(' ')
    indexes = [i for i, x in enumerate(sent) if x == ':value' and sent[i+1].isnumeric()]
    delete = set()
    for i in indexes:
        delete.add(i)
        delete.add(i+1)

    sent = [x for i, x in enumerate(sent) if i not in delete]

    return ' '.join(sent)

def remove_numbers(sent):
    sent = sent.split(' ')
    indexes = [i for i, x in enumerate(sent) if x == ':quant' and sent[i+1].isnumeric()]
    delete = set()
    for i in indexes:
        delete.add(i)
        delete.add(i+1)

    indexes = [i for i, x in enumerate(sent) if x[:3] == ':op' and sent[i+1].isnumeric()]
    for i in indexes:
        delete.add(i)
        delete.add(i+1)

    sent = [x for i, x in enumerate(sent) if i not in delete]

    return ' '.join(sent)


def remove_mod(sent):

    tag = ':mod ('
    count = sent.count(tag)  # number of occurrences of the tag
    start_index = -1
    for i in range(count):
        start_index = sent.find(tag)

        if start_index==-1:
            break

        idx = start_index + len(tag)
        open_bracket_count = 1

        while open_bracket_count>0:
            if sent[idx]=='(':
                open_bracket_count+=1
            elif sent[idx]==')':
                open_bracket_count-=1
            idx+=1

        sent = sent[:start_index] + sent[idx:]

    return sent

tags_to_remove = [':mod', ':value', 'numbers', 'fact']


new_sents = []

for a_ind in tqdm(range(a)):
    sent = a[a_ind]
    label = labels[a_ind]
    # print(sent)
    sent = replace_multiple_spaces(sent)
    # print(sent)
    for tag in tags_to_remove:
        # print(tag)
        if tag == ':value':
            sent = remove_value(sent)
            # print('value', sent)
        if tag == 'numbers':
            # print('hello')
            sent = remove_numbers(sent)
            # print('hello')
            # print('numbers', sent)
        if tag == ':wiki':
            sent = remove_wiki(sent)
            # print(':wiki', sent)
        if tag == ':mod':
            sent = remove_mod(sent)
            # print(':mod', sent)
        if tag == 'fact':
            depth = calculate_depth(sent)
            sent = remove_fact(sent, depth, label)
            # print('fact', sent)
    new_sents.append(replace_multiple_spaces(remove_invalid_parentheses(sent)))

final_sents = []


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args_new = {}

args_new = dotdict(args_new)
args_new.solver = 'ilp'
args_new.edges = 'reify'
args_new.score_dimension = 'main'
args_new.score_type = 'micromacro'
args_new.log_level = 50
args_new.bootstrap = True
args_new.remove_duplicates = True

def extract_parentheses_substrings(string):
    substrings = []
    stack = []
    for i, char in enumerate(string):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                substrings.append(string[start:i+1])

    substrings = [i for i in substrings if len(i.strip().split()) > 3 ]

    return substrings

def generate_string_pairs(list1, list2):
    a = []
    b = []
    for str1 in list1:
        for str2 in list2:
            a.append(str1)
            b.append(str2)
    return a, b

for i in tqdm(range(len(new_sents))):
    final_sents.append(new_sents[i])

    list1 = extract_parentheses_substrings(new_sents[i][1:-1])
    list2 = extract_parentheses_substrings(new_sents[text_similar[i]][1:-1])

    a, b = generate_string_pairs(list1, list2)

    try:
        if len(a)>0:
            args_new.a = a
            args_new.b = b
            results = main_func(args_new)
            # print(a, b)
            # print(results)

            idx = results.index(max(results))

            if args.method == 'append':
                final_sents.append(new_sents[i][:-1] + b[idx] + ')')
            else:
                final_sents.append(new_sents[i].replace(a[idx], b[idx]))
        else:
            final_sents.append(new_sents[i])
    except:
        final_sents.append(new_sents[i])

    # depth = calculate_depth(new_sents[i][:-1] + new_sents[text_similar[i]][1:])
    # sent = remove_fact(new_sents[i][:-1] + new_sents[text_similar[i]][1:], depth)
    # final_sents.append(new_sents[i][:-1] + new_sents[text_similar[i]][1:])

with open(args.output, 'w') as f:
    for i in range(0,len(final_sents),2):
        f.write(final_sents[i]+'\n')

with open(args.output_mixner, 'w') as f:
    for i in range(1,len(final_sents),2):
        f.write(final_sents[i]+'\n')
