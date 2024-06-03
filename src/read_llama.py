from tqdm import tqdm
import pandas as pd
import numpy as np
np.random.seed(42)

for file in ['llama0-20', 'llama20-40', 'llama40-60','llama60-80', 'llama80-100']:
    with open(f'{file}.txt', 'r') as f:
        a = f.readlines()

    temp = ''
    final = []
    for i in tqdm(range(len(a)-2)):
        # print('-------', a[i])
        if a[i]=='\n' and a[i+1]=='&&&\n' and a[i+2]=='\n' and temp:
            final.append(temp)
            temp = ''
            i+=3
            continue

        temp += ' ' + a[i].strip()
        temp = temp.strip()

    if temp:
        final.append(temp.strip())

    # print(len(final))
    # print(final[0])
    # asdasd

    csv = []
    count = 0
    for i in final:
        if len(i.split('\t,\t'))==2 and ':' not in i.split('\t,\t')[1] and 'Sure!' in i.split('\t,\t')[1]:
            count += 1
            print(i)
            asdasd
        # else:
        #     print(i)
        #     asdas


    for i in tqdm(final):
        try:
            prompt, reply = i.split('\t,\t')
            prompt = prompt.split('Here is the input document: ')[1].strip()
            if ':' in reply:
                reply = reply.split(':')[-1].strip()
            reply = reply.strip()
            csv.append([prompt,reply])
        except:
            print(i)

    x = pd.DataFrame(csv, columns=['prompt', 'abstract'])

    x.to_csv(f'{file}.csv', index=False)


df1 = pd.read_csv('llama0-20.csv')
df2 = pd.read_csv('llama20-40.csv')
df3 = pd.read_csv('llama40-60.csv')
df4 = pd.read_csv('llama60-80.csv')
df5 = pd.read_csv('llama80-100.csv')
final = pd.concat([df1, df2, df3, df4, df5], axis=0)
print(len(final))
final.drop_duplicates(inplace=True)
final = final.dropna()
print(len(final))
final.to_csv('llama0-100.csv', index=False)

final = final.values
np.random.shuffle(final)

divide = int(0.95*len(final))
train = final[:divide]
eval = final[divide:]

with open('./data/abstract/train.tgt', 'w') as f1:
    with open('./data/abstract/train.src','w') as f2:
        for i in train:
            f1.write(i[0]+'\n')
            f2.write(i[1]+'\n')

with open('./data/abstract/dev.tgt', 'w') as f1:
    with open('./data/abstract/dev.src','w') as f2:
        for i in eval:
            f1.write(i[0]+'\n')
            f2.write(i[1]+'\n')

print('warmup steps: ', int(0.05*len(train)))
