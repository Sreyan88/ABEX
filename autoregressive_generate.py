import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
filepath = '/fs/nexus-projects/audio-visual_dereverberation/sonal/diff/data/low_res/100/classification/huff_train_processed_filtered_generated.source'
tokenizer = AutoTokenizer.from_pretrained('/fs/nexus-projects/audio-visual_dereverberation/sonal/amr-data-augmentation/checkpoint-7480')
model_pipeline = pipeline("text2text-generation", model='/fs/nexus-projects/audio-visual_dereverberation/sonal/amr-data-augmentation/checkpoint-7480', tokenizer=tokenizer, device=0)

with open(filepath, 'r') as f:
    text = f.readlines()

text = [x.strip() for x in text]

train_og = pd.read_csv('/fs/nexus-projects/audio-visual_dereverberation/sonal/diff/data/low_res/100/classification/huff_train.tsv', delimiter='\t')
train_og = train_og.values

assert len(train_og) == len(text)

with open('random100.txt', 'w') as the_file:
    test = 0
    the_file.write('text\tlabel\n')
    for i in tqdm(range(len(text))):
        new_sketch = text[i]
        for f in range(5):
            generated_text = model_pipeline(new_sketch, num_beams=1, top_k=100, do_sample=True, max_length=256, num_return_sequences=1)
            # print(f'Aug {f}: ', generated_text[0]['generated_text'])
            aug = generated_text[0]['generated_text']
            label_aug = train_og[i][1]
            the_file.write(f'{aug}\t{label_aug}\n')