# ABEX
Code for ACL 2024 paper -- [ABEX: Data Augmentation for Low-Resource NLU via Expanding Abstract Descriptions](https://arxiv.org/abs/2406.04286)

![Proposed Methodology](./assets/ABEX-ACL.drawio.jpg)

Note: Synthetic data generation demo script and prompts used have been shared in [prompt](./prompt).

If you wish to train your own model on our synthetic abstract-expand dataset, you can use the dataset given [here](./data/) - train.src, train.tgt, dev.src, dev.tgt.
Our pretrained ABstract-EXpand model has been uploaded on HuggingFace: [utkarsh4430/ABEX-abstract-expand](https://huggingface.co/utkarsh4430/ABEX-abstract-expand).
Use the following code to generate expansions from abstract texts.

```python
from transformers import pipeline, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('utkarsh4430/ABEX-abstract-expand')
model_pipeline = pipeline("text2text-generation", model='utkarsh4430/ABEX-abstract-expand', tokenizer=tokenizer)

input_text = 'A chance to meet WWE stars and support a good cause.'

for f in range(5):
    generated_text = model_pipeline(input_text, num_beams=1, top_k=100, do_sample=True, max_length=350, num_return_sequences=1)
    print(f'Aug {f}: ', generated_text[0]['generated_text'])
```

Example Output:
```
Aug 0:  WWE stars to visit Detroit on January 20th for the second time in three years, with appearances at The Battle at the Fox Sports 1 World Headquarters, and proceeds going to a charity of your choice.
Aug 1:  A one-on-one experience at the WWE Creative Conference in New Jersey was provided, with an opportunity for the audience to meet WWE superstars and support a good cause.
Aug 2:  Sindrun welcomes WWE star Chris Jericho and hosts an event for attendees to meet WWE stars and support a local cause.
Aug 3:  Find out if you can meet WWE stars, including the Rock and Shake, at a benefit luncheon.
Aug 4:  The WWE Talent Showcase 2019 will feature exciting moments inside the WWE Studios, including the first one in over a decade, and features a chance to hug current and former stars and receive a check from a corporate sponsor.
```

The work adopts [SPRING](https://github.com/SapienzaNLP/spring) as AMR parser and [plms-graph2text](https://github.com/UKPLab/plms-graph2text) as AMR generator.

### Steps to generate data augmentations from ABEX

1. Install dependencies using:
```
pip install -r requirements.txt
```

2. Reformat the input files into the required format using:
```shell
python process.py --input <input file> --output <output file> --type <bio|tsv|sim>
```

3. Setup GitHub repository [AMR-DA: Data Augmentation by Abstract Meaning Representation](https://github.com/zzshou/amr-data-augmentation) for Text to AMR and AMR to Text pipelines

4. Setup [smatchpp](https://github.com/flipz357/smatchpp) github repository.

5. Text to AMR - Use output from Step 2 to get AMR graph
```shell
cd amr-parser-spring
bash predict_amr.sh <plain_text_file_path>(../data/wiki_data/wiki.txt)
```
Preprocess amr graph, convert to source and target string
```shell
cd data-utils/preprocess
bash prepare_data.sh <amr_file_path>(../../data/wiki_data/wiki.amr)
```

6. Run our AMR filtering pipeline:
```shell
python src/filter_amr.py
```

7. Convert the filtered AMRs back to text using plms-graph2text AMR generator
```shell
cd plms-graph2text
bash decode_AMR.sh <model-path> <checkpoint> <gpu_id> <source file> <output-name>
(bash decode_AMR.sh /path/to/t5-base amr-t5-base.ckpt 0 ../data/wiki-data/wiki.source wiki-perd-t5-base.txt)
```

8. Use [autoregressive_generate](./src/autoregressive_generate.py) to expand the edited abstract texts

9. Use the [hf_consistency.py](./src/hf_consistency.py) to check for consistency

10. Use [gold_classifier.py](./src/gold_classifier.py) to finally get the required metrics

Citation:

```
@inproceedings{
    ghosh2024abex,
    title={{ABEX}: Data Augmentation for Low-Resource {NLU} via Expanding Abstract Descriptions},
    author={Sreyan Ghosh and Utkarsh Tyagi and Sonal Kumar and Chandra Kiran Reddy Evuru and and Ramaneswaran S and S Sakshi and Dinesh Manocha},
    booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
    year={2024},
}
```
