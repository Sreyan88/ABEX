# ABEX
Code for ACL 2024 paper  -- ABEX: Data Augmentation for Low-Resource NLU via Expanding Abstract Descriptions

![Proposed Methodology](./assets/ABEX-ACL.drawio.jpg)

### Steps to generate data augmentations from ABEX

1. Install dependencies using:
```
pip install -r requirements.txt
```

2. Setup GitHub repository [AMR-DA: Data Augmentation by Abstract Meaning Representation](https://github.com/zzshou/amr-data-augmentation) for Text to AMR and AMR to Text pipelines

3. Take sample data from ![test_data](./test_data) and use the Spring AMR Parser to convert it to AMR

4. Run our AMR filtering pipeline:
```
python src/filter_amr.py
```

5. Convert the filtered AMRs back to text using plms-graph2text AMR generator

6. Use ![autoregressive_generate](./src/autoregressive_generate.py) to expand the edited abstract texts
