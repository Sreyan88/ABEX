import os
os.environ['TRANSFORMERS_CACHE'] = './cache/hub'
import tiktoken
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, GenerationConfig
from torch.utils.data import DataLoader
from huggingface_hub.hf_api import HfFolder
import datasets


from datasets import load_dataset, Dataset

model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left')

# raw_datasets = load_dataset('c4', 'realnewslike')

# print(raw_datasets)

dataset = load_dataset('csv', data_files={'train': 'llama-final.csv'})
abstract_list = dataset['train']['abstract']

raw_datasets = datasets.DatasetDict({'text': abstract_list, 'timestamp': [0]*len(abstract_list), 'url':len(abstract_list)*[0]})
raw_datasets = Dataset.from_dict(raw_datasets)
raw_datasets = datasets.DatasetDict({'train': raw_datasets})

print(raw_datasets)

temp = len(raw_datasets['train'])
start = 0+1853+1853
end = int(0.2*temp)

raw_datasets['train'] = raw_datasets['train'].select(range(start, end))

print(len(raw_datasets['train']))

@dataclass
class Template:

    prefix: List[Union[str, Dict[str, str]]]
    prompt: List[Union[str, Dict[str, str]]]
    system: str
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool
    efficient_eos: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids, answer_ids = prompt_ids + encoded_pairs[-1][0], encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        return encoded_pairs

    def _format(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        r"""
        Aligns inputs to the standard format.
        """
        system = system or self.system # use system if provided
        history = []
        history = history + [(query, resp)]
        return system, history

    def _get_special_ids(
        self,
        tokenizer: "PreTrainedTokenizer"
    ) -> Tuple[List[int], List[int]]:
        if tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", True):
            bos_ids = [tokenizer.bos_token_id]
        else: # baichuan, qwen and gpt2 models have no bos token
            bos_ids = []

        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token is required.")

        if self.efficient_eos: # used in baichuan, qwen, chatglm, etc.
            eos_ids = []
        else:
            eos_ids = [tokenizer.eos_token_id]

        return bos_ids, eos_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        query: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + sep + query    resp + eos
        Turn t: sep + bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self._convert_inputs_to_ids(tokenizer, context=self.sep)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0:
                prefix_ids = self._convert_inputs_to_ids(tokenizer, context=self.prefix, system=system)
                if len(prefix_ids) != 0: # has prefix
                    prefix_ids = bos_ids + prefix_ids + sep_ids
                else:
                    prefix_ids = bos_ids
            else:
                prefix_ids = sep_ids + bos_ids

            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query, idx=str(turn_idx))
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((prefix_ids + query_ids))

        return encoded_pairs

    def _convert_inputs_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        context: List[Union[str, Dict[str, str]]],
        system: Optional[str] = None,
        query: Optional[str] = None,
        idx: Optional[str] = None
    ) -> List[int]:
        r"""
        Converts context to token ids.
        """
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        token_ids = []
        for elem in context:
            if isinstance(elem, str):
                elem = elem.replace("{{system}}", system, 1) if system is not None else elem
                elem = elem.replace("{{query}}", query, 1) if query is not None else elem
                elem = elem.replace("{{idx}}", idx, 1) if idx is not None else elem
                if len(elem) != 0:
                    token_ids = token_ids + tokenizer.encode(elem, **kwargs)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            else:
                raise ValueError("Input must be string or dict[str, str], got {}".format(type(elem)))

        return token_ids

class Llama2Template(Template):

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + query    resp + eos
        Turn t: bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0: # llama2 template has no sep_ids
                query = self.prefix[0].replace("{{system}}", system) + query
            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query)
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((bos_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs

templates: Dict[str, Template] = {}

def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    system: str,
    sep: List[Union[str, Dict[str, str]]],
    stop_words: Optional[List[str]] = [],
    use_history: Optional[bool] = True,
    efficient_eos: Optional[bool] = False
) -> None:
    template_class = Llama2Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        system=system,
        sep=sep,
        stop_words=stop_words,
        use_history=False,
        efficient_eos=efficient_eos
    )

register_template(
    name="llama2",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST]"
    ],
    system=(
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
    sep=[]
)


def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:

    # if tokenizer.eos_token_id is None:
    #     tokenizer.eos_token = "<|endoftext|>"
    #     # logger.info("Add eos token: {}".format(tokenizer.eos_token))

    # if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
        # logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if name is None:
        return None

    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=template.stop_words),
        replace_additional_special_tokens=False
    )
    return template

template = get_template_and_fix_tokenizer("llama2", tokenizer)

prompt = """I will provide you with a small document. You need to return a short and abstract description of it. Don't mention named entities, and just describe the key message of the document in a few words.
Here are some examples:
Input 1: Shatrughan Sinha, a Congress candidate and actor-politician, will run against Union Law Minister Ravi Shankar Prasad, a BJP candidate, in the Patna Sahib seat. Sinha has dismissed BJP's claim that the seat is their stronghold and has expressed his confidence in winning the election. He has also criticized the BJP's decision to field Prasad, a four-term Rajya Sabha member, in the seat. Sinha has served two terms in the Rajya Sabha and has been a member of the union council of ministers. He has also defended his record, citing his spending of 106% of his MPLAD fund, which is available on the net.
Output 1: A political competition between two candidates from major parties for a significant electoral seat, involving critique of the opposition's choice and defense of personal achievements.
Input 2: Said Baalbaki, a Palestinian artist, has curated an exhibition featuring 50 of Abbo's sketches, etchings, and objects, along with texts from Baalbaki's personal collection, showcasing the elusive sculptor's work and life.
Output 2: An exhibition curated by an artist, displaying sketches, etchings, and objects from a lesser-known sculptor, accompanied by personal texts, highlighting the sculptor's work and life.
Here is the input document:"""

def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
    for i in range(len(examples['text'])):
        query = examples["text"][i]
        temp = tokenizer.tokenize(query)[:512]
        try:
            last_occurence=len(temp)-temp[::-1].index('.')-1
        except:
            last_occurence=511
        temp = temp[:last_occurence+1]
        query = tokenizer.convert_tokens_to_string(temp)
        query = prompt + "\n" + query
        yield query

def preprocess_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:

    model_inputs = {"input_ids": [], "attention_mask": []}

    for query in construct_example(examples):
        # print(query)
        input_ids, _ = template.encode_oneturn(tokenizer, query, "")

        if len(input_ids) > 2000:
            input_ids = input_ids[:2000]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))

    return model_inputs

kwargs = dict(num_proc=4, load_from_cache_file=True, desc="Running tokenizer on dataset")

dataset = raw_datasets.map(preprocess_dataset, batched=True, remove_columns=['text', 'timestamp', 'url'], **kwargs)

data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        label_pad_token_id=tokenizer.pad_token_id
    )

# temp = len(dataset['train'])//2
# print(len(dataset['train']), len(dataset['validation']))
dataloader = DataLoader(dataset['train'], batch_size=4, collate_fn=data_collator)
# print(dataloader)
# ,access_token="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"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",  use_safetensors = False)

generating_args = {}
generating_args.update(dict(
    # do_sample=True,
    # temperature=0.7,
    # top_p=0.1,
    # top_k=50,
    num_return_sequences=1,
    # repetition_penalty=1.2,
    eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
    pad_token_id=tokenizer.pad_token_id,
    max_length = 2000
))


from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList

def get_logits_processor() -> LogitsProcessorList:
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


for data in tqdm(dataloader):

    gen_kwargs = dict(
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor()
        )
    generate_output = model.generate(input_ids = data['input_ids'].cuda(), attention_mask = data['attention_mask'].cuda(),**gen_kwargs)
    # print(generate_output)

    # response_ids = generate_output[:, prompt_length:]
    response = tokenizer.batch_decode(generate_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    with open('llama0-20.txt', 'a') as f:
        for i in response:
            temp_prompt = i.split('[/INST]')[0]
            temp_res = i.split('[/INST]')[1]
            write_text = f'{temp_prompt}\t,\t{temp_res}'.encode('ascii', 'ignore').decode('ascii')
            f.write(write_text)
            f.write('\n\n&&&\n\n')
            # print("Prompt: ", i.split('[/INST]')[0])
            # print("Response: ", i.split('[/INST]')[1])

    # response_length = 0
    # for i in range(len(response_ids)):
    #     eos_index = (response_ids[i] == tokenizer.eos_token_id).nonzero()
    #     response_length += eos_index[0].item() if len(eos_index) else len(response_ids[i])




# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
# model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# output = model.generate(**model_inputs)

# print(tokenizer.decode(output[0], skip_special_tokens=True))