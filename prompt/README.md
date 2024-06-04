## Prompt Details

Document to Summary:
```
Write me a summary of the article in one line. Don’t include entities; write the summary just describing key events and concepts in the article. Here is the article:
```


Summary to Abstract:
```
For generating an abstract from the summary of a document in $\mathcal{D}_u$ with LLaMA-2 we use the following prompt: \textit{I will provide you with a small document. You need to return a short and abstract description of it. Don't mention named entities, and just describe the key message of the document in a few words.
Here are some examples:
Input 1: Shatrughan Sinha, a Congress candidate and actor-politician, will run against Union Law Minister Ravi Shankar Prasad, a BJP candidate, in the Patna Sahib seat. Sinha has dismissed BJP's claim that the seat is their stronghold and has expressed his confidence in winning the election. He has also criticized the BJP's decision to field Prasad, a four-term Rajya Sabha member, in the seat. Sinha has served two terms in the Rajya Sabha and has been a member of the union council of ministers. He has also defended his record, citing his spending of 106\% of his MPLAD fund, which is available on the net.
Output 1: A political competition between two candidates from major parties for a significant electoral seat, involving critique of the opposition's choice and defense of personal achievements.
Input 2: Said Baalbaki, a Palestinian artist, has curated an exhibition featuring 50 of Abbo's sketches, etchings, and objects, along with texts from Baalbaki's personal collection, showcasing the elusive sculptor's work and life.
Output 2: An exhibition curated by an artist, displaying sketches, etchings, and objects from a lesser-known sculptor, accompanied by personal texts, highlighting the sculptor's work and life.
Here is the input document:
```

[llama.py](./llama.py) is a demo code which can be used to generate new abstract - expansion pairs on a different dataset!

Note: Make syre to update the preprocessing code (lines 16-26)