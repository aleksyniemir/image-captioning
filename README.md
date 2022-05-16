# image-captioning w języku polskim

Na tym githubie znajduje się opis wybranych zagadnień używanych w image captioning. W opis wchodzą takie rzeczy jak sposób korzystania, teoretyczny opis, jak się korzysta, w jakich artykułach zostały opublikowane, itp. Lista zagadnień:
- stemming
- leamatyzacja
- dezambiguacja
- rozpoznawanie części mowy
- korpusy językowe
- zbiory uczące
- zbiory do wstępnego uczenia
- word embeddings
- przetwarzanie języka naturalnego

## Stemming

Stemming to inaczej proces wyciągnięcia rdzenia ze słowa (części, która jest odporna na odmiany przez przyimki, rodzaje, itp).

## Lematyzacja

Lematyzacja oznacza sprowadzenie grupy wyrazów stanowiących odmianę danego zwrotu do wspólnej postaci, umożliwiając traktowanie ich wszystkich jako to samo słowo. W przetwarzaniu języka naturalnego odgrywa rolę ujednoznaczenia, np. słowa "are", "is", "Am", pochodzą od słowa "Be".

Algorytm  do lematyzacji:

	Sprowadzamy wszystkie słowa do zapisu lower case.
  
  	W głównej pętli:
    
  	Znajdujemy pozycję ostatniej litery w słowie i sprawdzamy czy jest ona jednym z kluczy w naszym słowniku reguł. Jeżeli nie, kończymy pracę algorytmu.
    
  	Pobieramy "kubełek" z regułami ze słownika na podstawie klucza (ostatniej litery).
    
  	W pętli wewnętrznej dla każdego kubełka:
    
    	Sprawdzamy, czy końcówka zapisana w regule znajduje się na końcu modyfikowanego słowa. Jeżeli nie, przechodzimy do kolejnej reguły.
      
    	Sprawdzamy czy reguła miała ustawioną flagę nienaruszalności: jeżeli tak, sprawdzamy czy słowo przez nas przetwarzane nie zostało zmienione przez poprzednie reguły. Jeżeli zostało, pomijamy działanie tej reguły.
      
    	Modyfikujemy słowo na podstawie danych z reguły.
      
    	Oznaczamy słowo jako zmodyfikowane.
      
    	Dodatkowo jeżeli reguła miała ustawioną flagę kontynuacji, wychodzimy z wewnętrznej pętli i kontynuujemy działanie algorytmu.
      

Implementacja danego algorytmu znajduje się o [tutaj](http://horusiath.blogspot.com/2012/08/nlp-stemming-i-lematyzacja.html).

## Modele językowe 

Model propabilistyczny, którego celem jest obliczenie prawdopodobieństwa wystąpienia kolejnego słowa po zadanej wcześniej sekwencji słów. Na podstawie modelu językowego możemy oszacować następne najbardziej prawdopodobne słowo.

Na samym początku wspomnę o rankingu klej - który jest polskim odpowiednikiem angielskiego rankingu GLUE. Został stworzony przez Allegro, i określa ranking najlepszych modeli językowych dla językach polskiego. Rankking możemy znaleźć o [tutaj](https://klejbenchmark.com/leaderboard/)

Do tej pory pionierem do spraw języka naturalnego w wielu językach był googlowski BERT, jednak w klasyfikacji GLUE dostał tylko 80 na 100 punktów. Na podstawie BERT’a powstał Polski RoBERT, który osiągnął 87/100 punktów w teście KLEJ, a jeszcze potem na podstawie RoBERTA powstał HerBERT, osiągając najwyższą do tej pory notę 88/100.

### BERT

Model stworzony przez Google, na jego podstawie stworzono najlepsze polskie modele. Architektura BERT polega na nauce poprzez usuwanie ze zdania losowo kilka słów, i następnie model ma się nauczyć jak najlepiej wypełnić te dziury. Przy odpowiednio dużym korpusie z czasem  model coraz lepiej poznaje zależności semantyczne między słowami. Został stworzony przy pomocy [Huggingface Transformers](https://github.com/huggingface/transformers), i ma dwa warianty: 

#### Uncased  


Polbert został przetrenowany przez kod z googlowskiego [BERTa](https://github.com/google-research/bert), ma architekture bert-based-uncased (12 warstw, 768 ukrytych, 12 heads, 110M parametrów). Set-up modelu:
- 100.000 kroków - 128 sequence length, batch size 512, learning rate 1e-4 (10.000 steps warmup)
- 800.000 kroków - 128 sequence length, batch size 512, learning rate 5e-5
- 100.000 kroków - 512 sequence length, batch size 256, learning rate 2e-5

Sposób użycia:

```python
from transformers import *
model = BertForMaskedLM.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)
for pred in nlp(f"Adam Mickiewicz wielkim polskim {nlp.tokenizer.mask_token} był."):
  print(pred)
# Output:
# {'sequence': '[CLS] adam mickiewicz wielkim polskim poeta był. [SEP]', 'score': 0.47196975350379944, 'token': 26596}
# {'sequence': '[CLS] adam mickiewicz wielkim polskim bohaterem był. [SEP]', 'score': 0.09127858281135559, 'token': 10953}
# {'sequence': '[CLS] adam mickiewicz wielkim polskim człowiekiem był. [SEP]', 'score': 0.0647173821926117, 'token': 5182}
# {'sequence': '[CLS] adam mickiewicz wielkim polskim pisarzem był. [SEP]', 'score': 0.05232388526201248, 'token': 24293}
# {'sequence': '[CLS] adam mickiewicz wielkim polskim politykiem był. [SEP]', 'score': 0.04554257541894913, 'token': 44095}
```

#### Cased

Wszystko takie same jak w cased, tylko że ze zmianą na "Whole Word Masking" - maskowanie wszystkich podsłów danego słowa.

Sposób użycia:

```python
model = BertForMaskedLM.from_pretrained("dkleczek/bert-base-polish-cased-v1")
tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-cased-v1")
nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)
for pred in nlp(f"Adam Mickiewicz wielkim polskim {nlp.tokenizer.mask_token} był."):
  print(pred)
# Output:
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim pisarzem był. [SEP]', 'score': 0.5391148328781128, 'token': 37120}
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim człowiekiem był. [SEP]', 'score': 0.11683262139558792, 'token': 6810}
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim bohaterem był. [SEP]', 'score': 0.06021466106176376, 'token': 17709}
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim mistrzem był. [SEP]', 'score': 0.051870670169591904, 'token': 14652}
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim artystą był. [SEP]', 'score': 0.031787533313035965, 'token': 35680}
```


Wyniki obydwu wariantów w rankingu KLEJ:
- Polbert cased 81.7
- Polbert uncased 81.4

### RoBERTa

Model stworzony przez Ośrodek Przetwarzania Informacji na podstawie BERTa. Powstały dwa modele, large i base, z czego large został wytrenowany na około 130GB danych, a do mniejszy na 20GB. Z obu można korzystać w zależności od potrzeb i możliwości technicznych. Pierwszy oferuje większą precyzje ale zarazem wymaga większej mocy obliczeniowej, gdzie drugi jest szybszy lecz ofertuje nieco gorsze wyniki.

Modele zostały wytrenowane na dwa sposoby, korzystając z toolkitu [fairseq](https://github.com/pytorch/fairseq) i [Huggingface Transformers](https://github.com/huggingface/transformers). Fairseq służy do modelowania sekwencyjnego i pozwala trenenować nieszablonowe modele do tłumaczeń, modeli językowych, uogalniania oraz innych zadań związanych z tekstem, gdzie Huggingface dostarcza tysiące pre-trained modeli do zadań dotyczących tekstu, obrazów, oraz audio.


<table>
<thead>
<th>Model</th>
<th>L / H / A*</th>
<th>Batch size</th>
<th>Update steps</th>
<th>Corpus size</th>
<th>KLEJ Score**</th> 
<th>Fairseq</th>
<th>Transformers</th>
</thead>
<tr>
  <td>RoBERTa&nbsp;(base)</td>
  <td>12&nbsp;/&nbsp;768&nbsp;/&nbsp;12</td>
  <td>8k</td>
  <td>125k</td>
  <td>~20GB</td>
  <td>85.39</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_fairseq.zip">v0.9.0</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-transformers-v3.4.0/roberta_base_transformers.zip">v3.4</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&#8209;v2&nbsp;(base)</td>
  <td>12&nbsp;/&nbsp;768&nbsp;/&nbsp;12</td>
  <td>8k</td>
  <td>400k</td>
  <td>~20GB</td>
  <td>86.72</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_base_fairseq.zip">v0.10.1</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_base_transformers.zip">v4.4</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&nbsp;(large)</td>
  <td>24&nbsp;/&nbsp;1024&nbsp;/&nbsp;16</td>
  <td>30k</td>
  <td>50k</td>
  <td>~135GB</td>
  <td>87.69</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_large_fairseq.zip">v0.9.0</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-transformers-v3.4.0/roberta_large_transformers.zip">v3.4</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&#8209;v2&nbsp;(large)</td>
  <td>24&nbsp;/&nbsp;1024&nbsp;/&nbsp;16</td>
  <td>2k</td>
  <td>400k</td>
  <td>~200GB</td>
  <td>88.87</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_large_fairseq.zip">v0.10.2</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_large_transformers.zip">v4.14</a>
  </td>
</tr>
  <tr>
  <td>DistilRoBERTa</td>
  <td>6&nbsp;/&nbsp;768&nbsp;/&nbsp;12</td>
  <td>1k</td>
  <td>10ep.</td>
  <td>~20GB</td>
  <td>84.55</td>
  <td>
  n/a
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/distilroberta_transformers.zip">v4.13</a>
  </td>
</tr>
</table>

\* L - the number of encoder blocks, H - hidden size, A - the number of attention heads <br/>

#### Jak używać z Fairseq:

```python
import os
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils

model_path = "roberta_large_fairseq"
loaded = hub_utils.from_pretrained(
    model_name_or_path=model_path,
    data_name_or_path=model_path,
    bpe="sentencepiece",
    sentencepiece_vocab=os.path.join(model_path, "sentencepiece.bpe.model"),
    load_checkpoint_heads=True,
    archive_map=RobertaModel.hub_models(),
    cpu=True
)
roberta = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
roberta.eval()
input = roberta.encode("Zażółcić gęślą jaźń.")
output = roberta.extract_features(input)
print(output[0][1])
```

#### Jak używać z Hugging Transformers

```python
import torch, os
from transformers import RobertaModel, AutoModel, PreTrainedTokenizerFast

model_dir = "roberta_base_transformers"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model: RobertaModel = AutoModel.from_pretrained(model_dir)
input = tokenizer.encode("Zażółcić gęślą jaźń.")
output = model(torch.tensor([input]))[0]
print(output[0][1])
```

### HerBERT

Najlepiej sprawujący się model językowy do języka polskiego. Oparty jest na strukturze BERTa, a wzoruje się centralnie na RoBERTcie. Jest jego zoptymalizowaną i zmodyfikowaną wersją, która została przetrenowana na większej ilości danych. Istnieją trzy warianty:

| Model | Tokenizer | Vocab Size | Batch Size | Train Steps | KLEJ Score |
| :---- | --------: | ---------: | ---------: | ----------: | ---------: | 
| `herbert-klej-cased-v1` | BPE | 50K | 570 | 180k | 80.5 |
| `herbert-base-cased` | BPE-Dropout | 50K | 2560 | 50k | 86.3 |
| `herbert-large-cased` | BPE-Dropout | 50K | 2560 | 60k | 88.4 |

Przykład jak załadować model:

```python
from transformers import AutoTokenizer, AutoModel

model_names = {
    "herbert-klej-cased-v1": {
        "tokenizer": "allegro/herbert-klej-cased-tokenizer-v1", 
        "model": "allegro/herbert-klej-cased-v1",
    },
    "herbert-base-cased": {
        "tokenizer": "allegro/herbert-base-cased", 
        "model": "allegro/herbert-base-cased",
    },
    "herbert-large-cased": {
        "tokenizer": "allegro/herbert-large-cased", 
        "model": "allegro/herbert-large-cased",
    },
}

tokenizer = AutoTokenizer.from_pretrained(model_names["allegro/herbert-base-cased"]["tokenizer"])
model = AutoModel.from_pretrained(model_names["allegro/herbert-base-cased"]["model"])
```

Przykład jak używać modelu:

```python
output = model(
    **tokenizer.batch_encode_plus(
        [
            (
                "A potem szedł środkiem drogi w kurzawie, bo zamiatał nogami, ślepy dziad prowadzony przez tłustego kundla na sznurku.",
                "A potem leciał od lasu chłopak z butelką, ale ten ujrzawszy księdza przy drodze okrążył go z dala i biegł na przełaj pól do karczmy."
            )
        ],
        padding="longest",
        add_special_tokens=True,
        return_tensors="pt",
    )
)
```


[Praca](https://aclanthology.org/2021.bsnlp-1.1/) w której został dokładnie opisany HerBERT.

### Model językowy autorstwa Teresy Sas

Model jest zbudowany na zasobach tekstów o tematyce akademickiej i ogólnej, obejmując prawie 800 tyś. słów. W linku możemy pobrać zipa w którym znajdują się trzy pliki:
•	model_2d_forward.txt - model bigramowy zawierający prawdopodobieństwa następstwa słów p(w_i | w_{i-1} ) dla porządku od lewej do prawej
•	model 3d_bakward.txt - model trigramowy zawierający prawdopodobieństwa p( w_i | w_{i+1} w_{i+2} ) dla porządku odwróconego
•	word_list.txt - lista słów występujących w modelu

Model może być wykorzystany w badaniach nad rozpoznawaniem mowy i w inżynierii języka naturalnego dla języka polskiego.


## Korpusy językowe

Korpus językowy to zbiór tekstów służących badaniom lingwistycznym, np. określaniu częstości występowania form wyrazowych, konstrukcji składniowych lub kontekstów, w jakich pojawiają się dane wyrazy.

### Korpus NKJP
Istnieją dwie wyszukiwarki stworzone na jego bazie:

#### IPI PANh

ttp://nkjp.pl/poliqarp/

[Ściągawka](http://nkjp.pl/poliqarp/help/pl.html) do używania korpusu. Znajdują się w niej między innymi zapytania o:
- segmenty
- formy podstawowe
- znaczniki morfosyntaktyczne
- wieloznaczność i dezambiguacja

####	PELCRA

http://www.nkjp.uni.lodz.pl/

### Korpus opisanych obrazów z adnotacjami

http://zil.ipipan.waw.pl/Scwad/AIDe

### Korpus dyskursu parlamentarnego

https://kdp.nlp.ipipan.waw.pl/query_corpus/

## Word embeddings

[Ewaluacja polskich embeddingów](https://github.com/Ermlab/polish-word-embeddings-review) stworzona przez wiele grup badawczych.

W powyższej implementacji została użyta biblioteka „gensim”, która pozwala wczytać dane, wytrenować model, oraz końcowo sprawdzić wyniki. Oprócz tworzenia wektorowych przedstawień słów biblioteka ma wiele innych funkcji, między innymi może znajdować semantycznie podobne dokumenty do tego który został jej zadany. Żeby określić word-embedinngi w powyższej implementacji używana jest funkcja "evaluation_word_pairs" oraz "assessment_word_analogies" z biblioteki gensim.

Do testowania został użyty plik dostarczony przez [facebook fastext](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.vec.gz). Żeby określić podobieństwo słów użyto polskiej wersji [SimLex999](http://zil.ipipan.waw.pl/CoDeS?action=AttachFile&do=view&target=MSimLex999_Polish.zip), stworzonej przez IPIPAN.

Obszerna tabelka z wynikami znajduje się w podanym w pierwszej linijce linku.


## Zbiory danych

[Zbiór](https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-pl.txt) zawiwerający pary słów a następnie skojarzone z nimi następne pary słów.

[Zbiór](https://opus.nlpl.eu/OpenSubtitles-v2018.php) zawierający polskie napisy do filmów. Z dwóch źródeł się dowiedziałem, że zawiera sporo powtórzeń.

### Zbiory użyte do embeddingów

- https://clarin-pl.eu/dspace/handle/11321/442
- https://clarin-pl.eu/dspace/handle/11321/606
- https://clarin-pl.eu/dspace/handle/11321/600
- https://clarin-pl.eu/dspace/handle/11321/327
- http://vectors.nlpl.eu/repository (id: 62,167)
- https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz

### Zbiory użyte do HerBERTa
- https://huggingface.co/datasets/allegro/klej-psc
- https://huggingface.co/datasets/allegro/klej-dyk
- https://huggingface.co/datasets/allegro/klej-polemo2-in
- https://huggingface.co/datasets/allegro/klej-cdsc-e
- https://huggingface.co/datasets/allegro/klej-allegro-reviews
- https://huggingface.co/datasets/allegro/klej-polemo2-out
- https://huggingface.co/datasets/allegro/klej-nkjp-ner
- https://huggingface.co/datasets/allegro/klej-cdsc-r
- https://huggingface.co/datasets/allegro/klej-cbd
- https://huggingface.co/datasets/allegro/summarization-polish-summaries-corpus