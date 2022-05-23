# image-captioning w języku polskim

Na tym githubie znajduje się opis wybranych zagadnień używanych w image captioning do języka polskiego. 
W opis wchodzą takie rzeczy jak sposób korzystania, teoretyczny opis, jak się korzysta, 
w jakich artykułach zostały opublikowane, itp. 

Lista zagadnień:
- stemming
- leamatyzacja
- tagowanie
- tokenizacja
- korpusy językowe
- zbiory uczące
- zbiory do wstępnego uczenia
- word embeddings
- przetwarzanie języka naturalnego

## Stemming

Stemming to inaczej proces wyciągnięcia rdzenia ze słowa (części, która jest odporna na odmiany 
przez przyimki, rodzaje, itp). 

### NTLK
NLTK zawiera dużo bibliotek wykonujących stemming na wiele sposobów jak i narzędzia wykonujące
inne funkcje. 
Jest to platforma, która tworzy aplikacje opierające się o dane dotyczące ludzkiego języka.
Udostępnia łatwe w użyciu interfejsy do ponad 50 korpusów i zasobów leksykalnych, takich jak WordNet,
wraz z zestawieniem bibliotek do przetwarzania tekstu do klasyfikacji, tokenizacji, stemmingu,
tagowania, parsowania, wnioskowania semantycznego oraz wrapperami do bibliotek NLP o znaczeniu przemysłowym.

NLTK dostarcza wiele implementacji różnych algorytmów, między innymi:
* arlstem - do języka Arabskiego
* cistem - do języka Niemieckiego
* isri - do języka Arabskiego
* rslp - do języka Portugalskiego
* snowball - obsługuje parenaście języków, niestety bez języka polskiego
* lancester - oparty na algorytmie Lancestera
* porter - oparty na algorytmie Portera

Dla przykładu sposób działania stemmera Regexp, który używa wyrażeń regularnych do identyfikacji 
afiksów morfologicznych i usuwa wszystkie podłańcuchy pasujące do wyrażeń regularnych:
```python
>>> from nltk.stem import RegexpStemmer
>>> st = RegexpStemmer('ing$|s$|e$|able$', min=4)
>>> st.stem('cars')
'car'
>>> st.stem('mass')
'mas'
>>> st.stem('was')
'was'
>>> st.stem('bee')
'bee'
>>> st.stem('compute')
'comput'
>>> st.stem('advisable')
'advis'
```

Więcej informacji apropo każdego z wymienionych stemmerów znajduje się [tutaj](https://www.nltk.org/api/nltk.stem.html).

### Stempel
Jeśli chodzi specyficznie o język polski z pomocą przychodzi Stempel - stemmer początkowo 
napisany w javie. Został stworzony w projekcie [Egothor](https://www.egothor.org/product/egothor2/),
który polegał na stworzeniu Open Source wyszukiwarki tekstu ze wszystkimi funkcjami tego zagadnienia.
Po jakimś czasie zostal też włączony jako część [Apache Lucene](https://lucene.apache.org/core/3_1_0/api/contrib-stempel/index.html),
 darmowej i Open Source biblioteki wyszukiwarek. Jest on również wykorzystywana przez wyszukiwarkę 
Elastic Search.

Pakiet zawiera tabelki stemmingowe dla języka polskiego, oryginalnie przygotowane na
20,000 zestawach treningowych, oraz nową wersję przygotowaną na 259,000 zestawach ze
słownika Polimorf.

#### Przykład użycia:
Użycie oryginalnej wersji:
```python
>>> stemmer = StempelStemmer.default()
```
Lub wersji z nowszą tabelką steemingową pretrained na słowniku Polimorf:
```python
>>> stemmer = StempelStemmer.polimorf()
```
Użycie:
```python
>>> for word in ['książka', 'książki', 'książkami', 'książkowa', 'książkowymi']:
...   print(stemmer(word))
...
książek
książek
książek
książkowy
książkowy
```

Więcej informacji o steemerze znajduje się o [tutaj](https://github.com/dzieciou/pystempel).


## Lematyzacja i Tagowanie

Lematyzacja oznacza sprowadzenie grupy wyrazów stanowiących odmianę danego zwrotu do wspólnej 
postaci, umożliwiając traktowanie ich wszystkich jako to samo słowo. W przetwarzaniu języka 
naturalnego odgrywa rolę ujednoznaczenia, np. słowa "are", "is", "Am", pochodzą od słowa "Be".

Tagowanie to inaczej proces klasyfikowania słów na ich części mowy i odpowiedniego ich oznaczania .
Części mowy są również znane jako klasy słów lub kategorie leksykalne. Zbiór znaczników używanych 
w danym zadaniu nazywany jest zbiorem znaczników. Praca która dokładnie opisuje wiele aspektów
dotyczących tagowania znajduje się [tutaj](http://nlp.ipipan.waw.pl/Bib/kob:kie:16.pdf).

### Morfeusz 2

Morfeusz 2 stanowi nową implementację
analizatora morfologicznego dla języka polskiego. Poprzednie wersje programu  stały się 
podstawą wielu narzędzi przetwarzania języka, w szczególności kilku tagerów
i parserów języka polskiego. 
Wersja druga programu, opracowana jako część infrastruktury Clarin-PL,
różni się od porzednika wieloma ważnymi ulepszeniami.

Oto przykład wyników działania programu dla tekstu „Mam próbkę analizy morfologicznej.”:

|Segment	|Forma ortograficzna | 	Lemat                                | 	Znacznik                 |
| --------  | ------------------ |---------------------------------------|---------------------------|
|0–1	    |Mam	              | mama	                                 | subst:pl:gen:f            |
|        |                       | mamić	                                | impt:sg:sec:imperf        |
|        |                       | mieć	                                 | fin:sg:pri:imperf         |
|1–2	    |próbkę	               | próbka	                               | subst:sg:acc:f            |
|2–3	    |analizy	           | analiza	                              | subst:sg:gen:f            |
|        |                         |                                       | subst:pl:nom.acc.voc:f    |
|3–4	    |morfologicznej	        | morfologiczny | 	adj:sg:gen.dat.loc:f:pos |
|4–5	    |.	                    | .	       |  interp                   |

Istnieją trzy warianty danych fleksyjnych dostępnych w Morfeuszu 2:

* najstarszy — SIaT, przygotowany poprzez skonfrontowanie danych „Schematycznego 
indeksu a tergo polskich form wyrazowych” (SIaT) Jana Tokarskiego i Zygmunta Saloniego z
listą haseł słownika Doroszewskiego,
* SGJP, korzystający z danych „Słownika gramatycznego języka polskiego” — SGJP (dane liczbowe
można znaleźć tutaj),
* Polimorf, wykorzystujący słownik fleksyjny Polimorf stanowiący połączenie danych SGJP z 
tworzonymi społecznościowo danymi Morfologika/sjp.pl.

### Przykład użycia

##### GUI
Morfeusz 2 zawiera [GUI](http://morfeusz.sgjp.pl/download/gui/) napisane w C++, które nie wymaga przykładu użycia.

##### Użycie z poziomu C++
[Dokumentacja](http://download.sgjp.pl/morfeusz/Morfeusz2.pdf) zawiera sześć stron (7-13) dokładnie opisanych kroków,
które należy wykonać do użycia interesująych nas opcji.




## Tokenizacja 

Tokenizatory dzielą ciągi znaków na listy podłańcuchów, na przykład mogą 
być używane do znajdowania słów i znaków interpunkcyjnych w łańcuchu znaków.

### NTLK
Przykład użycia tokenizatora z [paczki NTLK](https://www.nltk.org/api/nltk.tokenize.html), najprostszy moduuł
tokenizujący:
```python
>>> from nltk.tokenize import word_tokenize
>>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
... two of them.\n\nThanks.'''
>>> word_tokenize(s)
['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.',
'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
```

NLTK udostępnia również prostszy tokenizer oparty na wyrażeniach regularnych, który
dzieli tekst na podstawie spacji i interpunkcji:

```python
>>> from nltk.tokenize import wordpunct_tokenize
>>> wordpunct_tokenize(s)
['Good', 'muffins', 'cost', '$', '3', '.', '88', 'in', 'New', 'York', '.',
'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
```

Możemy również operować na poziomie zdań, używając tokenizatora zdań 
w następujący sposób:

```python
>>> from nltk.tokenize import sent_tokenize, word_tokenize
>>> sent_tokenize(s)
['Good muffins cost $3.88\nin New York.', 'Please buy me\ntwo of them.', 'Thanks.']
>>> [word_tokenize(t) for t in sent_tokenize(s)]
[['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.'],
['Please', 'buy', 'me', 'two', 'of', 'them', '.'], ['Thanks', '.']]
```

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

Model stworzony przez Ośrodek Przetwarzania Informacji na podstawie BERTa. Powstały dwa modele, large i base, z czego large został wytrenowany na około 130GB danych, a do mniejszy na 20GB - wśród korpusy znajdowały się wysokiej jakości teksty z wikipedii, dokumenty polskiego parlamentu, wypowiedzi z mediów społecznościowych, książki, artykuły, oraz dłuższe formy pisane. Z obu można korzystać w zależności od potrzeb i możliwości technicznych. Pierwszy oferuje większą precyzje ale zarazem wymaga większej mocy obliczeniowej, gdzie drugi jest szybszy lecz ofertuje nieco gorsze wyniki.

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

#### Jak używać Hugging Transformers

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

Więcej informacji apropo modelu można zaleźć o [tutaj](https://zasobynauki.pl/zasoby/model-jezykowy-dla-jezyka-polskiego,55644/).










## Word embeddings

### Ewaluacja polskich word embeddingów stworzona przez wiele grup badawczych

W powyższej implementacji została użyta biblioteka „gensim”, która pozwala wczytać dane, wytrenować model, 
oraz końcowo sprawdzić wyniki. Oprócz tworzenia wektorowych przedstawień słów biblioteka ma wiele innych funkcji,
między innymi może znajdować semantycznie podobne dokumenty do tego który został jej zadany. Żeby określić 
word-embedinngi w powyższej implementacji używana jest funkcja "evaluation_word_pairs" oraz "assessment_word_analogies" z biblioteki gensim.

Do testowania został użyty plik dostarczony przez [facebook fastext](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.vec.gz). Żeby określić podobieństwo słów użyto polskiej wersji [SimLex999](http://zil.ipipan.waw.pl/CoDeS?action=AttachFile&do=view&target=MSimLex999_Polish.zip), stworzonej przez IPIPAN.

Dokładniejszy opis jak i Obszerna tabelka z wynikami znajduje się w podanym w pierwszej linijce linku.

### Opis word embeddingów Sławomira Dadasa

Poniższe rozdziały zawierają pretrained word embeddings dla językach polskiego. Każdy model został wytrenowany
na korpusie składającym się z zasobów polskiej Wikipedii, polskich książek i artykułów, w sumie 1,5 miliarda tokenów.

#### Word2vec
Word2Vec został wytrenowany za pomocą biblioteki Gensim. 100 wymiarów, próbkowanie negatywne, zawiera lematyzowane słowa, 
które mają 3 lub więcej wystąpień w korpusie oraz dodatkowo zestaw predefiniowanych symboli interpunkcyjnych, 
wszystkie liczby od 0 do 10000, polskie imiona i nazwiska. 

 Przykład użycia:
```python
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word2vec = KeyedVectors.load("word2vec_polish.bin")
    print(word2vec.similar_by_word("bierut"))
    
# [('cyrankiewicz', 0.818274736404419), ('gomułka', 0.7967918515205383), ('raczkiewicz', 0.7757788896560669), ('jaruzelski', 0.7737460732460022), ('pużak', 0.7667238712310791)]
```

#### FastText

FastText trenowany za pomocą biblioteki Gensim. Słownictwo i wymiarowość są identyczne jak w modelu Word2Vec. 

Przykład użycia:

```python
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word2vec = KeyedVectors.load("fasttext_100_3_polish.bin")
    print(word2vec.similar_by_word("bierut"))
    
# [('bieruty', 0.9290274381637573), ('gierut', 0.8921363353729248), ('bieruta', 0.8906412124633789), ('bierutow', 0.8795544505119324), ('bierutowsko', 0.839280366897583)]
```

#### GloVe

Globalne wektory do reprezentacji słów (GloVe) trenowane przy użyciu implementacji referencyjnej ze Stanford NLP. 
100 wymiarów, zawiera lematyzowane słowa, które mają 3 lub więcej wystąpień w korpusie. 

Przykład użycia:

```python
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word2vec = KeyedVectors.load_word2vec_format("glove_100_3_polish.txt")
    print(word2vec.similar_by_word("bierut"))
    
# [('cyrankiewicz', 0.8335597515106201), ('gomułka', 0.7793121337890625), ('bieruta', 0.7118682861328125), ('jaruzelski', 0.6743760108947754), ('minc', 0.6692837476730347)]
```

#### Skompresowane Word2vec

Jest to skompresowana wersja opisanego powyżej modelu Word2Vec. Do kompresji wykorzystano metodę opisaną w 
[Compressing Word Embeddings via Deep Compositional Code Learning](https://arxiv.org/abs/1711.01068) autorstwa Shu
i Nakayamy. Skompresowane embeddingi 
nadają się do stosowania w urządzeniach o ograniczonej pamięci, takich jak telefony komórkowe. Model waży 38 MB, co 
stanowi zaledwie 4,4% rozmiaru oryginalnych embeddingów Word2Vec. Mimo że autorzy artykułu twierdzą, że kompresja ich
metodą nie wpływa na wydajność modelu, został zauważony niewielki, ale akceptowalny spadek dokładności, przy użyciu 
skompresowanej wersji embeddingów.

Przykładowa klasa dekodera wraz z użyciem:
```python
import gzip
from typing import Dict, Callable
import numpy as np

class CompressedEmbedding(object):

    def __init__(self, vocab_path: str, embedding_path: str, to_lowercase: bool=True):
        self.vocab_path: str = vocab_path
        self.embedding_path: str = embedding_path
        self.to_lower: bool = to_lowercase
        self.vocab: Dict[str, int] = self.__load_vocab(vocab_path)
        embedding = np.load(embedding_path)
        self.codes: np.ndarray = embedding[embedding.files[0]]
        self.codebook: np.ndarray = embedding[embedding.files[1]]
        self.m = self.codes.shape[1]
        self.k = int(self.codebook.shape[0] / self.m)
        self.dim: int = self.codebook.shape[1]

    def __load_vocab(self, vocab_path: str) -> Dict[str, int]:
        open_func: Callable = gzip.open if vocab_path.endswith(".gz") else open
        with open_func(vocab_path, "rt", encoding="utf-8") as input_file:
            return {line.strip():idx for idx, line in enumerate(input_file)}

    def vocab_vector(self, word: str):
        if word == "<pad>": return np.zeros(self.dim)
        val: str = word.lower() if self.to_lower else word
        index: int = self.vocab.get(val, self.vocab["<unk>"])
        codes = self.codes[index]
        code_indices = np.array([idx * self.k + offset for idx, offset in enumerate(np.nditer(codes))])
        return np.sum(self.codebook[code_indices], axis=0)

if __name__ == '__main__':
    word2vec = CompressedEmbedding("word2vec_100_3.vocab.gz", "word2vec_100_3.compressed.npz")
    print(word2vec.vocab_vector("bierut"))
```

Więcej informacji o word embeddingach możemy znaleźć o [tutaj](https://github.com/sdadas/polish-nlp-resources).










## Zbiory danych

[Github](https://github.com/Ermlab/pl-sentiment-analysis) zawierający zbiór Opineo - recenzje ze sklepów online.

[Zbiór](https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-pl.txt) zawiwerający pary słów a następnie skojarzone z nimi następne pary słów.

[Zbiór](https://opus.nlpl.eu/OpenSubtitles-v2018.php) zawierający polskie napisy do filmów. Z dwóch źródeł się dowiedziałem, że zawiera sporo powtórzeń.

[Zbiór](https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-pl.txt) zawierjający analogie ("Ateny Grecja Bagdad Irak")
, przydatny do word embedinggsów.
### Korpusy językowe

Korpus językowy to zbiór tekstów służących badaniom lingwistycznym, np. określaniu częstości występowania 
form wyrazowych, konstrukcji składniowych lub kontekstów, w jakich pojawiają się dane wyrazy.

#### Korpus NKJP
Zawiera polską literature klasyczną, gazety, transkrypty konwersacji, różne teksty z internetu, itp.
Istnieją dwie wyszukiwarki stworzone na jego bazie:

* IPI PAN

http://nkjp.pl/poliqarp/

Używany między innymi do Morfeusza 2.

[Ściągawka](http://nkjp.pl/poliqarp/help/pl.html) do używania korpusu. Znajdują się w niej między innymi zapytania o:
- segmenty
- formy podstawowe
- znaczniki morfosyntaktyczne
- wieloznaczność i dezambiguacja

* PELCRA

http://www.nkjp.uni.lodz.pl/

#### Korpus opisanych obrazów z adnotacjami

http://zil.ipipan.waw.pl/Scwad/AIDe

#### Korpus dyskursu parlamentarnego

ile tego jest, do czego uzyto, czy jest przydatny
https://kdp.nlp.ipipan.waw.pl/query_corpus/

#### Korpus dla smenatyki kompozycyjnej dystrybucyjnej

Składa się z 10 tys. polskich par zdań, które są opisane przez człowieka pod kątem pokrewieństwa semantycznego.

http://zil.ipipan.waw.pl/Scwad/CDSCorpus

#### Korpus polskich recenzji 

 Korpus polskich recenzji opatrzonych opisami na poziomie całego tekstu oraz na 
 poziomie zdań dla następujących dziedzin: hotele, medycyna, produkty i uniwersytet.

https://clarin-pl.eu/dspace/handle/11321/700

#### Korpus zawierający mowę nienawiści

Składa się z ponad 2000 postów scrapowanych z około 2000 postów z mediów społecznościowych.

http://zil.ipipan.waw.pl/HateSpeech



### Zbiory użyte do embeddingów

- https://clarin-pl.eu/dspace/handle/11321/442
- https://clarin-pl.eu/dspace/handle/11321/606
- https://clarin-pl.eu/dspace/handle/11321/600
- https://clarin-pl.eu/dspace/handle/11321/327
- http://vectors.nlpl.eu/repository 
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