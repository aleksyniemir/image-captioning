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

Model stworzony przez Google, na jego podstawie stworzono najlepsze polskie modele. Architektura BERT polega na nauce poprzez usuwanie ze zdania losowo kilka słów, i następnie model ma się nauczyć jak najlepiej wypełnić te dziury. Przy odpowiednio dużym korpusie z czasem  model coraz lepiej poznaje zależności semantyczne między słowami.


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
### HerBERT

### Model językowy autorstwa Teresy Sas

Model jest zbudowany na zasobach tekstów o tematyce akademickiej i ogólnej, obejmując prawie 800 tyś. słów. W linku możemy pobrać zipa w którym znajdują się trzy pliki:
•	model_2d_forward.txt - model bigramowy zawierający prawdopodobieństwa następstwa słów p(w_i | w_{i-1} ) dla porządku od lewej do prawej
•	model 3d_bakward.txt - model trigramowy zawierający prawdopodobieństwa p( w_i | w_{i+1} w_{i+2} ) dla porządku odwróconego
•	word_list.txt - lista słów występujących w modelu

Model może być wykorzystany w badaniach nad rozpoznawaniem mowy i w inżynierii języka naturalnego dla języka polskiego.


## Korpusy językowe

Korpus językowy to zbiór tekstów służących badaniom lingwistycznym, np. określaniu częstości występowania form wyrazowych, konstrukcji składniowych lub kontekstów, w jakich pojawiają się dane wyrazy.

Korpus który rozpatrzyłem został stworzony w projekcie autorstwa NKJP. Istnieją dwie wyszukiwarki stworzone na jego bazie:

### [IPI PAN](http://nkjp.pl/poliqarp/)

[Ściągawka](http://nkjp.pl/poliqarp/help/pl.html) do używania korpusu. Znajdują się w niej między innymi zapytania o:
- segmenty
- formy podstawowe
- znaczniki morfosyntaktyczne
- wieloznaczność i dezambiguacja

###	[PELCRA](http://www.nkjp.uni.lodz.pl/)

## Word embeddings

[Ewaluacja polskich embeddingów](https://github.com/Ermlab/polish-word-embeddings-review) stworzona przez wiele grup badawczych.

W powyższej implementacji została użyta biblioteka „gensim”, która pozwala wczytać dane, wytrenować model, oraz końcowo sprawdzić wyniki. Oprócz tworzenia wektorowych przedstawień słów biblioteka ma wiele innych funkcji, między innymi może znajdować semantycznie podobne dokumenty do tego który został jej zadany. Żeby określić word-embedinngi w powyższej implementacji używana jest funkcja "evaluation_word_pairs" oraz "assessment_word_analogies" z biblioteki gensim.

Do testowania został użyty plik dostarczony przez [facebook fastext](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.vec.gz). Żeby określić podobieństwo słów użyto polskiej wersji [SimLex999](http://zil.ipipan.waw.pl/CoDeS?action=AttachFile&do=view&target=MSimLex999_Polish.zip), stworzonej przez IPIPAN.

Obszerna tabelka z wynikami znajduje się w podanym w pierwszej linijce linku.


## Zbiory danych

[Zbiór](https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-pl.txt) zawiwerający pary słów a następnie skojarzone z nimi następne pary słów.

## Pre-Trained modele

### Modele użyte do embeddingów

- https://clarin-pl.eu/dspace/handle/11321/442
- https://clarin-pl.eu/dspace/handle/11321/606
- https://clarin-pl.eu/dspace/handle/11321/600
- https://clarin-pl.eu/dspace/handle/11321/327
- http://vectors.nlpl.eu/repository (id: 62,167)
- https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz
