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

### RoBERTa

### HerBERT



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
