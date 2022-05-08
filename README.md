# image-captioning w języku polskim

Na tym githubie znajduję się opis wybranych zagadnień używanych w image captioning. Lista zagadnień:
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

Lematyzacja oznacza sprowadzenie grupy wyrazów stanowiących odmianę danego zwrotu do wspólnej postaci, umożliwiając traktowanie ich wszystkich jako to samo słowo. W przetwarzaniu języka naturalnego odgrywa rolę ujednoznaczenia, jako że 

## Modele językowe 

Model propabilistyczny, którego celem jest obliczenie prawdopodobieństwa wystąpienia kolejnego słowa po zadanej wcześniej sekwencji słów. Na podstawie modelu językowego możemy oszacować następne najbardziej prawdopodobne słowo.

Na samym początku wspomnę o rankingu klej - który jest polskim odpowiednikiem angielskiego rankingu GLUE. Został stworzony przez Allegro, i określa ranking najlepszych modeli językowych dla językach polskiego. Rankking możemy znaleźć o [tutaj](https://klejbenchmark.com/leaderboard/)

Do tej pory pionierem do spraw języka naturalnego w wielu językach był googlowski BERT, jednak w klasyfikacji GLUE dostał tylko 80 na 100 punktów. Na podstawie BERT’a powstał Polski RoBERT, który osiągnął 87/100 punktów w teście KLEJ, a jeszcze potem na podstawie RoBERTA powstał HerBERT, osiągając najwyższą do tej pory notę 88/100.

## Korpusy językowe

Korpus który rozpatrzyłem został stworzony w projekcie autorstwa NKJP. Istnieją dwie wyszukiwarki stworzone na jego bazie:

- [IPI PAN](http://nkjp.pl/poliqarp/)

-	[PELCRA](http://www.nkjp.uni.lodz.pl/)

##
