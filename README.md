# Genetic_Algorithm_Parking_System

Projekt ma na celu opracowanie algorytmów parkowania pojazdów czterokołowych za pomocą podejścia genetycznego. Synteza zaprojektowanego układu sterującego przeprowadzana jest na podstawie wielokryterialnej optymalizacji. 

## Wybór środowiska
Pierwszym etapem procesu opracowywania algorytmu genetycznego sterowania pojazdem czterokołowym jest wybór właściwego środowiska oraz języka programowania. Na potrzeby tego projektu wybrane zostały: środowisko PyCharm 2019.3.4 Community Edition oraz język programowania wysokiego poziomu Python w wersji 3.7. Taki dobór narzędzi uwarunkowany został przede wszystkim dużymi możliwościami w postaci bardzo rozbudowanego pakietu bibliotek pozwalającego między innymi na szczegółową wizualizację oraz analizę graficzną otrzymanych rezultatów. Wykorzystano moduły standardowe takie jak Math oraz Random. Biblioteka Numpy pozwoliła na obsługę dużych, wielowymiarowych tablic oraz macierzy wraz z równie pokaźnym zbiorem funkcji matematycznych pozwalających na efektywną pracę przy ich przetwarzaniu. Wizualizacja w projekcie została natomiast zrealizowana z wykorzystaniem biblioteki Matplotlib.

## Wymagania i ograniczenia 
Kolejnym krokiem prowadzącym do poprawnej implementacji algorytmu genetycznego jest określenie założeń projektowych dla zadanego problemu, ograniczeń dla parametrów oraz kryteriów optymalizacji. Zgodnie ze schematem poniżej założeniami w problemie parkowania pojazdu czterokołowego są: skrętna oś przednia oraz nieskrętna oś tylna w pojeździe, możliwość poruszania się pojazdu do przodu lub do tyłu wyłącznie ze stałą prędkością równą 1 metr na 1 krok sterowania. 

<img src="https://raw.githubusercontent.com/MacKur/Genetic_Algorithm_Parking_System/main/Problem_parkowania.PNG">

Kolejno wykonywane kroki objęte są ograniczeniami nałożonymi na współrzędne położenia środka pojazdu, w przypadku zderzenia auta z jedną z granic parkingu, automatycznie funkcja przystosowania dla danej sekwencji ruchów objęta zostaje karą, która powoduje usunięcie wadliwego sterowania z przestrzeni dostępnych rozwiązań. Dodatkowo parametry będące składowymi rozwiązań algorytmu, czyli kąt skrętu kół oraz kierunek poruszania się pojazdu V mogą przyjmować wartości tylko i wyłącznie należące do z góry określonych zbiorów. Kąt skrętu kół przyjmuje wartości z przedziału (-PI/6, PI/6) natomiast parametr V wartość z przedziału {-1, 1}. Problem parkowania pojazdu czterokołowego rozpatrywany jest jako problem minimalizacji dwóch kryteriów: odległości w linii prostej środka pojazdu oraz miejsca parkingowego oraz kąta alfa odchylenia osi pojazdu od osi miejsca parkingowego. Manewr parkowania składający się z szeregu sterowań, uznany zostaje za udany w przypadku gdy odległość pomiędzy środkiem pojazdu oraz miejsca parkingowego jest mniejsza niż zadana wartość 0.7 metra przy jednoczesnym zachowaniu kąta alfa w przedziale (-10 stopni, +10 stopni). 

Mając przygotowane ograniczenia przestrzeni poszukiwań oraz kryteria optymalizacji, kolejną czynnością jest stworzenie modelu definiującego sposób poruszania się pojazdu, który określi zależności występujące pomiędzy parametrami rozwiązań, a kryteriami optymalizacji. W tym przypadku model zaimplementowany w języku programowania Python został zainspirowany modelem ze strony laboratorium sztucznej inteligencji prowadzonym na wydziale ETI politechniki Gdańskiej [1]. Model przedstawia się następująco:

<img src="https://raw.githubusercontent.com/MacKur/Genetic_Algorithm_Parking_System/main/model.PNG">

## Opis i implementacja
Problem optymalizacyjny do rozwiązania jest zadaniem minimalizacji dwukryterialnej, gdzie dla obu funkcji przystosowania danego rozwiązania osiągnięcie wartości na poziomie równym 0 traktowane jest równoznacznie z bezbłędnym zaparkowaniem pojazdu w miejscu parkingowym (pod względem pozycji środka pojazdu i kąta odchylenia osi pojazdu względem osi miejsca parkingowego). Rysunek zamieszczony poniżej przedstawia kolejne kroki wykonywane przez algorytm w celu osiągnięcia wspomnianego minimum. W projekcie wybrany został algorytm NSGA-II wzbogacony dodatkowo o mechanizm wyszukiwania lokalnego (z ang. local search algorithm). 

<img src="https://raw.githubusercontent.com/MacKur/Genetic_Algorithm_Parking_System/main/Schemat_implementacji.PNG">

Populacja składa się z określonej przez użytkownika liczby N osobników będących sekwencjami sterowań zawierających w sobie informacje o kierunku ruchu oraz kącie skrętu kół podczas jego wykonywania. Tak skonstruowane osobniki poddawane są operacjom genetycznym. Do algorytmu wybrane zostały krzyżowanie jednopunktowe oraz mutacja fenotypowa. Jako dodatek w celu usprawnienia algorytmu dodany został mechanizm wyszukiwania lokalnego w względem drugiego z parametrów, czyli kąta skrętu kół. Wszystkie powyżej wymienione operacje skutkują tworzeniem mniejszych populacji potomnych o z góry określonych wielkościach. Po ich wykonaniu algorytm sumuje populację bazową z populacją rodziców w wyniku czego otrzymuje populację składającą się z 1,5 N rozwiązań, dla których następnie zgodnie z modelem przedstawionym na rysunku powyżej obliczane są funkcje przystosowania. Następnie wykorzystując sortowanie niezdominowane oraz mechanizm zatłoczenia przeprowadza się selekcję turniejową pomiędzy osobnikami przypisując im przynależność do poszczególnych frontów. W ten sposób usuwany jest nadmiar osobników, które według algorytmu nie przybliżały go do osiągnięcia pożądanego minimum. Po wykonaniu wszystkich wymienionych czynności sprawdzane jest czy którykolwiek z osobników spełnia kryterium zatrzymania algorytmu. Jeśli zastosowanie sterowania któregoś z nich pozwala na poprawne zaparkowanie w miejscu parkingowym, algorytm wizualizuje na pierwszym wykresie przystosowania całej populacji w której pojawił się pierwszy osobnik spełniający warunek zatrzymania, dzieląc ją na osobniki zdominowane w sensie Pareto oraz niezdominowane w tym właśnie wspomniane rozwiązanie optymalne. Drugi wykres obrazuje natomiast trajektorię przejazdu samochodu zanim trafił na miejsce parkingowe. Natomiast na trzecim wykresie obserwować można zmiany parametru GOL dla najlepszego osobnika w każdej iteracji, w przypadku nie spełnienia warunku zatrzymania algorytm wraca do kroku, w którym przeprowadzane są operacje genetyczne.

## Rezultaty
<img src="https://raw.githubusercontent.com/MacKur/Genetic_Algorithm_Parking_System/main/Konsola.png">
               Konsola <br />

<img src="https://raw.githubusercontent.com/MacKur/Genetic_Algorithm_Parking_System/main/Wykres%201.png" width="600" height="400">
               Wykres nr 1 <br />


<img src="https://raw.githubusercontent.com/MacKur/Genetic_Algorithm_Parking_System/main/Wykres%202.png" width="600" height="400">
               Wykres nr 2 <br />


<img src="https://raw.githubusercontent.com/MacKur/Genetic_Algorithm_Parking_System/main/Wykres%203.png" width="600" height="400">
               Wykres nr 3 <br />

[1] http://si-lab.cba.pl/SystemyRozmyte2015.html (20.12.2020) Laboratorium Sztuczna Inteligencja
