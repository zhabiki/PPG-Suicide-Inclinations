# *Исследовательский проект «Классификация состояния людей с суицидальными наклонностями»*

Этот репозиторий содержит реализацию программного продукта на языке Python с использованием машинного обучения и нейронных сетей, предназначенного для выявления и оценки риска наличия суициальных наклонностей у человека, на основе вероятностной оценки наличия у него связанных с этим риском психических расстройств, по различным параметрам, получаемым из его фотоплетизмограммы (ФПГ).


- С кодом для снятия собственных "кустарных" данных (которые мы объединяем с открытыми данными из исследований), а также методикой их получения, можно ознакомиться здесь: https://github.com/zhabiki/PPG-Dataset-DIY
- Ознакомиться же с самими исследованиями по теме (в том числе с открытыми и/или используемыми нами данными) и выводами из них, а также рассмотренными ранее датасетами, можно здесь: https://docs.google.com/document/d/1nuOdxSKehN-uu5WzF-iZv0fn3jvBSY7twHKKJxZaAmY


## Структура:

- [WIP] `__main__.py` — Полное решение, объединяющее и последовательно использующее одну из моделей, препроцессор и данные ФПГ для обучения конкретно взятой модели-классификатора.
- `model_*.py` — Модели различных подвидов (например, разные версии или для разных расстройств), обучаемые на и классифицирующие обработанные через препроцессор данные.
- `preprocess.py` — Класс препроцессора для данных, проходящийся по (желательно предварительно отфильтрованным) данным ФПГ и извлекающий из них параметры вариабельности сердечного ритма (ВСР), [WIP] LF/HF, их соотношение и RSA, и усреднённые расстояния шагов сердечного цикла.
- `preprocess_small.py` — Упрощённая версия препроцессора, извлекающая только ЧСС и SDNN.
- `examples/*` — Несколько файлов-примеров "сырых" данных ФПГ здоровых людей в спокойном состоянии.
- [WIP] `requirements.txt` — Все требуемые модули для запуска продукта.


## Запуск: 

[ WIP - Obviosuly blocked by `__main__.py` implementation, duh ]


## Каким образом работает наш код?

### 31.03.2025 01:40

Наша нейронная сеть состоит из нескольких полносвязных моделей внутри, каждая из которых обучена определять своё психиатрическое состояние (депрессия, треожность, биполярное расстройство, шизофрения и пр.).

На вход каждой модели подаётся четыре признака - ЧСС, ВСР (LF/HF соотношение), SDNN и RSA. Выдав один логит, нейросеть прогоняет его через сигмоиду для получения итоговой вероятности наличия расстройства личности.

Однако встроенный обработчик сигналов изучает не весь сигнал за раз, а проходит по нему скользящим окном, каждый раз прогоняя новые результаты через модели. Таким образом мы получаем списки вероятностей для каждого расстройства, которые затем можно усреднить.

После усреднения все вероятности подаются в последнюю иодель, которая предсказывает логит и точно такой же сигмоидой вычисляет вероятность наличия суицидальных наклонностей у пациента.

Для работы с сигналами мы используем **pandas** (если данные табличные), **scipy** (фильтрация), **pytorch** (сама модель) и **heartpy** (извлечение признаков, параметров ЭКГ и ФПГ в определённый промежуток времени)


### 03.04.2025 22:48

**Инициализация** 

Удалено: слой свёртки, self.fc и self.sigmoid. 

Добавлено: словарь моделей, ключи которого - названия болезней, передающихся новым параметром diseases.

**Метод forward**

Полностью изменён алгоритм. Он адаптирован под словарь моделей.

**Добавлен метод set_mode**
Он принимает на вход параметр mode, который может принимать только значения 'train' или 'test', от него зависит, в каком режиме будет работать нейронная сеть.

Были проведены тесты на маленьком "табличном" датасете из исследования на тему MDD. Результаты:

Loss для депрессии: 0.6931471824645996

Loss для контрольной группы: 0.31326165795326233


### 12.04.2025 13:40

**Пересмотр модели классификации**

Была пересмотрена модель классификации. Теперь вместо нейронных сетей мы используем классические методы машинного обучения.

**Методы обучения**

Наша модель представляет ансамбль из одинаковых методов. Среди методов присутствуют: LDA, QDA, линейная регрессия, наивный байесовский классификатор, SVM, дерево решений, xgboost, adaboost, градиентный бустинг.

Среди всех методов лучше себя показали градиентный бустинг и adaboost.

**forward**

Метод, отвечающий за классификацию. В зависимости от режима train/test занимается обучением или классификацией заболеваний. Обучение происходит по принципу "один против всех", когда модель определяет целевое состояние как 1, а остальные как 0. Таким образом параметры с соответствующей меткой подаются на каждую модель, и они учатся классифицировать не только своё состояние, но и отличать его от других. Тестовый режим, в зависимости от наличия в параметре "diseases" списка меток, может как выдавать предсказания меток так и выдавать точность работы ансамбля, если в параметре diseases есть заранее подготовленные метки.

**set_mode**

Метод делает то же, что и в предыдущих архитектурах - определяет режим работы моделей.

**Добавлен `preprocess_small.py`**

Файл содержит обработчик сигналов PreprocessSignals, альтернативный PreprocessPPG из `preprocess.py` обработчик, который имеет более упрощённую структуру и не определяет пиков, а просто проходит по сигналу скользящим окном, извлекая ключевые признаки кроме LF/HF составляющей.

Класс содержит только функцию извлечения параметров с помощью скользящего окна и встроенный фильтр Баттерворта четвёртого порядка.


### 13.04.2025 20:35

**Добавлены комментарии и строки-документации**

Добавлены комментарии и строки документации к методам в файлах `preprocess_small.py` и `model_ready-demo3.py`.
