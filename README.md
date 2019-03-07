# nlp-course-mipt 2019
Курс по анализу текстов

Преподаватели: 
* Екатерина Артемова
* Емельянов Антон (login-const@mail.ru)

## Правила.
* Домашние задания сдаются в anytask. Invite в телеграмм канале курса.

## Домашние задания
### 1. Named Entity Recognition
Определение именованных сущностей с помощью архитектуры SENNA.
* [Contest](https://www.kaggle.com/c/mipt-ner) на kaggle.
* Условие [git](hws/hw1.ipynb)
* Посылать в anytask.

## Занятия
### Занятие 5. Языковые модели
Материалы [тут](class05-LM/)

Полезные ссылки
* [LSTM explained](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [The Illustrated BERT, ELMo, and co.](http://jalammar.github.io/illustrated-bert/)
* [Transformer-XL explained](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html)

Ссылки на реализации языковых моделей/примеры использования:
* ELMO: tf - [ru](http://docs.deeppavlov.ai/en/master/apiref/models/embedders.html#deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder), [en](https://tfhub.dev/google/elmo/2); pytorch - [for many languages](https://github.com/HIT-SCIR/ELMoForManyLangs/), [paper](https://arxiv.org/pdf/1802.05365.pdf)
* ULMFit: pytorch - [en](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb), [ru, тайга](https://github.com/mamamot/Russian-ULMFit/), [ru wiki](https://github.com/ppleskov/Russian-Language-Model), [paper](https://arxiv.org/pdf/1801.06146.pdf)
* Flair: pytorch - [en chars](https://github.com/zalandoresearch/flair), [paper](http://aclweb.org/anthology/C18-1139)
* BERT: tf - [multilingual](https://github.com/google-research/bert), pytorch - [multilingual](https://github.com/huggingface/pytorch-pretrained-BERT), [paper](https://arxiv.org/pdf/1810.04805.pdf)
* Transformer-XL: tf, pytorch - [en](https://github.com/kimiyoung/transformer-xl), [paper](https://arxiv.org/pdf/1901.02860.pdf), [explanation](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html)
* OpenAI GPT-2: [en](https://github.com/openai/gpt-2), [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)


## Рекомендуемые ресурсы
### На английском

* [Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/)
* [Курс Лауры Каллмайер по МО для NLP](https://user.phil.hhu.de/~kallmeyer/MachineLearning/index.html)
* [Курс Нильса Раймерса по DL для NLP](https://github.com/UKPLab/deeplearning4nlp-tutorial)
* [Курс в Оксфорде по DL для NLP](https://github.com/UKPLab/deeplearning4nlp-tutorial)
* [Курс в Стенфорде по DL для NLP](http://cs224d.stanford.edu)
* [Reinforcment Learning for NLP](https://github.com/jiyfeng/rl4nlp)


### На русском (и про русский, в основном)

* [Курс nlp в яндексе](https://github.com/yandexdataschool/nlp_course)
* [НКРЯ](http://ruscorpora.ru)
* [Открытый корпус](http://opencorpora.org)
* [Дистрибутивные семантические модели для русского языка](http://rusvectores.org/ru/)
* [Морфология](https://tech.yandex.ru/mystem/)
* [Синтаксис](https://habrahabr.ru/post/317564/)
* [Томита-парсер](https://tech.yandex.ru/tomita/)
* [mathlingvo](http://mathlingvo.ru)
* [nlpub](https://nlpub.ru)
* [Text Visualisation browser](http://textvis.lnu.se)



## Литература

* Manning, Christopher D., and Hinrich Schütze. Foundations of statistical natural language processing. Vol. 999. Cambridge: MIT press, 1999.
* Martin, James H., and Daniel Jurafsky. "Speech and language processing." International Edition 710 (2000): 25.
* Cohen, Shay. "Bayesian analysis in natural language processing." Synthesis Lectures on Human Language Technologies 9, no. 2 (2016): 1-274.
* Goldberg, Yoav. "Neural Network Methods for Natural Language Processing." Synthesis Lectures on Human Language Technologies 10, no. 1 (2017): 1-309, [URL](https://github.com/shucunt/book/blob/master/2017%20-%20Neural%20Network%20Methods%20for%20Natural%20Language%20Processing.pdf).