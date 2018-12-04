# Домашнее задание № 5

## Случайный лес

В этом задании вам нужно будет реализовать метод случайного леса (random forest). Научным заданием будет определение красного смещения до галактик по их фотометрическим измерениям.

**Дедлайн 17 декабря 23:55**

Вы должны реализовать следующие алгоритмы:

1. Метод решающего дерева для решения задачи регрессии в файле `tree.py`
2. Метод random forest в файле `forest.py`, используя ваше дерево из `tree.py`

В файле `galaxies.py` загрузите данные `sdss_redshift.csv` с известными красными смещениями (колонка `redshift`) и на данных фотометрии (колонки `u`, `g`, `r`, `i`, `z`) обучите ваш лес. Подберите такие фичи и гиперпараметры, чтобы достигнуть наименьшего стандартного отклонения предсказания от истинного значения. Нарисуйте график истинное значение — предсказание со всеми вашеми подвыборками в файл `redhift.png`. В файл `redhsift.json` выведите стандартное отклонение для всех ваших подвыборок.

Загрузите данные из файла `sdss.csv` с неизвестными красными смещениями и используя обученный лес предскажите значения красных смещений. Результат сохраните в файл `sdss_predict.csv` в том же формате, что и `sdss_redshift.csv`.

![Красное смещение](./IMAGE%202018-12-04%2010:35:17.jpg)

**Справка**

Связь красных смещений и фотометрического расстояния см. в ДЗ 2. Фотометрическое расстояние d связано с зведной величиной m, данной вам в таблице как m ~ 5\*log(d). Подробнее о звездных величинах см [астронет](http://www.astronet.ru/db/msg/1174337)
