# Задание в команду Alignment

RLHF [1] — фундаментальный метод алаймента языковых моделей, который применялся при обучении ChatGPT, LLama 2, Qwen 2.5 и т.д. Как известно RLHF очень требователен к ресурсам и чувствителен к гиперпараметрам. Сложность данного метода обоснована сложностью алгоритма PPO [2], который лежит в основе RLHF.

Статья Back to Basics [3] предлагает использовать более простой алгоритм REINFORCE для алаймента языковых моделей. Согласно результатам данной работы, данный алгоритм не только проще в реализации, но и показывает лучшие метрики, чем PPO.

## Level 1

Мы предлагаем вам реализовать алгоритм REINFORCE w/ baseline для алаймента языковых моделей. Обращаем ваше внимание, что вам не нужно реализовывать RLOO для подсчёта baseline, достаточно взять moving average.

### Шаги выполнения:

1. **SFT модель**: Используйте `HuggingFaceTB/SmolLM2-135M-Instruct` в качестве SFT модели.

2. **Reward Model**: Поверх SFT обучите Reward Model на парах из датасета `esfrankel17/HelpSteer2_binarized`.
   - Разбейте `average_rting_split` на train и validation подвыборки
   - Рекомендуется использовать `RewardTrainer` из библиотеки `trl`
   - Достаточно обучить одну эпоху
   - Используйте learning rate = 5e-5, fp16
   - Ограничьте максимальную длину так, чтобы избежать Out of Memory
   - Рекомендуем куда-нибудь сохранить полученную RM

3. **REINFORCE**: Реализуйте алгоритм REINFORCE w/ baseline из статьи, используя SFT и RM, полученные на предыдущих шагах.
   - НЕ рекомендуется использовать `RLOOTrainer` из библиотеки `trl`, так как реализация в нём отличается от алгоритма, описанного в статье
   - **Bonus**: в отчёте опишите почему именно
   - Используйте batch size и количество итераций, которые позволяют ваши ресурсы
   - В качестве baseline используйте moving average

**Вопрос**: Выросла ли средняя награда на отложенной выборке (validation split) по сравнению c SFT моделью?

## Level 2

Предположим, что мы хотим обучить Reward Model, которая выдаёт не скалярную оценку, а распределение вероятности поверх дискретных оценок. Пусть оценка текста — натуральное число от 1 до 10, тогда RM выдаёт 10 чисел, каждое из которых — вероятность текста получить соответствующую оценку (соответственно сумма этих значений равна 1).

### Задачи:

1. **Функция потерь**: Подумайте, как должна выглядеть функция потерь для обучения такой модели наград, если мы по-прежнему хотим максимизировать вероятность $p(y_w \succ y_l | x)$.

2. **Обучение RM**: Обучите Reward Model с полученной функцией потерь на том же датасете пар.

3. **Интеграция в REINFORCE**:
   - Придумайте, как данную модель интегрировать в алгоритм REINFORCE
   - Какую дополнительную информацию мы можем использовать, если работаем с распределением над оценками?
   - Как этот сигнал интерпретируется?

4. **Обучение**: Обучите REINFORCE поверх SFT с вероятностной RM. Используйте те же гиперпараметры, что использовали при обучении в Level 1.

**Вопросы**:
- Получилось ли улучшить качество алаймента?
- Как вы объясните полученный результат?

## Правила

1. **Анализ результатов**: Проанализируйте полученные результаты. Это самый важный пункт, потому что хочется увидеть не только числа с полученными метриками. Как вы объясняете увиденное поведение? Напишите отчет о проведенных экспериментах. Что получилось? Что нет?

2. **Исследовательский подход**: Нет правильного способа решить задачу. Не стоит беспокоиться, что вы делаете что-то неправильно. Мы хотим увидеть ваши способности к исследованиям, а не какое-то конкретное решение задачи. Возможно, вы придумаете то, о чем мы даже не задумывались изначально – это будет высший класс.

3. **Достоверность**: Убедитесь, что результатам можно доверять. Исключите вариант случайности, etc.

4. **Ресурсы**: Вы можете использовать Google Colab или Kaggle Code, чтобы получить доступ к бесплатным вычислительным ресурсам.

5. **Оформление**:
   - Присылайте решение в виде репозитория на github с отчетом по решению и чёткими инструкциями, как запустить ваш код
   - Убедитесь, что мы сможем запустить ваше решение по этим инструкциям
   - Вы можете использовать любые библиотеки и фреймворки, которые вам могут быть необходимы

6. **Качество кода**: Сфокусируйтесь на том, чтобы код был чист и понятен. Если вы считаете, что какая-то его часть может быть непонятна, то добавьте комментарии. Мы очень сильно ценим хорошо написанный код, поэтому если решение задачи будет оформлено грязно, то мы можем отклонить заявку.

## References

[1] Ouyang et al, Training language models to follow instructions with human feedback

[2] Schulman et al, Proximal Policy Optimization Algorithms

[3] Ahmadian et al, Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs
