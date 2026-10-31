"""
Генерация данных эксперимента exp_data_seed_{MY_SEED}.csv.

Основа — исходный код из задания (см. generate_data_original.py).
Отличия от исходника (задание разрешает менять параметры рандомизации):

1. Seed можно передать аргументом командной строки, чтобы файл можно было
   воспроизвести:  python generate_data.py 1234567
   Без аргумента seed выбирается случайно, как в исходнике.
2. Peeking. В исходном коде variant дважды перезаписывается, поэтому в итоге
   entry_time никак не связано с группой и нарушение 1 в данных не появляется.
   Здесь после финального распределения по группам часть test-пользователей
   (доля LATE_SHARE выбирается случайно) получает позднее время входа
   из интервала [0.6, 1). Так в тесте оказываются систематически более
   поздние пользователи — след подглядывания и остановки теста по времени.
3. Файл сохраняется в корень репозитория, в сообщении печатается реальный путь.

Остальная логика (SRM, confounder weekend, ложный сигнал в platform=2,
завышенный error_flag в test) не изменена.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----- Seed -----
if len(sys.argv) > 1:
    MY_SEED = int(sys.argv[1])
else:
    MY_SEED = np.random.randint(5000000, size=1)[0]
np.random.seed(MY_SEED)

# ----- Параметры -----
n_users = np.random.randint(1000, 10000, size=1)[0]
true_effect = 0  # истинного эффекта фичи нет, но в данных есть артефакты
expected_ratio = 0.5

# ----- НАРУШЕНИЕ 1: Peeking (часть 1 — время входа) -----
time_of_entry = np.random.uniform(0, 1, n_users)  # время от 0 до 1 (в днях)
variant = np.where(time_of_entry > 0.4, 'test', 'control')  # перезаписывается ниже

# ----- SRM: неравные группы -----
p_ = np.random.uniform(low=0.4001, high=0.599999, size=1)[0]
variant = np.random.choice(['control', 'test'], size=n_users, p=[p_, 1 - p_])

# ----- НАРУШЕНИЕ 2: confounder weekend -----
weekend = np.random.binomial(1, 0.3, n_users)
p_ = np.random.uniform(low=0.40001, high=0.499999, size=1)[0]
variant = np.where((variant == 'test') & (weekend == 1), 'test',
                   np.where((variant == 'control') & (weekend == 0), 'control',
                            np.random.choice(['control', 'test'], size=n_users, p=[p_, 1 - p_])))

# Конверсия зависит только от weekend (истинный эффект фичи = 0)
conversion_prob = 0.1 + 0.05 * weekend
converted = np.random.binomial(1, conversion_prob, n_users)

# ----- НАРУШЕНИЕ 3: сегменты и ложный сигнал -----
p_ = np.random.uniform(low=0.02, high=0.15, size=1)[0]
p_2 = np.random.uniform(low=0.25, high=0.35, size=1)[0]
platform = np.random.choice([0, 1, 2], size=n_users, p=[1 - p_2 - p_, p_2, p_])
converted = np.where((platform == 2) & (variant == 'test'),
                     np.random.binomial(1, 0.2, size=len(converted)),
                     converted)

# ----- Guardrail: в test ошибки завышены -----
error_flag = np.zeros(n_users)
error_flag[variant == 'control'] = np.random.binomial(1, 0.02, sum(variant == 'control'))
error_flag[variant == 'test'] = np.random.binomial(1, 0.06, sum(variant == 'test'))

# ----- НАРУШЕНИЕ 1: Peeking (часть 2 — привязка времени к группе) -----
# Доля test-пользователей, «добранных» в конце теста.
LATE_SHARE = np.random.uniform(low=0.15, high=0.30, size=1)[0]
late_mask = (variant == 'test') & (np.random.uniform(0, 1, n_users) < LATE_SHARE)
time_of_entry = np.where(late_mask,
                         np.random.uniform(0.6, 1.0, n_users),
                         time_of_entry)

# ----- Финальный датафрейм -----
df = pd.DataFrame({
    'user_id': range(n_users),
    'variant': variant,
    'converted': converted,
    'error_flag': error_flag,
    'weekend': weekend,
    'platform': platform,
    'entry_time': time_of_entry,
})

out_path = Path(__file__).resolve().parent.parent / f'exp_data_seed_{MY_SEED}.csv'
df.to_csv(out_path, index=False)

print(f"MY_SEED = {MY_SEED}")
print(f"Данные сохранены в файл {out_path.name}")
print(f"Размеры групп: control={sum(variant == 'control')}, test={sum(variant == 'test')}")
