import numpy as np

import pandas as pd


# СГЕНЕРИРУЙТЕ СВОЙ SEED 

MY_SEED = np.random.randint(5000000, size=1)[0]

np.random.seed(MY_SEED)


# Параметры

n_users = np.random.randint(1000, 10000, size=1)[0]

true_effect = 0  # на самом деле эффекта нет, но данные будут содержать артефакты

expected_ratio = 0.5


# ----- НАРУШЕНИЕ 1: Peeking (ранняя остановка) -----

# Имитируем, что тест останавливали каждые 1000 пользователей, 

# и если p-value становился <0.05, то останавливались. 

# В результате в финальной выборке дисбаланс по времени регистрации.

# Для простоты: генерируем время входа в эксперимент так, 

# чтобы в тестовой группе были более поздние пользователи (что создаёт временной тренд)

time_of_entry = np.random.uniform(0, 1, n_users)  # время от 0 до 1

# В test группе делаем сдвиг во времени (позднее включение)

variant = np.where(time_of_entry > 0.4, 'test', 'control')  # не 50/50


# Попробуем сымитировать  SRM: сделаем неравными группы (например, 55/45)

# Переопределим variant с нужной пропорцией

p_ = np.random.uniform(low=0.4001, high=0.599999, size=1)[0]

variant = np.random.choice(['control', 'test'], size=n_users, p=[p_, 1-p_])


# ----- НАРУШЕНИЕ 2: confounder (погода/день недели влияет и на участие, и на результат) -----

# Допустим, есть скрытая переменная "weekend" (0/1), которая увеличивает конверсию

weekend = np.random.binomial(1, 0.3, n_users)

# В тестовой группе больше weekend-пользователей

p_ = np.random.uniform(low=0.40001, high=0.499999, size=1)[0]

variant = np.where((variant == 'test') & (weekend == 1), 'test', 
                   np.where((variant == 'control') & (weekend == 0), 'control', 
                            np.random.choice(['control', 'test'], size=n_users, p=[p_, 1-p_])))  # оставляем некоторую случайность


# Конверсия зависит от weekend (истинный эффект фичи = 0)

conversion_prob = 0.1 + 0.05 * weekend

converted = np.random.binomial(1, conversion_prob, n_users)


#  ----- НАРУШЕНИЕ 3: множественные проверки и сегментация -----

# Добавим несколько сегментов (платформа: 0,1,2)

p_ = np.random.uniform(low=0.02, high=0.15, size=1)[0]

p_2 = np.random.uniform(low=0.25, high=0.35, size=1)[0]

platform = np.random.choice([0,1,2], size=n_users, p=[1-p_2-p_,p_2,p_])

# В сегменте platform=2 тестовая группа получила искусственный подъём конверсии (ложный сигнал)

converted = np.where((platform == 2) & (variant == 'test'), 
                     np.random.binomial(1, 0.2, size=len(converted)), 
                     converted)


# ----- Guardrail: частота ошибок (должна быть одинаковой, но в test завышена) -----

error_flag = np.zeros(n_users)

error_flag[variant == 'control'] = np.random.binomial(1, 0.02, sum(variant == 'control'))

error_flag[variant == 'test'] = np.random.binomial(1, 0.06, sum(variant == 'test'))


# Финальный датафрейм

df = pd.DataFrame({
    'user_id': range(n_users),
    'variant': variant,
    'converted': converted,
    'error_flag': error_flag,
    'weekend': weekend,
    'platform': platform,
    'entry_time': time_of_entry
})


PATH = 'exp_data/'


df.to_csv(f'exp_data_seed_{MY_SEED}.csv', index=False)

print(f"Данные сохранены в файл {PATH}exp_data_seed_{MY_SEED}.csv")

print(f"Размеры групп: control={sum(variant=='control')}, test={sum(variant=='test')}")
