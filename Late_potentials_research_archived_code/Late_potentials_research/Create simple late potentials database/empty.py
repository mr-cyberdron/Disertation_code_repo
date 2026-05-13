import pandas as pd
import numpy as np

# создаем датафрейм
df = pd.DataFrame({'A': [1, 1, np.nan, 4, 5], 'B': [6, np.nan, 8, 10, 10]})

# заменяем пропущенные значения в столбце A на медиану этого столбца
median_A = df['A'].median()
df['A'] = df['A'].fillna(median_A)

print(df)