import pandas as pd
import numpy as np
from scipy import stats

pd.set_option('display.width', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print(df.head(10))

# Transformação Logaritimica
df['salario_log'] = np.log1p(df['salario'])  # log1p é usado para evitar problemas com valores zero


print("\nDataFrame após transformação logaritimica no 'salario':\n", df.head(10))

# Transformação Box-Cox
df['salario_boxcox'], _ = stats.boxcox(df['salario'] + 1)

print("\nDataFrame com transformação BoxCox no 'salario':\n", df.head(10))

# Codificação de Frequência para 'estado'
estado_freq = df['estado'].value_counts() / len(df)
df['estado_freq'] = df['estado'].map(estado_freq)

print("\nDataFrame após condição de frequência para o 'estado':\n", df.head(10))

# Interações
df['interacao_idade_filhos'] = df['idade'] * df['numero_filhos']


print("\nDataFrame após criação de interações 'idade' e 'numero_filhos':\n", df.head(10))
