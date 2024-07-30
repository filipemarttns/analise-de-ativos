import yfinance as fy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import requests
from bs4 import BeautifulSoup

ativo = 'VIVA3.SA'
inicio_previsao = '2023-12-01'
fim_previsao = '2024-07-28'

data = fy.download(ativo, start=inicio_previsao, end=fim_previsao)
data.dropna(inplace=True)

escala = MinMaxScaler()
data_escala = escala.fit_transform(data)

data['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
data['SMA_200'] = data['Close'].rolling(window=200, min_periods=1).mean()

X = data[['Open', 'High', 'Low', 'Close']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_regressao = LinearRegression()
modelo_regressao.fit(X_train, y_train)

y_pred = modelo_regressao.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

ultimo_preco = data['Close'].iloc[-1]
media_movel_50 = data['SMA_50'].iloc[-1]
media_movel_200 = data['SMA_200'].iloc[-1]

tendencia_curto_prazo = "positiva" if ultimo_preco > media_movel_50 else "negativa" if ultimo_preco < media_movel_50 else "estável"
tendencia_longo_prazo = "positiva" if ultimo_preco > media_movel_200 else "negativa" if ultimo_preco < media_movel_200 else "estável"

returns = data['Close'].pct_change()
volatilidade = returns.std() * np.sqrt(252)

preco_hoje = data['Close'].iloc[-1]
preco_ontem = data['Close'].iloc[-2]
variacao_percentual = ((preco_hoje - preco_ontem) / preco_ontem) * 100

correlacao = data['Close'].corr(data['Volume'])

cor_preco_fechamento = '#1f77b4'
cor_media_50_dias = '#ff7f0e'
cor_media_200_dias = '#2ca02c'

plt.style.use('dark_background')

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(data.index, data['Close'], label='Preço de Fechamento', color=cor_preco_fechamento, linewidth=1.5)
ax1.plot(data.index, data['SMA_50'], label='Média Móvel de 50 dias', color=cor_media_50_dias, linewidth=1.5)
ax1.plot(data.index, data['SMA_200'], label='Média Móvel de 200 dias', color=cor_media_200_dias, linewidth=1.5)

ax1.set_title('Médias Móveis de 50 e 200 dias', fontsize=16, fontweight='bold', color='white')
ax1.set_xlabel('Data', fontsize=12, color='white')
ax1.set_ylabel('Preço', fontsize=12, color='white')
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

for ax in [ax1]:
    ax.tick_params(axis='x', labelrotation=45, labelsize=10, colors='white')
    ax.tick_params(axis='y', labelsize=10, colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

plt.tight_layout()
plt.show()

print('')

print("Análise de Tendências:")

print("Tendência de curto prazo:", tendencia_curto_prazo)

print("Tendência de longo prazo:", tendencia_longo_prazo)

print("\nVolatilidade Histórica da Ação:", volatilidade)

print("\nVariação Percentual entre o Preço de Fechamento de {} ao dia anterior:".format(fim_previsao), variacao_percentual)

print("\nCorrelação entre o Preço de Fechamento e o Volume de Negociação:", correlacao)

print("\nDesempenho do Modelo de Regressão Linear:")

print("Erro Médio Quadrático (MSE):", mse)

print("Coeficiente de Determinação (R²):", r2)

url = "https://finance.yahoo.com/quote/{}/key-statistics?p={}".format(ativo, ativo)
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

metricas = soup.find_all('td', class_='Fz(s) Fw(500) Ta(end) Pstart(10px) Miw(60px)')
metricas_valores = soup.find_all('td', class_='Fz(s) Ta(end) Pstart(10px) Miw(60px)')

info_financeira = {}
for i in range(len(metricas)):
    info_financeira[metricas[i].text] = metricas_valores[i].text

print("\nAnálise Fundamentalista:")
for metrica, valor in info_financeira.items():
    print(metrica + ":", valor)

preco_por_lucro = float(info_financeira.get('Trailing P/E') or 0)
print("Razão P/L (Preço/Lucro):", preco_por_lucro)

margem_lucro = float(info_financeira.get('Profit Margin') or 0)
print("Margem de Lucro Líquido (%):", margem_lucro)

roe = float(info_financeira.get('Return on Equity') or 0)
print("ROE (Retorno sobre o Patrimônio Líquido) (%):", roe)
