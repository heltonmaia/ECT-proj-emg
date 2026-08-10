"""Template for the regenerated prediction.py at the project root.

Placeholders (in curly-brace string-format style) are filled in by train.py:
  {HEADER}                  - auto-generated metadata comment
  {NUM_AMOSTRAS}            - window size in samples
  {FEATURE_FUNCTIONS}       - bodies of the feature funcs used by the model
  {PREDICT_SIGNATURE}       - prever_movimento(rms, mav, ...) — kwargs only
  {PREDICT_BODY}            - the decision tree as nested if/else (indented 4 spaces)
  {FEATURE_COMPUTE_BLOCK}   - lines that compute the feature kwargs from the window
  {PREDICT_KWARGS}          - call args used at the prediction site
"""

TEMPLATE = '''\
{HEADER}
from machine import ADC, Pin
import time
import math

# ---------- Parametros ----------
NUM_AMOSTRAS = {NUM_AMOSTRAS}
TAMANHO_FILTRO = 5

# ---------- Inicializacao ----------
adc = ADC(26)

# ---------- Feature functions ----------
{FEATURE_FUNCTIONS}

# ---------- Decision tree (auto-generated) ----------
def prever_movimento({PREDICT_SIGNATURE}):
{PREDICT_BODY}

# ---------- Funcoes auxiliares ----------
def media_movel(dados, tamanho):
    filtrado = []
    for i in range(len(dados)):
        janela = dados[max(0, i - tamanho + 1):i + 1]
        filtrado.append(sum(janela) / len(janela))
    return filtrado

# ---------- Loop principal ----------
print("Iniciando leitura EMG com filtro digital...")

while True:
    janela = []
    for _ in range(NUM_AMOSTRAS):
        leitura = adc.read_u16() >> 4   # 12-bit (0-4095)
        janela.append(leitura * 10)     # amplification (matches original code)
        time.sleep(1/500)
    filtrado = media_movel(janela, TAMANHO_FILTRO)

{FEATURE_COMPUTE_BLOCK}

    classe = prever_movimento({PREDICT_KWARGS})
    if classe == 1:
        print("Mao FECHADA")
    else:
        print("Mao ABERTA")
    time.sleep(0.1)
'''
