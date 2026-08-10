# `tcc/treinamento/` — pipeline de treino do classificador EMG (500 Hz)

Pipeline reproduzível para o classificador binário (mão aberta / mão
fechada) descrito no TCC. Carrega as 10 gravações em `rec_emg/`, aplica
filtragem offline, fatia em janelas rotuladas, extrai features, roda um
sweep de 30 configs com Leave-One-Group-Out CV, escolhe a vencedora,
treina o modelo final, salva artefatos e regenera `prediction.py` na
raiz do repositório.

**Spec do design**: `docs/superpowers/specs/2026-05-23-classificador-500hz-design.md`

## Como rodar

```bash
source ../../.venv/bin/activate
python tcc/treinamento/train.py
```

Demora ~1-3 minutos. Saídas em `results/` + `dt_500hz.pkl` + `prediction.py` (na raiz).

## Decisões de projeto

### Dados
- Origem: `rec_emg/new_emg_data{1..10}.csv` — 10 CSVs, todos a 500 Hz, 30 s cada.
- Protocolo de prompts: 0s aberta → 5s fecha → 10s abre → 15s fecha → 20s abre → 25s fecha.

### Filtragem (offline, zero-phase, scipy)
- Notch 60 Hz, Q=30, via `filtfilt`.
- Passa-faixa Butterworth 4ª ordem, 20–240 Hz, via `sosfiltfilt`.
- **Nota**: o filtro no `prediction.py` da Pico é só média móvel de 5 amostras. Mismatch herdado do TCC — fica como trabalho futuro (quando o pipeline mover pro ESP32-C5 o filtro on-device deve passar a casar com o offline).

### Janelamento
- Tamanho: 100 amostras = 200 ms (= `NUM_AMOSTRAS` do `prediction.py`).
- Sobreposição: 50% (step = 50 amostras).
- Margem de transição: 500 ms (= 250 amostras) descartados em cada lado de uma mudança de rótulo.
- ~244 janelas por CSV × 10 CSVs ≈ 2440 janelas totais.

### Features (`features.py`)
- `rms`, `mav`, `sd`, `wl`, `var`, `zc`, `ssc`, `wamp`.
- **Crítico**: `mav` ≠ `mean`. O `prediction.py` antigo usava a média aritmética simples como uma das features; aqui usamos a definição correta `MAV = mean(|x|)`. Testes em `test_features.py` cobrem esse caso explicitamente.
- Thresholds de `zc`, `ssc`, `wamp`: começam em 0 (sem deadzone).
- Cada feature tem variante escalar (`features.py`) e vetorizada (`data.py`, `VECTORIZED_FEATURE_FUNCS`) — `extract_features` usa a vetorizada (~38× mais rápida). Os testes garantem paridade entre as duas.

### Sweep
- 6 feature sets × 5 valores de `max_depth` (3, 5, 7, 10, None) = 30 configs.
- Cada uma roda 10-fold Leave-One-Group-Out CV (cada CSV é um group).
- Selector: maior `mean_acc` no LOGO, tiebreak por `std`, `n_features`, `max_depth`.

### Modelo final
- Re-treinado no dataset inteiro com a config vencedora.
- Salvo em `dt_500hz.pkl` (joblib, payload: `{model, features, fs}`).
- Exportado como if/else inline para `prediction.py` na raiz.

### Regeneração do `prediction.py`
- Template em `prediction_template.py` (placeholders `{HEADER}`, `{NUM_AMOSTRAS}`, etc).
- Esqueleto MicroPython (ADC, loop, prints) preservado entre regenerações.
- Bloco modelo (features + árvore) regerado a cada `python train.py`.
- Header com timestamp + accuracy + features pra auditoria visual.

## Estrutura

```
tcc/treinamento/
├── README.md                       (este arquivo)
├── features.py                     funções puras de feature (escalar)
├── filter.py                       design do filtro EMG (notch + bandpass)
├── data.py                         loader + janelamento + extração vetorizada
├── train.py                        orchestrator + sweep + figuras + export
├── prediction_template.py          template do prediction.py
├── conftest.py                     ajuste de sys.path pra pytest
├── test_features.py                testes unitários das features
├── test_filter.py                  testes do filtro
├── test_data.py                    testes de loader + janelas + extract_features
├── dt_500hz.pkl                    modelo final (gerado por train.py)
└── results/                        gerados por train.py
    ├── metrics.txt                 top 5 configs + winner + relatório + confusão
    ├── confusion_matrix.{png,svg}  da config vencedora
    ├── decision_tree.{png,svg}     visualização da árvore
    └── feature_importance.{png,svg} importância de cada feature
```

## Rodar testes

```bash
pytest tcc/treinamento/
```

`conftest.py` injeta o diretório no `sys.path`, então roda de qualquer cwd.
