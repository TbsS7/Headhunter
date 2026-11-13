# HeadHunter

*English: [README.md](README.md)*

O HeadHunter encontra um **circuito suficiente** num transformer: as cabeças de atenção, os MLPs e as arestas que, mantidos sozinhos, reproduzem uma tarefa (IOI, capitais, cópia de sequência, Greater-Than, múltipla escolha). Ele usa o gradiente barato (EAP-IG) em tudo e gasta o patching exato só onde a aproximação comprovadamente falha. O circuito é **validado em frases novas**, que não foram usadas para montá-lo.

> **Status: código de pesquisa, versão 0.12.0.** Um relatório (arXiv / Alignment Forum) está em preparação. A v0.12 passou toda a API para inglês; a versão antiga, em português, continua na tag `v0.11.5`.

## Instalar

```bash
pip install git+https://github.com/TbsS7/Headhunter.git
```

Ou a partir de um clone: `pip install -e .` e `python -m pytest tests -q` (testes na CPU com modelos aleatórios pequenos, ~10 s, sem internet nem GPU). Requer `transformer_lens < 4` (testado com a 3.9). O HeadHunter roda em **float32**.

## Uso rápido

```python
import headhunter as hh
m = hh.load_model("gpt2-small")          # ou um id do Hugging Face, ex. "Qwen/Qwen2.5-0.5B"
hh.selftest(m, "IOI")
r = hh.analyze(m, "IOI")                # núcleo, circuito até 90% de fidelidade, arestas
print(r["circuit"]["heads"])           # índice da cabeça = camada * n_cabeças + cabeça
v = hh.validate_circuit(m, "IOI", r)   # frases novas, circuitos aleatórios, só MLPs
```

## Resultados e limites

Números completos em [docs/RESULTS.md](docs/RESULTS.md) (em inglês). Em resumo: o núcleo de cabeças não cresce com o modelo (IOI: 11 a 18 cabeças de 0,5B a 14B); os circuitos mantêm ~90% de fidelidade em frases novas; contra o EAP-IG oficial as notas são idênticas em float32 e a qualidade empata. No Qwen2.5-1.5B levamos 0,34-0,52× do tempo do oficial em float32 (0,58-0,88× do bfloat16) com 0,8-0,9× da memória (bfloat16: 1,4-1,6×); no 0,5B somos mais lentos. Só dois tamanhos foram comparados; no 7B só o HeadHunter foi medido. Os circuitos são suficientes, mas **não provados mínimos**; o ACDC oficial e o Pythia-12B não foram rodados.

## Licença

MIT.
