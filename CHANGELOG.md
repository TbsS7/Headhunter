# Changelog

## 0.12.0 (2026-11-13)

**Breaking: the whole API is now in English.** No numbers changed: on tiny MHA and GQA models, every public function returns the same values as v0.11.5 (1,537 values compared, 0 differences apart from wall-clock time). The Portuguese version stays available as tag `v0.11.5`, and `experiments/` targets it.

- Functions, parameters, constants, result keys and task names were renamed (tables below). Task `COPIA` is now `COPY`, `COPIA_EMB` is `COPY_EMB`, `COPIA_NOVA` is `COPY_NEW`; `IOI`, `CAP`, `GT`, `MCQA` are unchanged.
- Comments, docstrings and printed messages are in English.
- Added a GitHub Actions workflow that runs the CPU tests.

Main result keys: `cabecas` -> `heads`, `nucleo` -> `core`, `circuito` -> `circuit`, `fidelidade` -> `faithfulness`, `arestas` -> `edges`, `tempo` -> `elapsed`, `atingiu` -> `reached`, `alvo` -> `target`, `errado` -> `wrong`.

<details><summary>Renamed public functions</summary>

| v0.11 | v0.12 |
|---|---|
| `carregar` | `load_model` |
| `preparar` | `prepare` |
| `adaptador` | `adapter` |
| `montar_ioi` | `build_ioi` |
| `paises_conhecidos` | `known_countries` |
| `montar_capitais` | `build_capitals` |
| `metrica` | `metric` |
| `tarefa` | `get_task` |
| `logits_finais` | `final_logits` |
| `exato_cabecas_lento` | `exact_heads_slow` |
| `exato_cabecas` | `exact_heads` |
| `exato_mlps` | `exact_mlps` |
| `nucleo_economico` | `economic_core` |
| `fidelidade` | `faithfulness` |
| `fidelidade_lotes` | `faithfulness_batches` |
| `circuito_por_alvo` | `circuit_for_target` |
| `autoteste` | `selftest` |
| `regressao` | `regression` |
| `salvar` | `save` |
| `ordem_cabecas` | `head_order` |
| `arestas_entre` | `edges_between` |
| `arestas_partes` | `edge_parts` |
| `gradiente_entradas` | `input_gradients` |
| `notas_arestas` | `edge_scores` |
| `fidelidade_arestas` | `edge_faithfulness` |
| `fidelidade_arestas_lotes` | `edge_faithfulness_batches` |
| `aresta_exata` | `exact_edge` |
| `busca_arestas` | `edge_search` |
| `podar_arestas` | `prune_edges` |
| `gradiente_nos` | `node_gradient` |
| `promover_degraus` | `promote_steps` |
| `analisar` | `analyze` |
| `ruido_busca` | `search_noise` |
| `palavras_copia` | `copy_words` |
| `montar_copia` | `build_copy` |
| `gradiente_nos_ig` | `node_gradient_ig` |
| `ordem_cabecas_ig` | `head_order_ig` |
| `montar_copia_emb` | `build_copy_emb` |
| `montar_copia_nova` | `build_copy_new` |
| `bateria` | `corruption_battery` |
| `faixa_circuito` | `circuit_range` |
| `placar_ioi_gpt2` | `scoreboard_ioi_gpt2` |
| `notas_ig` | `ig_scores` |
| `circuito_com_mlps` | `circuit_with_mlps` |
| `carregar_sae_gpt2` | `load_sae_gpt2` |
| `features_por_camada` | `features_per_layer` |
| `necessidade` | `necessity` |
| `necessidade_x_suficiencia` | `necessity_vs_sufficiency` |
| `validar_circuito` | `validate_circuit` |
| `montar_ioi_alt` | `build_ioi_alt` |
| `montar_capitais_alt` | `build_capitals_alt` |
| `montar_copia_alt` | `build_copy_alt` |
| `acerto` | `accuracy` |
| `generalizacao` | `generalization` |
| `montar_gt` | `build_gt` |
| `montar_mcqa` | `build_mcqa` |
| `tarefa_nova` | `new_task` |

</details>

<details><summary>Renamed constants</summary>

| v0.11 | v0.12 |
|---|---|
| `VERSAO` | `VERSION` |
| `ESQUEMA_IG` | `IG_SCHEME` |
| `GRANDES` | `LARGE_MODELS` |
| `MODERNOS` | `MODERN_FAMILIES` |
| `NOMES` | `NAMES` |
| `MODELOS_IOI` | `IOI_MODELS` |
| `PAISES` | `COUNTRIES` |
| `MOLDE` | `TEMPLATE` |
| `RECEITAS` | `RECIPES` |
| `REFERENCIA` | `REFERENCE` |
| `LOTE_CALCULO` | `COMPUTE_BATCH` |
| `BLOCO_EXATO` | `EXACT_BLOCK` |
| `LOTE_GRAD` | `GRAD_BATCH` |
| `PALAVRAS_COPIA` | `COPY_WORDS` |
| `L_COPIA` | `COPY_LEN` |
| `PERGUNTAS` | `QUESTIONS` |
| `GABARITO_IOI_GPT2` | `GOLD_IOI_GPT2` |
| `MODELOS_IOI_ALT` | `IOI_MODELS_ALT` |
| `MOLDES_CAP_ALT` | `CAP_TEMPLATES_ALT` |
| `SUBST_GT` | `GT_NOUNS` |
| `CATEGORIAS_MCQA` | `MCQA_CATEGORIES` |
| `GABARITO_GT_GPT2` | `GOLD_GT_GPT2` |

</details>

Parameter names follow the same translation (for example `alvo` -> `target`, `alvo_poda` -> `prune_target`, `podar` -> `prune`, `grupos` -> `groups`, `sementes_ordem` -> `order_seeds`, `verbose` unchanged).

## 0.11.5 (2026-11-12)

Fused compute chunk (v0.11.4) never exceeds its memory target and cleans up after an out-of-memory error outside the `except` block. See `docs/RESULTS.md` for the cost measurements.
