# Etapa 06 — Frontend e experiência do usuário Tech Layoffs

Data: 2026-11-26. Base: commit `7f0d0ec` (etapa 05, 145 testes). Implementação com apoio do
**Claude Code** (Anthropic); revisão e execução pelo grupo.

## 1. Objetivos

Interface profissional "Tech Layoffs" em HTML/CSS/JS puro servida pelo Flask, sem frameworks;
indicadores reais do dataset; chat com etapas e fontes reais; gráficos só a partir de resultados
estruturados do Python; estados de erro; responsividade e acessibilidade; Streamlit adaptado;
duas correções pequenas no agente (§17 do enunciado).

## 2. Análise da interface atual

`static/index.html` (96 linhas, CSS e JS inline): título "Assistente de Aluguéis", boas-vindas e
placeholder de aluguel, `<input>` de uma linha, etapas como texto "🔧 a · b", sem fontes, sem estado de
carregamento além do botão desabilitado, histórico sem limite de tamanho (pode passar de 64 KB),
"Nova conversa" não descarta respostas pendentes, `caixa.innerHTML = ""` (seguro, mas a página usa
`textContent` para o conteúdo — mantido).

## 3. Identidade visual

Paleta do enunciado em variáveis CSS (`--color-primary #142432`, `--color-accent #267D70`, fundo
`#F7F8FA`, superfície branca, texto `#142432`, `--color-muted #64748B`, borda `#E2E8F0`) + cores de
estado (aviso/erro) sempre acompanhadas de texto/ícone. Tipografia do sistema (sem fontes externas),
números tabulares nos indicadores, raio 6–8 px, sombras mínimas, sem gradientes, sem emojis na
identidade. Marca: monograma SVG simples ("TL" + barras) também usado como favicon.
Contraste: accent sobre branco ≈ 4,9:1; muted sobre branco ≈ 4,8:1 (AA para texto normal).

## 4. Estrutura de componentes

```text
static/index.html        estrutura semântica (header, section de indicadores, main, aside, footer)
static/css/app.css       variáveis, layout, componentes, responsivo, foco, movimento reduzido
static/js/app.js         estado, API, renderização segura (textContent), etapas, fontes, gráficos
static/assets/favicon.svg
```
Componentes: cabeçalho (marca, descrição, estado do servidor e do Gemini, "Nova conversa");
faixa de indicadores; chat (boas-vindas + sugestões, mensagens, indicador de processamento,
erros com "Tentar novamente"); composer (textarea, contador, Enter envia / Shift+Enter quebra linha);
painel lateral (demissões por ano, modelo e aviso de escopo); rodapé acadêmico.

## 5. Desktop e mobile

- ≥ 1100 px: chat central + painel lateral; indicadores em 5 colunas.
- 700–1099 px: sem painel lateral; indicadores em 5 colunas compactas.
- < 700 px: indicadores em grade 2×3 compacta; mensagens em largura total; composer fixo no fim;
  altura com `100dvh` (fallback `100vh`) para o teclado virtual; sem rolagem horizontal.

## 6. Contrato da API

Mantido: `POST /api/chat` `{mensagem, historico}` → `{resposta, etapas, fontes}`.
Extensões **opcionais e retrocompatíveis**:
- `GET /api/dataset/resumo` (somente leitura, calculado de `ferramentas.DF`/`METRICAS`): eventos,
  demitidos, empresas, países, período, demissões por ano, modelo e MAE. Não expõe linhas do dataset.
- `/api/chat` pode devolver `dados` = resultado estruturado de `consultar_dados()` (só quando a
  consulta foi executada sem erro), para gráficos. Os números do gráfico nunca vêm do texto do Gemini.

## 7. Estados

inicial (boas-vindas + sugestões) · enviando (mensagem preservada, botão e sugestões desabilitados,
indicador discreto, `aria-busy`) · resposta · erro de conexão · timeout (55 s, AbortController) ·
429 (limite) · 503 (Gemini indisponível) · 4xx · nova conversa (aborta a requisição e descarta a
resposta antiga via identificador de conversa). Estado do Gemini: "não verificado" até uma resposta
real; "respondeu" após sucesso; "indisponível" após 503.

## 8. Acessibilidade

HTML semântico; `<label>` no campo; `role="log"` + `aria-live="polite"` nas mensagens; erros com
`role="alert"`; foco visível (`:focus-visible`); navegação por teclado (sugestões e detalhes são
`<button>`/`<details>`); estados com texto + ícone (não só cor); `prefers-reduced-motion`; textos
alternativos nos gráficos (figcaption + rótulos com valores).

## 9. Segurança

Todo conteúdo da API via `textContent`/`createElement` (sem `innerHTML` com dado externo); chave só no
servidor; nada em `localStorage`; histórico limitado a 20 itens × 2.000 caracteres e < 56 KB;
endpoint de resumo somente leitura e agregado.

## 10. Checklist

- [x] Ler planos, interface, `app.py`, `agente.py`, Streamlit; confirmar contrato
- [x] Criar este plano
- [x] Correções do agente (§17): `pergunta_interpretada` fora das evidências; resposta segura de previsão com alertas
- [x] `GET /api/dataset/resumo` e campo opcional `dados`
- [x] `static/index.html`, `css/app.css`, `js/app.js`, `assets/favicon.svg`
- [x] Streamlit adaptado
- [x] Testes novos (endpoints, estáticos, segurança do JS) + suíte completa + scripts do kit
- [x] Validação visual headless (desktop e mobile) com agente simulado
- [x] README e este plano

## 11. Testes

`unittest`: rota `/`, arquivos estáticos, `/api/dataset/resumo` (valores vs. CSV original), campo `dados`
retrocompatível, correções do agente. Verificação estática do JS (sem `innerHTML`/`eval` com dados
externos, sem `localStorage`, sem chave). Validação visual com `chromium-browser --headless
--screenshot` em servidor local com o agente simulado (sem Gemini).

## 12. Critérios de conclusão

Os do enunciado (§22).

## 13. Resultados

### Testes — `.venv/bin/python -m unittest discover -s tests -v`
**162 testes OK** = 145 anteriores (sem alteração) + `tests/test_frontend.py` 17:
- página e arquivos estáticos;
- ausência de "aluguel";
- sugestões;
- indicadores não fixos no código;
- `/api/dataset/resumo` comparado com o CSV original e sem expor linhas;
- `dados` opcional (retrocompatível);
- JS sem `innerHTML`/`eval`/`localStorage` e HTML sem handlers inline;
- limites do JS iguais aos do backend;
- todas as etapas do agente traduzidas no JS;
- HTML semântico e regras de CSS;
- as duas correções do agente.

`testar_sem_api.py` e `conferir.py`: exit 0. Hashes de `dados/` e `modelo/` inalterados.
Um teste novo falhou na primeira execução: "innerHTML" aparecia só no **comentário** do cabeçalho do JS. O teste
passou a ignorar comentários (e confere que o código continua presente).

### Validação visual (chromium-browser 158, headless)
Servidor de validação no scratchpad (fora do repositório): `app.py` real com `agente.responder` substituído
por respostas fixas que usam dados reais de `consultar_dados()`; roteiro de cliques injetado só na cópia do
HTML servida em `/visual`. Resoluções: 1440×900, 820×1100, 390×844, e 1000×3400 para a conversa inteira.
Problemas encontrados e corrigidos:
1. faixa vazia de ~150 px no topo: `<svg hidden>` do sprite é exibido no Chromium → classe `.sprite` (largura e altura 0);
2. no mobile, a faixa empurrava o campo de mensagem para fora da tela (resolvido junto com o item 1);
3. botão "Nova conversa" quebrava em 2 linhas no mobile → `white-space: nowrap`;
4. R² exibido como 0,55 → 4 casas (0,5537);
5. placeholder cortado no mobile → texto mais curto.

Confirmado no navegador:
- gráficos de ranking e de comparação;
- selos de aviso ("Verificação com aviso", "Machine Learning" com imputação);
- etapas expansíveis;
- erro 503 com "Tentar novamente" e estado "Gemini: indisponível";
- **XSS:** uma resposta com `<img onerror>` foi exibida como texto, sem nenhum `<img>` no DOM e sem mudar o título;
- **Nova conversa durante uma requisição:** botão travado durante o envio, requisição abortada (1), 0 mensagens depois, boas-vindas visível e botão liberado.

Não validado: navegadores Safari/Firefox, teclado virtual real de celular, leitor de tela real.

### Streamlit
Adaptado e revisado pela sintaxe; **não executado**, porque `streamlit` não está instalado no `.venv`
(não foi instalado, conforme a restrição de dependências).

### Gemini real
**Não executado** nesta etapa (sem autorização para consumir a cota). Toda a validação usou o agente simulado.

## 14. Limitações

- Os gráficos aparecem só para resultados agrupados ou comparações vindos de `consultar_dados()`; previsões não têm gráfico.
- A contagem de países (42) trata UK e United Kingdom como grafias separadas; isso está explicado no tooltip do indicador.
- O estado do Gemini só fica verificável depois de uma pergunta real; não há verificação ativa (seria gasto de cota).
- O painel lateral fica oculto abaixo de 1100 px; os indicadores principais continuam visíveis.

## 15. Pendências para a etapa 07

1. Validação real com o Gemini (roteiro do README) e registro das respostas.
2. Bateria de testes do professor (12 categorias), `comparar_sem_rag.py --executar` e relatório.
3. Testar a interface em celular real, Firefox e Safari, e com leitor de tela.
4. Hospedagem (Python 3.14.10 × `.python-version` 3.13; `requirements` do Streamlit).
