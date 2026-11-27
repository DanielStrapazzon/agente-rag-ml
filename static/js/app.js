/* Tech Layoffs — lógica da interface (etapa 06). JavaScript puro, sem dependências.
 *
 * Contrato: POST /api/chat {mensagem, historico} -> {resposta, etapas, fontes, dados?}
 *           GET  /api/dataset/resumo                -> indicadores agregados do dataset real
 * Segurança: todo conteúdo vindo da API é inserido com textContent / createElement (nunca innerHTML).
 * Nada é guardado em localStorage; a chave do Gemini existe só no servidor.
 * Implementado com apoio do Claude Code (Anthropic).
 */
"use strict";

(() => {
  const LIMITES = {
    caracteres: 1000,          // igual a MAX_CARACTERES do app.py
    itensHistorico: 20,        // igual a MAX_HISTORICO do agente.py
    caracteresItem: 2000,      // igual a MAX_TEXTO_HISTORICO do agente.py
    bytesCorpo: 56000,         // abaixo do limite de 64 KB do Flask
    timeoutMs: 55000,
  };

  const $ = (id) => document.getElementById(id);
  const ui = {
    rolagem: $("rolagem"), mensagens: $("mensagens"), boasVindas: $("boas-vindas"),
    form: $("form"), entrada: $("entrada"), enviar: $("enviar"), nova: $("nova"),
    contador: $("contador"), anuncio: $("anuncio"),
    estadoServidor: $("estado-servidor"), estadoGemini: $("estado-gemini"),
  };

  const estado = { historico: [], conversa: 0, controlador: null, enviando: false };

  const numero = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 2 });
  const fmt = (v) => (typeof v === "number" && Number.isFinite(v) ? numero.format(v) : String(v ?? "—"));
  const reduzirMovimento = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- utilidades de DOM (seguras)
  function el(tag, classe, texto) {
    const n = document.createElement(tag);
    if (classe) n.className = classe;
    if (texto !== undefined && texto !== null) n.textContent = String(texto);
    return n;
  }

  function icone(nome, classe = "icone") {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", classe);
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", "#i-" + nome);
    svg.appendChild(use);
    return svg;
  }

  function anunciar(texto) {
    ui.anuncio.textContent = "";
    window.setTimeout(() => { ui.anuncio.textContent = texto; }, 30);
  }

  function rolarParaFim() {
    ui.rolagem.scrollTo({ top: ui.rolagem.scrollHeight, behavior: reduzirMovimento ? "auto" : "smooth" });
  }

  function definirEstado(item, situacao, texto) {
    item.dataset.estado = situacao;
    item.querySelector(".estado-texto").textContent = texto;
  }

  // ---------------------------------------------------------------- etapas e fontes
  // Traduz as etapas REAIS devolvidas pela API (agente.py). Nada é exibido se a API não informou.
  function classificarEtapa(texto) {
    const [chave, ...resto] = String(texto).split(":");
    const detalhe = resto.join(":").trim();
    const k = chave.trim().toLowerCase();
    if (k === "interpretação") {
      return { nome: "Interpretação", detalhe, status: detalhe.includes("JSON inválido") ? "aviso" : "ok" };
    }
    if (k === "rag") return { nome: "RAG", detalhe, status: detalhe.startsWith("0 ") ? "info" : "ok" };
    if (k === "consulta") {
      if (detalhe === "nenhuma") return { nome: "Consulta estatística", detalhe: "não executada", status: "info" };
      return { nome: "Consulta estatística", detalhe, status: detalhe.endsWith("(erro)") ? "erro" : "ok" };
    }
    if (k === "faltam") return { nome: "Dados pendentes", detalhe, status: "aviso" };
    if (k === "modelo") {
      const aviso = /extrapola|imputa|inconsistente/.test(detalhe);
      return { nome: "Machine Learning", detalhe, status: aviso ? "aviso" : "ok" };
    }
    if (k === "validação") {
      if (detalhe.startsWith("casos semelhantes")) {
        return { nome: "Casos semelhantes", detalhe, status: detalhe.includes("fallback") ? "aviso" : "ok" };
      }
      return { nome: "Validação", detalhe, status: detalhe.includes("erro") ? "erro" : "ok" };
    }
    if (k === "geração") return { nome: "Geração", detalhe: "Gemini", status: "ok" };
    if (k === "verificação") {
      const status = detalhe === "ok" ? "ok" : detalhe === "refeita" ? "aviso"
        : detalhe.includes("determinística") ? "aviso" : "erro";
      const textos = {
        "ok": "números conferidos no contexto",
        "refeita": "resposta refeita após números não conferidos",
        "resposta determinística": "resposta montada pelo sistema (a redação não passou na verificação)",
        "números não conferidos": "alguns números não foram conferidos",
        "resposta vazia": "o Gemini não devolveu texto",
      };
      return { nome: "Verificação", detalhe: textos[detalhe] || detalhe, status };
    }
    if (k === "esclarecimento solicitado") return { nome: "Esclarecimento", detalhe: "pergunta ambígua", status: "info" };
    return { nome: chave.trim(), detalhe, status: "info" };
  }

  const ICONE_STATUS = { ok: "ok", aviso: "aviso", erro: "erro", info: "info" };
  const TEXTO_STATUS = { ok: "concluída", aviso: "com aviso", erro: "com erro", info: "informação" };

  const NOMES_FONTES = {
    "consultar_dados (dataset)": "Dataset Tech Layoffs (consulta)",
    "modelo MLP": "Modelo MLP",
    "fatos_do_dataset (calculados do CSV)": "Fatos calculados do dataset",
    "conhecimento/descricao_dataset.md": "Descrição do dataset",
    "conhecimento/analise_exploratoria.md": "Análise exploratória",
    "conhecimento/modelo_metricas.md": "Métricas do modelo",
    "conhecimento/limitacoes.md": "Limitações",
  };

  function selos(etapas) {
    const lista = el("div", "selos");
    const principais = etapas.filter((e) => ["Consulta estatística", "Machine Learning", "Verificação", "Dados pendentes",
      "Validação", "Esclarecimento"].includes(e.nome) && !(e.nome === "Consulta estatística" && e.status === "info")
      || (e.nome === "Casos semelhantes" && e.status === "aviso"));
    for (const e of principais) {
      const s = el("span", "selo");
      s.dataset.status = e.status;
      s.appendChild(icone(ICONE_STATUS[e.status]));
      const rotulo = e.nome === "Verificação"
        ? (e.status === "ok" ? "Números verificados" : e.status === "aviso" ? "Verificação com aviso" : "Verificação falhou")
        : e.nome;
      s.appendChild(document.createTextNode(rotulo));
      s.title = e.detalhe;
      lista.appendChild(s);
    }
    return principais.length ? lista : null;
  }

  function blocoEtapas(etapas) {
    const det = el("details", "detalhes");
    det.appendChild(el("summary", null, `Etapas executadas (${etapas.length})`));
    const ol = el("ol", "etapas");
    for (const e of etapas) {
      const li = el("li", "etapa");
      li.dataset.status = e.status;
      li.appendChild(icone(ICONE_STATUS[e.status]));
      li.appendChild(el("span", "etapa-nome", e.nome));
      const d = el("span", "etapa-detalhe", e.detalhe || "—");
      d.appendChild(el("span", "sr-only", ` (${TEXTO_STATUS[e.status]})`));
      li.appendChild(d);
      ol.appendChild(li);
    }
    det.appendChild(ol);
    return det;
  }

  function blocoFontes(fontes) {
    const div = el("div", "fontes");
    div.appendChild(el("span", "fontes-rotulo", "Fontes:"));
    for (const f of fontes) {
      const item = el("span", "fonte");
      item.appendChild(icone("fonte"));
      item.appendChild(document.createTextNode(NOMES_FONTES[f] || f));
      item.title = f;
      div.appendChild(item);
    }
    return div;
  }

  // ---------------------------------------------------------------- gráfico (só dados estruturados do Python)
  function graficoBarras(titulo, itens, nota) {
    const validos = itens.filter((i) => typeof i.valor === "number" && Number.isFinite(i.valor) && i.valor >= 0);
    if (validos.length < 2) return null;
    const maximo = Math.max(...validos.map((i) => i.valor)) || 1;
    const fig = el("figure", "grafico");
    fig.appendChild(el("figcaption", null, titulo));
    const ul = el("ul", "grafico-linhas");
    for (const i of validos) {
      const li = el("li", "barra");
      li.appendChild(el("span", "barra-rotulo", i.rotulo)).title = i.rotulo;
      const trilho = el("span", "barra-trilho");
      trilho.setAttribute("aria-hidden", "true");
      const barra = el("span", "barra-valor-visual");
      barra.style.width = `${Math.max(1, (100 * i.valor) / maximo)}%`;
      barra.style.display = "block";
      trilho.appendChild(barra);
      li.appendChild(trilho);
      li.appendChild(el("span", "barra-numero", fmt(i.valor)));
      ul.appendChild(li);
    }
    fig.appendChild(ul);
    if (nota) fig.appendChild(el("p", "grafico-nota", nota));
    return fig;
  }

  const NOMES_OPERACAO = {
    soma: "Soma", contagem: "Eventos", contagem_distinta: "Contagem distinta", media: "Média",
    mediana: "Mediana", minimo: "Mínimo", maximo: "Máximo", desvio_padrao: "Desvio-padrão",
  };

  function graficoDoResultado(d) {
    if (!d || typeof d !== "object") return null;
    const agregacao = NOMES_OPERACAO[d.agregacao] || d.agregacao || "";
    const unidade = d.unidade ? ` (${d.unidade})` : "";
    const nota = "Calculado pelo sistema a partir do dataset; não vem do texto do Gemini.";
    if (Array.isArray(d.resultado) && Array.isArray(d.agrupar_por) && d.agrupar_por.length) {
      const itens = d.resultado.slice(0, 15).map((g) => ({
        rotulo: d.agrupar_por.map((c) => String(g[c])).join(" · "), valor: g.resultado,
      }));
      return graficoBarras(`${agregacao} por ${d.agrupar_por.join(" e ")}${unidade}`, itens, nota);
    }
    if (d.resultado && Array.isArray(d.resultado.grupos)) {
      const itens = d.resultado.grupos.map((g) => ({ rotulo: String(g.valor), valor: g.resultado }));
      return graficoBarras(`Comparação por ${d.comparar_por || ""}: ${agregacao}${unidade}`, itens, nota);
    }
    return null;
  }

  // ---------------------------------------------------------------- mensagens
  function esconderBoasVindas() { ui.boasVindas.hidden = true; }

  function mostrarUsuario(texto) {
    const art = el("article", "msg msg-usuario");
    art.setAttribute("aria-label", "Sua mensagem");
    art.appendChild(el("div", "msg-texto", texto));
    ui.mensagens.appendChild(art);
    rolarParaFim();
  }

  function cabecalhoAgente() {
    const c = el("div", "msg-cabecalho");
    c.appendChild(icone("marca", "msg-avatar"));
    c.appendChild(document.createTextNode("Tech Layoffs"));
    return c;
  }

  function botaoCopiar(texto) {
    const b = el("button", "acao");
    b.type = "button";
    b.appendChild(icone("copiar"));
    const rotulo = el("span", null, "Copiar");
    b.appendChild(rotulo);
    b.setAttribute("aria-label", "Copiar resposta");
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(texto);
        rotulo.textContent = "Copiado";
        anunciar("Resposta copiada.");
      } catch {
        rotulo.textContent = "Não foi possível copiar";
      }
      window.setTimeout(() => { rotulo.textContent = "Copiar"; }, 2000);
    });
    return b;
  }

  function mostrarResposta(dados) {
    const etapas = (Array.isArray(dados.etapas) ? dados.etapas : []).map(classificarEtapa);
    const fontes = Array.isArray(dados.fontes) ? dados.fontes.filter((f) => typeof f === "string") : [];
    const texto = typeof dados.resposta === "string" ? dados.resposta : "";

    const art = el("article", "msg msg-agente");
    art.setAttribute("aria-label", "Resposta do assistente");
    art.appendChild(cabecalhoAgente());
    const corpo = el("div", "msg-corpo");
    corpo.appendChild(el("div", "msg-texto", texto));
    const grafico = graficoDoResultado(dados.dados);
    if (grafico) corpo.appendChild(grafico);
    const s = selos(etapas);
    if (s) corpo.appendChild(s);
    if (fontes.length) corpo.appendChild(blocoFontes(fontes));
    if (etapas.length) corpo.appendChild(blocoEtapas(etapas));
    const acoes = el("div", "msg-acoes");
    acoes.appendChild(botaoCopiar(texto));
    corpo.appendChild(acoes);
    art.appendChild(corpo);
    ui.mensagens.appendChild(art);
    rolarParaFim();
  }

  function mostrarCarregando() {
    const div = el("div", "msg msg-agente carregando");
    div.setAttribute("role", "status");
    const pontos = el("span", "pontos");
    pontos.setAttribute("aria-hidden", "true");
    pontos.append(el("span"), el("span"), el("span"));
    div.append(pontos, el("span", null, "Analisando a pergunta e consultando os dados..."));
    ui.mensagens.appendChild(div);
    rolarParaFim();
    return div;
  }

  function mostrarErro(titulo, detalhe, pergunta) {
    const div = el("div", "erro");
    div.setAttribute("role", "alert");
    div.appendChild(icone("erro"));
    const texto = el("div");
    texto.appendChild(el("p", "erro-titulo", titulo));
    if (detalhe) texto.appendChild(el("p", null, detalhe));
    if (pergunta) {
      const b = el("button", "acao");
      b.type = "button";
      b.appendChild(icone("repetir"));
      b.appendChild(document.createTextNode("Tentar novamente"));
      b.addEventListener("click", () => { div.remove(); enviar(pergunta, { repetir: true }); });
      texto.appendChild(b);
    }
    div.appendChild(texto);
    ui.mensagens.appendChild(div);
    rolarParaFim();
  }

  // ---------------------------------------------------------------- envio
  function historicoParaEnvio(mensagem) {
    let itens = estado.historico.slice(-LIMITES.itensHistorico)
      .map((h) => ({ papel: h.papel, texto: h.texto.slice(0, LIMITES.caracteresItem) }));
    const bytes = (lista) => new TextEncoder().encode(JSON.stringify({ mensagem, historico: lista })).length;
    while (itens.length && bytes(itens) > LIMITES.bytesCorpo) itens = itens.slice(2);   // descarta o par mais antigo
    return itens;
  }

  function travar(sim) {
    estado.enviando = sim;
    ui.enviar.disabled = sim;
    ui.rolagem.setAttribute("aria-busy", sim ? "true" : "false");
    document.querySelectorAll(".sugestao").forEach((b) => { b.disabled = sim; });
  }

  async function enviar(texto, { repetir = false } = {}) {
    const mensagem = String(texto || "").trim().slice(0, LIMITES.caracteres);
    if (!mensagem || estado.enviando) return;
    esconderBoasVindas();
    if (!repetir) mostrarUsuario(mensagem);
    travar(true);
    const conversa = estado.conversa;
    const controlador = new AbortController();
    estado.controlador = controlador;
    let estourou = false;
    const relogio = window.setTimeout(() => { estourou = true; controlador.abort(); }, LIMITES.timeoutMs);
    const carregando = mostrarCarregando();
    try {
      const r = await fetch("/api/chat", {
        method: "POST", signal: controlador.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mensagem, historico: historicoParaEnvio(mensagem) }),
      });
      const dados = await r.json().catch(() => ({}));
      if (conversa !== estado.conversa) return;                  // nova conversa: descarta a resposta antiga
      carregando.remove();
      if (r.ok && typeof dados.resposta === "string") {
        mostrarResposta(dados);
        estado.historico.push({ papel: "user", texto: mensagem }, { papel: "model", texto: dados.resposta });
        definirEstado(ui.estadoGemini, "ok", "Gemini: respondeu às " +
          new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" }));
        anunciar("Resposta recebida.");
        return;
      }
      const erro = typeof dados.erro === "string" ? dados.erro : "";
      if (r.status === 429) {
        mostrarErro("Limite de mensagens atingido.", erro || "Aguarde alguns minutos e tente novamente.", mensagem);
      } else if (r.status === 503) {
        definirEstado(ui.estadoGemini, "erro", "Gemini: indisponível");
        mostrarErro("O Gemini está indisponível no momento.", erro, mensagem);
      } else if (r.status === 400 || r.status === 413) {
        mostrarErro("A mensagem não pôde ser enviada.", erro || "Verifique o texto e tente de novo.", null);
      } else {
        mostrarErro("Erro no servidor.", erro || `Código ${r.status}.`, mensagem);
      }
    } catch (e) {
      if (conversa !== estado.conversa) return;                  // abortado pela nova conversa
      carregando.remove();
      if (e && e.name === "AbortError" && estourou) {
        mostrarErro("O servidor demorou demais para responder.",
          "Se o app estava dormindo na hospedagem gratuita, a primeira resposta pode levar cerca de um minuto.", mensagem);
      } else {
        definirEstado(ui.estadoServidor, "erro", "Servidor: sem conexão");
        mostrarErro("Não foi possível conectar ao servidor.", "Verifique sua conexão e tente novamente.", mensagem);
      }
    } finally {
      window.clearTimeout(relogio);
      if (conversa === estado.conversa) {
        estado.controlador = null;
        travar(false);
        ui.entrada.focus();
      }
    }
  }

  function novaConversa() {
    estado.conversa += 1;                                        // respostas pendentes passam a ser descartadas
    if (estado.controlador) estado.controlador.abort();
    estado.controlador = null;
    estado.historico = [];
    ui.mensagens.replaceChildren();
    ui.boasVindas.hidden = false;
    travar(false);
    ui.entrada.value = "";
    atualizarContador();
    ui.rolagem.scrollTop = 0;
    ui.entrada.focus();
    anunciar("Nova conversa iniciada.");
  }

  // ---------------------------------------------------------------- composer
  function atualizarContador() {
    const n = ui.entrada.value.length;
    ui.contador.textContent = `${n}/${LIMITES.caracteres}`;
    ui.contador.dataset.perto = n > LIMITES.caracteres * 0.9 ? "sim" : "nao";
    ui.entrada.style.height = "auto";
    ui.entrada.style.height = `${Math.min(ui.entrada.scrollHeight, 160)}px`;
  }

  ui.entrada.addEventListener("input", atualizarContador);
  ui.entrada.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      ui.form.requestSubmit();
    }
  });
  ui.form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const texto = ui.entrada.value;
    if (!texto.trim() || estado.enviando) return;
    ui.entrada.value = "";
    atualizarContador();
    enviar(texto);
  });
  ui.nova.addEventListener("click", novaConversa);
  document.querySelectorAll(".sugestao").forEach((b) => b.addEventListener("click", () => enviar(b.textContent)));

  // ---------------------------------------------------------------- indicadores (dados reais do servidor)
  function preencherResumo(r) {
    $("kpi-eventos").textContent = fmt(r.eventos);
    $("kpi-demitidos").textContent = fmt(r.demitidos);
    $("kpi-empresas").textContent = fmt(r.empresas);
    $("kpi-paises").textContent = fmt(r.paises);
    $("kpi-paises").title = r.observacao || "";
    const data = (iso) => (typeof iso === "string" ? iso.split("-").reverse().join("/") : "—");
    $("kpi-periodo").textContent = `${data(r.periodo?.inicio)} a ${data(r.periodo?.fim)}`;
    const anos = Array.isArray(r.por_ano) ? r.por_ano : [];
    const grafico = graficoBarras("Funcionários demitidos por ano (coluna Year)",
      anos.map((a) => ({ rotulo: String(a.ano), valor: a.demitidos })),
      "Soma de Laid_Off nos eventos registrados de cada ano.");
    $("grafico-anos").replaceChildren(grafico || el("p", "nota", "Sem dados."));
    if (r.modelo) {
      $("modelo-nome").textContent = r.modelo.nome;
      $("modelo-mae").textContent = fmt(r.modelo.mae_teste);
      $("modelo-r2").textContent = typeof r.modelo.r2_teste === "number"
        ? r.modelo.r2_teste.toLocaleString("pt-BR", { maximumFractionDigits: 4 }) : "—";
    }
  }

  async function carregarResumo() {
    try {
      const r = await fetch("/api/dataset/resumo", { headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error(String(r.status));
      preencherResumo(await r.json());
      definirEstado(ui.estadoServidor, "ok", "Servidor: conectado");
    } catch {
      definirEstado(ui.estadoServidor, "erro", "Servidor: indisponível");
      $("grafico-anos").replaceChildren(el("p", "nota", "Não foi possível carregar o resumo do dataset."));
    }
  }

  atualizarContador();
  carregarResumo();
})();
