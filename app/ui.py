import os

import requests
import streamlit as st

from app.core.config import get_settings


_SETTINGS = get_settings()
API_URL = os.getenv(
    "RAG_API_URL",
    f"http://{_SETTINGS.api_host}:{_SETTINGS.api_port}",
).rstrip("/")


def api_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    key = os.getenv("RAG_UI_API_KEY")
    if not key and _SETTINGS.api_key is not None:
        key = _SETTINGS.api_key.get_secret_value()
    workspace = os.getenv("RAG_WORKSPACE_ID", _SETTINGS.workspace_id).strip()
    if key:
        headers["X-API-Key"] = key
    if workspace:
        headers["X-Workspace-ID"] = workspace
    return headers


SUGGESTIONS = [
    (":material/payments:", "Quel est le montant de l'allocation annuelle ?"),
    (":material/schedule:", "Quels sont les horaires du support informatique ?"),
    (":material/home_work:", "Quelles sont les règles concernant le télétravail ?"),
]


st.set_page_config(
    page_title="ClairDoc — Assistant documentaire",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17203b;
        --muted: #69728d;
        --primary: #5b5ce2;
        --primary-dark: #4546bd;
        --accent: #ff8a65;
        --surface: rgba(255, 255, 255, 0.88);
        --line: #e7e9f4;
        --success: #198754;
    }

    .stApp {
        background:
            radial-gradient(circle at 78% 4%, rgba(113, 100, 255, 0.12), transparent 28rem),
            radial-gradient(circle at 18% 88%, rgba(255, 138, 101, 0.10), transparent 24rem),
            #f7f8fc;
        color: var(--ink);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.5rem; }
    #MainMenu, footer { visibility: hidden; }

    .main .block-container {
        max-width: 1040px;
        padding-top: 2.2rem;
        padding-bottom: 7rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(249, 250, 255, 0.96);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.6rem 1.25rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: .75rem;
        margin: .1rem 0 1.35rem;
    }

    .brand-mark {
        width: 2.65rem;
        height: 2.65rem;
        display: grid;
        place-items: center;
        border-radius: .9rem;
        color: white;
        font-size: 1.25rem;
        background: linear-gradient(145deg, #6f70ef, #4c4dc7);
        box-shadow: 0 10px 24px rgba(91, 92, 226, .28);
    }

    .brand-name { font-size: 1.13rem; font-weight: 800; color: var(--ink); }
    .brand-subtitle { color: var(--muted); font-size: .78rem; margin-top: -.1rem; }

    .connection-card {
        display: flex;
        align-items: center;
        gap: .65rem;
        border: 1px solid var(--line);
        background: white;
        padding: .72rem .85rem;
        border-radius: .9rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 5px 18px rgba(39, 46, 84, .05);
    }

    .status-dot {
        width: .62rem;
        height: .62rem;
        border-radius: 50%;
        background: #c4c7d2;
        box-shadow: 0 0 0 4px rgba(196, 199, 210, .18);
    }

    .status-dot.online {
        background: #20b26b;
        box-shadow: 0 0 0 4px rgba(32, 178, 107, .13);
    }

    .connection-title { font-size: .84rem; font-weight: 700; color: var(--ink); }
    .connection-meta { font-size: .72rem; color: var(--muted); }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        padding: .34rem .65rem;
        border-radius: 999px;
        color: #4c4dc7;
        background: #eeeeff;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .02em;
        margin-bottom: .9rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(91, 92, 226, .12);
        background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(244,244,255,.92));
        border-radius: 1.6rem;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 22px 55px rgba(42, 48, 95, .09);
    }

    .hero:after {
        content: "✦";
        position: absolute;
        right: 1.9rem;
        top: .3rem;
        color: rgba(91, 92, 226, .10);
        font-size: 8rem;
        transform: rotate(12deg);
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        max-width: 700px;
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.04;
        letter-spacing: -.045em;
        margin: 0 0 .8rem;
    }

    .hero p {
        position: relative;
        z-index: 1;
        max-width: 650px;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
        margin: 0;
    }

    .steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .8rem;
        margin: 1.3rem 0 1.8rem;
    }

    .step {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 1rem;
        padding: 1rem;
    }

    .step-number {
        color: var(--primary);
        font-weight: 800;
        font-size: .76rem;
        margin-bottom: .28rem;
    }

    .step-title { color: var(--ink); font-weight: 750; font-size: .9rem; }
    .step-copy { color: var(--muted); font-size: .78rem; margin-top: .22rem; }

    .section-label {
        color: var(--ink);
        font-weight: 800;
        font-size: 1rem;
        margin: 1.1rem 0 .55rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        gap: .65rem;
        background: white;
        border: 1.5px dashed #b9bced;
        border-radius: 1rem;
        padding: 1rem;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary);
        background: #f8f8ff;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        font-size: 0;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:after {
        content: "Déposez vos documents ici";
        display: block;
        font-size: .9rem;
        font-weight: 700;
        color: var(--ink);
    }

    [data-testid="stFileUploaderDropzoneInstructions"] > div > small {
        font-size: 0;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] > div > small:after {
        content: "PDF, DOCX, TXT ou MD · 20 Mo maximum par fichier";
        display: block;
        margin-top: .2rem;
        font-size: .72rem;
        color: var(--muted);
    }

    [data-testid="stFileUploaderDropzone"] button {
        order: 2;
        width: 100%;
        font-size: 0 !important;
    }

    [data-testid="stFileUploaderDropzone"] button * {
        display: none !important;
    }

    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Parcourir les fichiers";
        display: inline-block;
        font-size: .82rem;
        font-weight: 700;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        order: 1;
        width: 100%;
        text-align: center;
    }

    .stButton > button {
        border-radius: .82rem;
        min-height: 2.7rem;
        font-weight: 700;
        border-color: var(--line);
        transition: all .18s ease;
    }

    .stButton > button:hover {
        border-color: var(--primary);
        color: var(--primary);
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(91, 92, 226, .10);
    }

    .stButton > button[kind="primary"] {
        color: white;
        border: 0;
        background: linear-gradient(135deg, #6667ed, #5051cd);
        box-shadow: 0 8px 20px rgba(91, 92, 226, .22);
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, .78);
        border: 1px solid var(--line);
        border-radius: 1.15rem;
        padding: .35rem .65rem;
        margin-bottom: .75rem;
        box-shadow: 0 8px 24px rgba(35, 42, 79, .045);
    }

    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, .96);
        border: 1px solid #dfe1ef;
        border-radius: 1rem;
        box-shadow: 0 15px 35px rgba(38, 44, 83, .13);
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: .85rem;
        background: #fbfbfe;
    }

    [data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: .9rem;
        background: white;
        padding: .6rem .8rem;
    }

    .privacy-note {
        color: var(--muted);
        font-size: .72rem;
        line-height: 1.45;
        padding: .75rem .2rem 0;
    }

    .stApp {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stButton > button:focus-visible,
    [data-testid="stChatInput"]:focus-within,
    [data-testid="stFileUploaderDropzone"]:focus-within {
        outline: 3px solid rgba(91, 92, 226, .28);
        outline-offset: 2px;
    }

    .stButton > button:disabled {
        opacity: .56;
        transform: none;
        box-shadow: none;
    }

    [data-testid="stChatMessageContent"] {
        color: var(--ink);
        line-height: 1.7;
    }

    [data-testid="stAlert"] {
        border-radius: .9rem;
        border: 1px solid rgba(91, 92, 226, .16);
    }

    @media (max-width: 760px) {
        .main .block-container { padding: 1rem 1rem 6rem; }
        .hero { padding: 1.5rem; border-radius: 1.2rem; }
        .hero:after { display: none; }
        .steps { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_error(response: requests.Response) -> str:
    if response.status_code == 401:
        return "Accès refusé par l'API. Vérifiez la clé RAG_UI_API_KEY."
    if response.status_code >= 500:
        return "Le service documentaire rencontre un problème. Réessayez dans quelques instants."
    try:
        return str(response.json().get("detail", "La requête n'a pas pu aboutir."))
    except requests.JSONDecodeError:
        return "La requête n'a pas pu aboutir."


@st.cache_data(ttl=5, show_spinner=False)
def get_api_health() -> tuple[bool, dict]:
    try:
        response = requests.get(
            f"{API_URL}/health", timeout=3, headers=api_headers()
        )
        if response.status_code == 401:
            # API is reachable but protected; UI can still show it as online.
            return True, {"auth_required": True}
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException:
        return False, {}


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    label = f"{len(sources)} source{'s' if len(sources) > 1 else ''} consultée{'s' if len(sources) > 1 else ''}"
    with st.expander(label, icon=":material/menu_book:"):
        for position, source in enumerate(sources, start=1):
            source_name = source.get("source", "Document inconnu")
            metadata = []
            if source.get("page"):
                metadata.append(f"Page {source['page']}")
            if source.get("score") is not None:
                metadata.append(f"Classement {source['score']:.0%}")
            if source.get("confidence") is not None:
                metadata.append(f"Confiance {source['confidence']:.0%}")
            preview = source.get("preview") or source.get("content") or "Extrait indisponible."
            with st.container(border=True):
                st.markdown(f"**{position:02d} · {source_name}**")
                if metadata:
                    st.caption(" · ".join(metadata))
                st.write(preview)


def display_message(message: dict) -> None:
    avatar = ":material/person:" if message["role"] == "user" else ":material/smart_toy:"
    with st.chat_message(message["role"], avatar=avatar, width="stretch"):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))


for key, default in {
    "messages": [],
    "indexed_files": 0,
    "indexed_chunks": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

api_online, config = get_api_health()
provider_names = {
    "anthropic": "Claude",
    "openai": "OpenAI",
    "ollama": "Ollama",
    "local": "Local",
    "local-lite": "Local léger",
    "semantic": "Sémantique",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "chroma": "ChromaDB",
    "faiss": "FAISS",
}

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">✦</div>
            <div>
                <div class="brand-name">ClairDoc</div>
                <div class="brand-subtitle">Assistant documentaire</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if api_online:
        if config.get("auth_required"):
            st.markdown(
                """
                <div class="connection-card">
                    <div class="status-dot online"></div>
                    <div>
                        <div class="connection-title">API disponible</div>
                        <div class="connection-meta">Authentification requise (RAG_UI_API_KEY)</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            engine_value = config.get("engine", "langchain")
            store_value = config.get("vector_store", "chroma")
            llm_value = config.get("llm_provider", "anthropic")
            engine = provider_names.get(engine_value, engine_value)
            store = provider_names.get(store_value, store_value)
            llm = provider_names.get(llm_value, llm_value)
            st.markdown(
                f"""
                <div class="connection-card">
                    <div class="status-dot online"></div>
                    <div>
                        <div class="connection-title">Assistant disponible</div>
                        <div class="connection-meta">{llm} · {engine} · {store}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="connection-card">
                <div class="status-dot"></div>
                <div>
                    <div class="connection-title">Assistant hors ligne</div>
                    <div class="connection-meta">Vérifiez que FastAPI est démarré</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Bibliothèque documentaire")
    st.caption("Importez vos sources pour interroger votre base documentaire")
    uploads = st.file_uploader(
        "Sélectionner des documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        max_upload_size=20,
        width="stretch",
        label_visibility="collapsed",
    )

    if uploads:
        total_size = sum(upload.size for upload in uploads) / (1024 * 1024)
        st.caption(f"{len(uploads)} fichier(s) sélectionné(s) · {total_size:.1f} Mo")

    if st.button(
        "Indexer les documents",
        icon=":material/upload_file:",
        type="primary",
        disabled=not uploads or not api_online,
        width="stretch",
    ):
        payload = [
            ("files", (upload.name, upload.getvalue(), upload.type)) for upload in uploads
        ]
        with st.status("Préparation de votre base documentaire…", expanded=True) as status:
            st.write("Lecture et découpage des documents")
            try:
                response = requests.post(
                    f"{API_URL}/documents/ingest",
                    files=payload,
                    timeout=300,
                    headers=api_headers(),
                )
                if response.ok:
                    result = response.json()
                    indexed_count = len(result["files"])
                    st.session_state.indexed_files += indexed_count
                    st.session_state.indexed_chunks += result["chunks"]
                    if indexed_count:
                        status_label = f"{indexed_count} document(s) prêt(s) à interroger"
                        st.toast("Base documentaire mise à jour", icon="✅")
                    else:
                        status_label = "Aucun nouveau document"
                        st.info("Ces documents sont déjà présents dans votre base.")
                    status.update(
                        label=status_label,
                        state="complete",
                        expanded=False,
                    )
                else:
                    status.update(label="Indexation interrompue", state="error")
                    st.error(api_error(response))
            except requests.RequestException:
                status.update(label="API inaccessible", state="error")
                st.error("Impossible de contacter l'API. Vérifiez que le serveur FastAPI est démarré.")

    if st.session_state.indexed_files:
        metric_files, metric_chunks = st.columns(2)
        metric_files.metric("Documents indexés", st.session_state.indexed_files)
        metric_chunks.metric("Extraits disponibles", st.session_state.indexed_chunks)

    st.space("small")
    if st.button(
        "Nouvelle conversation",
        icon=":material/add_comment:",
        disabled=not st.session_state.messages,
        width="stretch",
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div class="privacy-note">
            🔒 Les documents restent dans votre index local. Seuls les extraits utiles sont envoyés au modèle de réponse.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Votre espace documentaire</div>
        <h1>Vos documents ont beaucoup à vous dire.</h1>
        <p>
            Ajoutez vos fichiers, posez une question naturellement et obtenez une réponse claire,
            accompagnée des passages qui la justifient.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not api_online:
    st.warning(
        "L'assistant est momentanément indisponible. Vérifiez que l'API FastAPI est démarrée.",
        icon=":material/cloud_off:",
    )
elif config.get("auth_required"):
    st.info(
        "L'API est en ligne, mais l'authentification est requise. Configurez RAG_UI_API_KEY pour interroger l'assistant.",
        icon=":material/key:",
    )

suggested_question = None
if not st.session_state.messages:
    st.markdown(
        """
        <div class="steps">
            <div class="step">
                <div class="step-number">01 · Importer</div>
                <div class="step-title">Ajoutez vos fichiers</div>
                <div class="step-copy">Utilisez le panneau à gauche pour créer votre base.</div>
            </div>
            <div class="step">
                <div class="step-number">02 · Interroger</div>
                <div class="step-title">Posez une question</div>
                <div class="step-copy">Écrivez comme vous parleriez à un collègue.</div>
            </div>
            <div class="step">
                <div class="step-number">03 · Vérifier</div>
                <div class="step-title">Vérifiez les sources</div>
                <div class="step-copy">Chaque réponse indique les extraits consultés.</div>
            </div>
        </div>
        <div class="section-label">Essayez avec une question</div>
        """,
        unsafe_allow_html=True,
    )
    suggestion_columns = st.columns(3)
    for column, (icon, prompt) in zip(suggestion_columns, SUGGESTIONS, strict=True):
        with column:
            if st.button(
                prompt,
                icon=icon,
                key=f"suggestion-{prompt}",
                width="stretch",
                disabled=not api_online,
            ):
                suggested_question = prompt
else:
    for message in st.session_state.messages:
        display_message(message)

typed_question = st.chat_input(
    "Posez une question sur vos documents…",
    max_chars=4000,
    disabled=not api_online,
)
question = typed_question or suggested_question

if question:
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[-10:]
    ]
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    display_message(user_message)

    with st.chat_message("assistant", avatar=":material/smart_toy:", width="stretch"):
        with st.spinner("Je cherche les passages les plus utiles…"):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "history": history},
                    timeout=180,
                    headers=api_headers(),
                )
                if response.ok:
                    result = response.json()
                    assistant_message = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                    st.markdown(assistant_message["content"])
                    render_sources(assistant_message["sources"])
                    st.session_state.messages.append(assistant_message)
                    st.rerun()
                else:
                    st.error(f"Je n'ai pas pu répondre : {api_error(response)}")
            except requests.RequestException:
                st.error("La connexion avec l'assistant a été interrompue. Réessayez dans quelques instants.")
