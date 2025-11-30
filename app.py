import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import warnings
import os

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Анализ тональности", layout="wide")

# === ИДЕАЛЬНЫЕ ТЕМЫ 2025 (сайдбар тоже чёрный!) ===
if "theme" not in st.session_state:
    st.session_state.theme = "Светлая"

# Переключатель в сайдбаре
with st.sidebar:
    st.markdown("### Тема оформления")
    theme_choice = st.radio(
        "Выберите тему",
        ["Светлая", "Тёмная"],
        index=0 if st.session_state.theme == "Светлая" else 1,
        key="theme"
    )

# === ПРИМЕНЯЕМ ТЕМУ ===
if st.session_state.theme == "Тёмная":
    st.markdown("""
    <style>
        /* Полностью чёрный фон */
        .stApp, [data-testid="stSidebar"] > div:first-child {
            background-color: #0e1117 !important;
        }

        /* Сайдбар — чёрный и читаемый */
        [data-testid="stSidebar"] {
            background-color: #0e1117 !important;
        }
        [data-testid="stSidebar"] .css-1d391kg, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: #e0e0e0 !important;
        }

        /* Все тексты белые */
        h1,h2,h3,h4,h5,h6,p,span,div,label,.stMarkdown {
            color: #e0e0e0 !important;
        }

        /* Уведомления */
        .stAlert, [data-testid="stNotification"] > div {
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            color: white !important;
        }
        div[data-testid="stSuccess"] {border-left: 5px solid #00ff9d !important;}
        div[data-testid="stInfo"]    {border-left: 5px solid #4a9eff !important;}
        div[data-testid="stWarning"] {border-left: 5px solid #ffd60a !important;}
        div[data-testid="stError"]   {border-left: 5px solid #ff4b4b !important;}

        /* Кнопки и селекты */
        .stButton > button {
            background: #1e1e2e !important;
            color: white !important;
            border: 1px solid #444 !important;
        }
        .stButton > button:hover {background: #2d2d44 !important;}
    </style>
    """, unsafe_allow_html=True)
else:
    # Светлая — чистый дефолт Streamlit
    st.markdown("<style>.stApp {background: white !important;}</style>", unsafe_allow_html=True)

# === ЛОГОТИП СГТУ ===
logo_path = "assets/СГТУ_имени_Гагарина_Ю.А.png"
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png", width=240)
else:
    st.sidebar.warning("Логотип не найден в папке assets")

st.sidebar.markdown("**СГТУ им. Гагарина Ю.А.**\nКафедра программной инженерии")

# === МОДЕЛЬ ===
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cointegrated/rubert-tiny-sentiment-balanced")

model = load_model()

# === LSTM ===
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size=15000, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 3)
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h.squeeze(0)))

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, vocab=None, max_len=80):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        if vocab is None:
            self.vocab = {'<pad>': 0, '<unk>': 1}
            idx = 2
            for t in texts:
                for w in t.lower().split():
                    if w not in self.vocab:
                        self.vocab[w] = idx
                        idx += 1
        else:
            self.vocab = vocab
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        words = self.texts[idx].lower().split()[:self.max_len]
        ids = [self.vocab.get(w, 1) for w in words]
        ids += [0] * (self.max_len - len(ids))
        item = torch.tensor(ids)
        if self.labels is not None:
            return item, torch.tensor(self.labels[idx])
        return item

def collate_fn(batch):
    if len(batch[0]) == 2:
        texts, labels = zip(*batch)
        texts = pad_sequence(texts, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        return texts, labels
    texts = pad_sequence(batch, batch_first=True, padding_value=0)
    return texts

st.title("Система анализа тональности текстовых сообщений")

tab1, tab2, tab3 = st.tabs(["Одиночный анализ", "Пакетный анализ", "Сравнение моделей"])

# === ВКЛАДКА 1: ОДИНОЧНЫЙ АНАЛИЗ ===
with tab1:
    text = st.text_area("Введите текст для анализа", height=140)
    if st.button("Анализировать", type="primary") and text.strip():
        with st.spinner("Анализ..."):
            res = model(text)[0]
            label = {"positive": "Положительный", "negative": "Отрицательный", "neutral": "Нейтральный"}[res["label"]]
            score = res["score"]
            if res["label"] == "positive":
                st.success(f"Положительный ({score:.1%})")
            elif res["label"] == "negative":
                st.error(f"Отрицательный ({score:.1%})")
            else:
                st.info(f"Нейтральный ({score:.1%})")

# === ВКЛАДКА 2: ПАКЕТНЫЙ АНАЛИЗ ===
with tab2:
    uploaded = st.file_uploader("Загрузите файл (.txt или .csv)", type=["txt", "csv"])
    if uploaded and st.button("Запустить пакетный анализ", type="primary"):
        try:
            if uploaded.name.endswith(".txt"):
                texts = [line.decode("utf-8").strip() for line in uploaded if line.strip()]
            else:
                df = pd.read_csv(uploaded)
                col = "text" if "text" in df.columns else df.columns[0]
                texts = df[col].astype(str).dropna().tolist()

            with st.spinner(f"Анализ {len(texts)} текстов..."):
                preds = model(texts, truncation=True, max_length=512)

            results = pd.DataFrame({
                "Текст": texts,
                "Тональность": [
                    {"positive": "Положительный", "negative": "Отрицательный", "neutral": "Нейтральный"}[p["label"]]
                    for p in preds
                ],
                "Уверенность": [f"{p['score']:.1%}" for p in preds]
            })

            st.dataframe(results, use_container_width=True)

            fig = px.histogram(
                results, x="Тональность", color="Тональность",
                color_discrete_map={"Положительный": "#27ae60", "Отрицательный": "#e74c3c", "Нейтральный": "#95a5a6"}
            )
            st.plotly_chart(fig, use_container_width=True)

            csv = results.to_csv(index=False, encoding="utf-8-sig").encode()
            st.download_button("Скачать результаты", csv, "результаты_анализа.csv", "text/csv")

        except Exception as e:
            st.error(f"Ошибка: {e}")

# === ВКЛАДКА 3: СРАВНЕНИЕ МОДЕЛЕЙ ===
with tab3:
    st.header("Обучение и сравнение моделей")
    st.info("Поддерживаются любые форматы разметки: positive/neutral/negative, 0/1/2, pos/neu/neg и т.д.")

    dataset = st.file_uploader(
        "Загрузите CSV с текстами и разметкой (до 200 МБ)",
        type="csv",
        help="Колонки могут называться: text, message, review, content + sentiment, label, target, class и т.д."
    )

    if dataset:
        try:
            # Читаем максимально гибко
            df = pd.read_csv(
                dataset,
                sep=None,
                engine="python",
                encoding="utf-8",
                on_bad_lines="skip",
                dtype=str
            )

            st.success(f"Загружено {len(df)} строк")

            # --- Автоопределение колонок ---
            text_cols = [c for c in df.columns if any(x in c.lower() for x in ["text", "message", "review", "content", "отзыв", "сообщен"])]
            label_cols = [c for c in df.columns if any(x in c.lower() for x in ["sentiment", "label", "target", "class", "тональность", "оценка"])]

            if not text_cols:
                st.error("Не найдена колонка с текстом (ищется по словам: text, message, review...)")
                st.stop()
            if not label_cols:
                st.error("Не найдена колонка с разметкой (ищется по словам: sentiment, label, target...)")
                st.stop()

            text_col = st.selectbox("Колонка с текстом", text_cols)
            label_col = st.selectbox("Колонка с разметкой", label_cols)

            df = df[[text_col, label_col]].dropna()
            df = df[df[text_col].str.strip() != ""]

            # --- Автонормализация меток ---
            raw_labels = df[label_col].astype(str).str.lower().str.strip()

            # Словарь всех возможных вариантов → 0,1,2
            label_mapping = {}
            for label in raw_labels.unique():
                if label in ["positive", "pos", "1", "+1", "позитив", "хорошо", "good"]:
                    label_mapping[label] = 0
                elif label in ["neutral", "neu", "2", "0", "нейтрал", "норма", "normal"]:
                    label_mapping[label] = 1
                elif label in ["negative", "neg", "0", "-1", "отрицательный", "плохо", "bad"]:
                    label_mapping[label] = 2
                else:
                    # если число — пробуем напрямую
                    try:
                        num = int(float(label))
                        if num in [0, 1, 2]:
                            label_mapping[label] = num
                    except:
                        pass

            if len(label_mapping) < 3:
                st.error(f"Не удалось распознать 3 класса тональности. Найдено: {list(label_mapping.keys())}")
                st.stop()

            df["label"] = raw_labels.map(label_mapping).astype(int)
            df["text"] = df[text_col].astype(str)

            st.success(f"Успешно распознано {len(df)} примеров, классы: {dict((v,k) for k,v in label_mapping.items())}")

            # Дальше — как раньше: train/test split и обучение
            X_train, X_test, y_train, y_test = train_test_split(
                df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
            )

        except Exception as e:
            st.error(f"Ошибка: {e}")

        if "results" not in st.session_state:
            st.session_state.results = []

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("BoW + LogReg"):
                with st.spinner("Обучение..."):
                    vec = CountVectorizer(max_features=15000)
                    clf = LogisticRegression(max_iter=1000)
                    X_vec_train = vec.fit_transform(X_train)
                    X_vec_test = vec.transform(X_test)
                    clf.fit(X_vec_train, y_train)
                    acc = accuracy_score(y_test, clf.predict(X_vec_test))
                    st.session_state.results.append({"Модель": "BoW + LogReg", "Точность": acc})
                    st.success(f"{acc:.1%}")

        with col2:
            if st.button("Своя LSTM"):
                with st.spinner("Обучение LSTM (1–2 мин)..."):
                    train_ds = TextDataset(X_train.tolist(), y_train.tolist())
                    test_ds = TextDataset(X_test.tolist(), y_test.tolist(), vocab=train_ds.vocab)
                    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
                    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)

                    lstm = SimpleLSTM(vocab_size=len(train_ds.vocab))
                    opt = torch.optim.Adam(lstm.parameters(), lr=0.001)
                    crit = nn.CrossEntropyLoss()

                    for _ in range(6):
                        lstm.train()
                        for x, y in train_loader:
                            opt.zero_grad()
                            out = lstm(x)
                            loss = crit(out, y)
                            loss.backward()
                            opt.step()

                    lstm.eval()
                    all_preds = []
                    with torch.no_grad():
                        for x, _ in test_loader:
                            preds = torch.argmax(lstm(x), dim=1)
                            all_preds.extend(preds.tolist())

                    acc = accuracy_score(y_test.tolist(), all_preds)
                    st.session_state.results.append({"Модель": "Своя LSTM", "Точность": acc})
                    st.success(f"{acc:.1%}")

        with col3:
            if st.button("RuBERT-tiny"):
                with st.spinner("Оценка..."):
                    preds = model(X_test.tolist(), truncation=True, max_length=512)
                    pred_labels = [0 if p["label"] == "positive" else 1 if p["label"] == "neutral" else 2 for p in preds]
                    acc = accuracy_score(y_test.tolist(), pred_labels)
                    st.session_state.results.append({"Модель": "RuBERT-tiny", "Точность": acc})
                    st.success(f"{acc:.1%}")

        if st.session_state.results:
            res_df = pd.DataFrame(st.session_state.results).drop_duplicates("Модель")
            st.table(res_df)
            fig = px.bar(res_df, x="Модель", y="Точность", text="Точность", color="Модель", range_y=[0, 1])
            fig.update_traces(texttemplate="%{text:.1%}")
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Очистить результаты"):
                st.session_state.results = []
                st.rerun()

st.caption("© 2025 Саратовский государственный технический университет имени Ю.А. Гагарина")