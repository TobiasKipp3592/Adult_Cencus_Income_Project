# AI Assistant for Data Science

## Inhaltsverzeichnis

- [Projektbeschreibung](#projektbeschreibung)
- [Dictionary](#dictionary)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Nutzung](#nutzung-des-streamlit-tools)
- [Technologien] (#technologien)


## Projektbeschreibung

Dieses Projekt bietet ein interaktives Streamlit-Tool zur Exploratory Data Analysis (EDA) und Modellierung von Machine Learning-Problemen. Die Anwendung erlaubt das Hochladen, Bereinigen, Analysieren und Modellieren von Daten.

## Dictionary

age – Alter der Person (numerisch)
workclass – Art der Beschäftigung (z. B. "Private", "Self-emp-not-inc", "Federal-gov")
fnlwgt – Gewichtungsfaktor für die Stichprobe (numerisch)
education – Höchster Bildungsabschluss (z. B. "Bachelors", "HS-grad", "Masters")
education.num – Numerische Darstellung des Bildungsniveaus (1 bis 16)
marital.status – Familienstand (z. B. "Married-civ-spouse", "Never-married")
occupation – Beruf (z. B. "Tech-support", "Craft-repair", "?")
relationship – Beziehung innerhalb der Familie (z. B. "Wife", "Own-child", "Unmarried")
race – Ethnische Zugehörigkeit (z. B. "White", "Black", "Asian-Pac-Islander")
sex – Geschlecht ("Male" oder "Female")
capital.gain – Kapitalgewinne aus Investitionen (numerisch)
capital.loss – Kapitalverluste aus Investitionen (numerisch)
hours.per.week – Wöchentliche Arbeitsstunden (numerisch)
native.country – Herkunftsland (z. B. "United-States", "Mexico", "?")
income – Einkommen (Zielvariable: entweder <=50K oder >50K)

## Projektstruktur



## Vorraussetzungen

### Repository klonen:

~~~python
git clone <URL>
~~~

### Erforderliche Bibliotheken installieren

~~~python
pip install -r requirements.txt
~~~

### Umgebungsvariablen konfigurieren

Erstelle eine .env-Datei und füge deinen OPENAI-API-Key hinzu

~~~python
OPENAI_API_KEY=dein_api_key
~~~

### Anwendung starten

~~~python
streamlit run app.py
~~~


## Nutzung des Streamlit-Tools

1. Datensatz hochladen (CSV-Datei)
2. Daten bereinigen
3. Daten analysieren
4. Data Science Frage stellen
5. Machine-Learning-Modell auswählen


## Technologien

- Streamlit - Interaktive UI
- LangChain & OpenAI API - KI gestützte Analysen
- Pandas, seaborn, Matplotlib - Datenanalyse & Visualisierung
- Scikit-Learn & Imbalanced-Learn - Machine-Learning
- Optuna - Hyperparameter-Optimierung
- Fairlearn - Fairness-Analyse


## Autor:in

Tobias Kipp
Kimberly Koblinsky