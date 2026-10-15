from collections import Counter
from math import exp, factorial
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="Pronosticatore Calcio & Telegram Schedina",
    page_icon="⚽",
    layout="centered",
)

# ==================================================
# CONFIGURAZIONE COMPETIZIONI
# ==================================================
COMPETIZIONI = {"Serie A": "serieA.csv", "europa League": "europaleague.csv"}

ELO_START = 1500
K = 20
HOME_ADVANTAGE = 70
SIMULATIONS = 5000

# Inizializzazione del carrello schedina nella memoria di sessione di Streamlit
if "schedina" not in st.session_state:
  st.session_state.schedina = []

# ==================================================
# SELETTORE COMPETIZIONE NELLA GUI (TITOLO CENTRATO)
# ==================================================
st.markdown(
    "<h1"
    " style='text-align: center; color: #FFFFFF;'>⚽ Pronostici Calcio &"
    " Schedina di Pietro Marzano</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align: center; color: #A0A0A0;'>Analisi dedicata a"
    " <b>Over/Under 2.5</b> e <b>Goal / No Goal</b> con invio Telegram e"
    " Ricevuta</h3>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Menu a tendina per scegliere la competizione
competizione_scelta = st.selectbox(
    "🏆 Seleziona Competizione", list(COMPETIZIONI.keys())
)
CSV_FILE = COMPETIZIONI[competizione_scelta]


@st.cache_data
def load_data(file_path):
  try:
    df = pd.read_csv(file_path, encoding="utf-8")
  except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="latin1")
  df.columns = df.columns.str.strip()
  return df


# Tentativo di caricamento del file CSV della competizione scelta
try:
  df = load_data(CSV_FILE)
except FileNotFoundError:
  st.error(
      f"⚠️ Attenzione: Il file '{CSV_FILE}' per la competizione"
      f" '{competizione_scelta}' non è stato trovato nella cartella del"
      " progetto!"
  )
  st.stop()

# Rilevamento automatico flessibile dei nomi delle colonne
possible_result_cols = [
    "risultati",
    "Risultato",
    "risultato",
    "Score",
    "FT",
    "Risultati",
]
RESULT_COL = next(
    (col for col in possible_result_cols if col in df.columns), None
)

possible_home_cols = [
    "Squadra-A",
    "HomeTeam",
    "Casa",
    "Home",
    "squadra_casa",
]
HOME_COL = next(
    (col for col in possible_home_cols if col in df.columns), "Squadra-A"
)

possible_away_cols = [
    "Squadra-B",
    "AwayTeam",
    "Ospiti",
    "Away",
    "squadra_trasferta",
]
AWAY_COL = next(
    (col for col in possible_away_cols if col in df.columns), "Squadra-B"
)

if RESULT_COL is None:
  st.error(
      f"⚠️ Errore nel file '{CSV_FILE}': Non riesco a trovare la colonna dei"
      f" risultati. Colonne presenti: {list(df.columns)}"
  )
  st.stop()

# Pulizia partite giocate in modo sicuro
played = df[
    (~df[RESULT_COL].isna()) & (df[RESULT_COL].astype(str).str.strip() != "")
].copy()


def estrai_gol(valore, posizione):
  try:
    valore_str = str(valore).replace("–", "-").replace("—", "-")
    parti = valore_str.split("-")
    if len(parti) == 2:
      return int(parti[posizione].strip())
  except (ValueError, TypeError):
    pass
  return None


played["gol_casa"] = played[RESULT_COL].apply(lambda x: estrai_gol(x, 0))
played["gol_trasferta"] = played[RESULT_COL].apply(lambda x: estrai_gol(x, 1))

played = played.dropna(subset=["gol_casa", "gol_trasferta"])
played["gol_casa"] = played["gol_casa"].astype(int)
played["gol_trasferta"] = played["gol_trasferta"].astype(int)

# ==================================================
# CALCOLO ELO E STATISTICHE
# ==================================================
ratings = {}


def expected_score(r1, r2):
  return 1 / (1 + 10 ** ((r2 - r1) / 400))


for _, row in played.iterrows():
  home = row[HOME_COL]
  away = row[AWAY_COL]

  if home not in ratings:
    ratings[home] = ELO_START
  if away not in ratings:
    ratings[away] = ELO_START

  hg = row["gol_casa"]
  ag = row["gol_trasferta"]

  ra = ratings[home]
  rb = ratings[away]

  ea = expected_score(ra + HOME_ADVANTAGE, rb)
  eb = expected_score(rb, ra + HOME_ADVANTAGE)

  if hg > ag:
    sa, sb = 1, 0
  elif hg < ag:
    sa, sb = 0, 1
  else:
    sa, sb = 0.5, 0.5

  ratings[home] += K * (sa - ea)
  ratings[away] += K * (sb - eb)

avg_home_goals = played["gol_casa"].mean() if len(played) > 0 else 1.2
avg_away_goals = played["gol_trasferta"].mean() if len(played) > 0 else 1.0

teams = sorted(set(played[HOME_COL]).union(set(played[AWAY_COL])))

attack_home = {}
attack_away = {}
defense_home = {}
defense_away = {}

for team in teams:
  home_games = played[played[HOME_COL] == team]
  away_games = played[played[AWAY_COL] == team]

  attack_home[team] = (
      home_games["gol_casa"].mean() if len(home_games) > 0 else avg_home_goals
  )
  attack_away[team] = (
      away_games["gol_trasferta"].mean()
      if len(away_games) > 0
      else avg_away_goals
  )
  defense_home[team] = (
      home_games["gol_trasferta"].mean()
      if len(home_games) > 0
      else avg_away_goals
  )
  defense_away[team] = (
      home_games["gol_casa"].mean() if len(home_games) > 0 else avg_home_goals
  )


def recent_form(team, n=5):
  recent = played[
      (played[HOME_COL] == team) | (played[AWAY_COL] == team)
  ].tail(n)

  points = 0
  for _, row in recent.iterrows():
    hg = row["gol_casa"]
    ag = row["gol_trasferta"]

    if row[HOME_COL] == team:
      if hg > ag:
        points += 3
      elif hg == ag:
        points += 1
    else:
      if ag > hg:
        points += 3
      elif ag == hg:
        points += 1
  return points


def predict_match(home, away):
  form_home = recent_form(home)
  form_away = recent_form(away)
  form_factor = (form_home - form_away) / 15

  lambda_home = (attack_home[home] * defense_away[away]) / avg_home_goals
  lambda_away = (attack_away[away] * defense_home[home]) / avg_home_goals

  lambda_home *= 1 + form_factor * 0.15
  lambda_away *= 1 - form_factor * 0.15

  lambda_home = max(lambda_home, 0.20)
  lambda_away = max(lambda_away, 0.20)

  over25 = 0
  both_score = 0
  exact_scores = Counter()

  for _ in range(SIMULATIONS):
    hg = np.random.poisson(lambda_home)
    ag = np.random.poisson(lambda_away)

    exact_scores[f"{hg}-{ag}"] += 1

    if hg + ag > 2:
      over25 += 1
    if hg > 0 and ag > 0:
      both_score += 1

  p_over = round(over25 / SIMULATIONS * 100, 2)
  p_under = round(100 - p_over, 2)
  p_goal = round(both_score / SIMULATIONS * 100, 2)
  p_nogoal = round(100 - p_goal, 2)

  return {
      "OVER25": p_over,
      "UNDER25": p_under,
      "GOAL": p_goal,
      "NOGOAL": p_nogoal,
      "xG_home": round(lambda_home, 2),
      "xG_away": round(lambda_away, 2),
      "total_xG": round(lambda_home + lambda_away, 2),
      "top_results": exact_scores.most_common(5),
  }


# ==================================================
# SELEZIONE SQUADRE E VISUALIZZAZIONE GUI (DOPPIA RICERCA)
# ==================================================
st.markdown("### 🔍 Seleziona le Squadre")

col1, col2 = st.columns(2)

with col1:
  st.markdown("#### 🏠 Casa")
  filtro_casa = (
      st.text_input("Cerca squadra in casa:", "", key="search_home")
      .strip()
      .lower()
  )

  teams_casa = (
      [t for t in teams if filtro_casa in t.lower()] if filtro_casa else teams
  )
  if not teams_casa:
    st.warning("Nessuna squadra trovata.")
    teams_casa = teams

  home_team = st.selectbox(
      "Seleziona Casa", teams_casa, key="home_select", label_visibility="collapsed"
  )

with col2:
  st.markdown("#### ✈️ Trasferta")
  filtro_trasferta = (
      st.text_input("Cerca squadra in trasferta:", "", key="search_away")
      .strip()
      .lower()
  )

  teams_trasferta = (
      [t for t in teams if filtro_trasferta in t.lower()]
      if filtro_trasferta
      else teams
  )
  teams_trasferta = [t for t in teams_trasferta if t != home_team]

  if not teams_trasferta:
    teams_trasferta = [t for t in teams if t != home_team]

  away_team = st.selectbox(
      "Seleziona Trasferta",
      teams_trasferta,
      key="away_select",
      label_visibility="collapsed",
  )

st.markdown("---")

if home_team == away_team:
  st.warning("⚠️ Seleziona due squadre diverse per poter effettuare l'analisi.")
else:
  if st.button("🔍 Analizza Partita", type="primary", use_container_width=True):
    p = predict_match(home_team, away_team)
    st.session_state.current_analysis = {
        "competizione": competizione_scelta,
        "partita": f"{home_team} vs {away_team}",
        "p": p,
    }

  # Se un'analisi è stata eseguita, mostriamo i risultati e il box per aggiungere al carrello
  if (
      "current_analysis" in st.session_state
      and st.session_state.current_analysis["partita"]
      == f"{home_team} vs {away_team}"
  ):
    p = st.session_state.current_analysis["p"]

    # Sezione Gol Attesi (xG)
    xg1, xg2, xg3 = st.columns(3)
    xg1.metric(label=f"xG {home_team}", value=p["xG_home"])
    xg2.metric(label=f"xG {away_team}", value=p["xG_away"])
    xg3.metric(label="Totale xG stimati", value=p["total_xG"])

    st.markdown("---")

    # Sezione Over / Under 2.5
    st.markdown("#### ⚽ Mercato Over / Under 2.5")
    ou1, ou2 = st.columns(2)
    ou1.metric(label="Over 2.5", value=f"{p['OVER25']}%")
    ou2.metric(label="Under 2.5", value=f"{p['UNDER25']}%")

    if p["OVER25"] > 55:
      st.success("🟢 **Tendenza Consigliata:** OVER 2.5 (Partita aperta e da gol)")
    elif p["UNDER25"] > 55:
      st.error(
          "🔴 **Tendenza Consigliata:** UNDER 2.5 (Partita chiusa o bloccata)"
      )
    else:
      st.info(
          "⚖️ **Tendenza:** Situazione di perfetto equilibrio (valutare le"
          " quote)"
      )

    st.markdown("---")

    # Sezione Goal / No Goal
    st.markdown("#### 🥅 Mercato Goal / No Goal")
    gg1, gg2 = st.columns(2)
    gg1.metric(label="Goal (Entrambe a segno)", value=f"{p['GOAL']}%")
    gg2.metric(label="No Goal (Almeno una a zero)", value=f"{p['NOGOAL']}%")

    if p["GOAL"] > 55:
      st.success(
          "🟢 **Tendenza Consigliata:** GOAL (Entrambe le squadre probabile"
          " marcatrici)"
      )
    elif p["NOGOAL"] > 55:
      st.error(
          "🔴 **Tendenza Consigliata:** NO GOAL (Probabile clean sheet di una"
          " difesa)"
      )
    else:
      st.info(
          "⚖️ **Tendenza:** Situazione di perfetto equilibrio (valutare le"
          " quote)"
      )

    st.markdown("---")

    # Sezione Risultati Esatti
    st.markdown("#### 🎯 Top 5 Risultati Esatti Più Probabili")
    for score, count in p["top_results"]:
      perc = round(count / SIMULATIONS * 100, 2)
      st.write(f"• **Risultato {score}** → `{perc}%`")

    st.markdown("---")
    st.markdown("### 🛒 Aggiungi alla Schedina / Carrello")

    col_mercato, col_quota = st.columns(2)
    with col_mercato:
      mercato_scelto = st.selectbox(
          "Seleziona Pronostico",
          ["Over 2.5", "Under 2.5", "Goal", "No Goal"],
          key="sel_mercato",
      )
    with col_quota:
      quota_inserita = st.number_input(
          "Quota Bookmaker",
          min_value=1.01,
          max_value=100.0,
          value=1.75,
          step=0.01,
          key="sel_quota",
      )

    if st.button("➕ Inserisci nel Carrello Schedina", use_container_width=True):
      item = {
          "competizione": competizione_scelta,
          "partita": f"{home_team} vs {away_team}",
          "pronostico": mercato_scelto,
          "quota": quota_inserita,
      }
      st.session_state.schedina.append(item)
      st.success("✅ Partita aggiunta con successo alla schedina!")

# ==================================================
# SEZIONE CARRELLO / SCHEDINA SALVATA (SEMPRE VISIBILE IN FONDO)
# ==================================================
st.markdown("---")
st.subheader("🛒 Carrello Schedina & Riepilogo Giocate")

if len(st.session_state.schedina) == 0:
  st.info(
      "ℹ️ Il carrello è attualmente vuoto. Analizza una partita e clicca su"
      " 'Inserisci nel Carrello Schedina' per vederla qui."
  )
else:
  quota_totale_multipla = 1.0

  # Testo formattato per Telegram e ricevuta
  testo_telegram = "⚽ *LA MIA SCHEDINA MULTIPLA* ⚽\n\n"

  for idx, bet in enumerate(st.session_state.schedina):
    quota_totale_multipla *= bet["quota"]

    col_info, col_del = st.columns([5, 1])
    with col_info:
      st.markdown(f"**{idx + 1}. [{bet['competizione']}] {bet['partita']}**")
      st.write(
          f"🎯 Pronostico: **{bet['pronostico']}** | 📈 Quota: `{bet['quota']}`"
      )
    with col_del:
      if st.button("❌ Rimuovi", key=f"del_{idx}"):
        st.session_state.schedina.pop(idx)
        st.rerun()

    testo_telegram += (
        f"*{idx + 1}. {bet['partita']}*\n   └ Pronostico:"
        f" `{bet['pronostico']}` (Quota: `{bet['quota']}`)\n"
    )
    st.markdown("---")

  # Gestione importo collettivo della schedina
  st.markdown("### 💶 Gestione Puntata")
  importo_totale_giocata = st.number_input(
      "Importo Totale da Puntare sulla Schedina (€):",
      min_value=0.50,
      max_value=1000.0,
      value=5.00,
      step=0.50,
      key="global_stake",
  )

  vincita_totale_multipla = round(
      importo_totale_giocata * quota_totale_multipla, 2
  )

  testo_telegram += f"\n📈 *Quota Totale:* `{quota_totale_multipla:.2f}`"
  testo_telegram += f"\n💶 *Importo:* `€{importo_totale_giocata:.2f}`"
  testo_telegram += f"\n💰 *Vincita Potenziale:* `€{vincita_totale_multipla:.2f}`"

  # Riepilogo Economico
  col_ep1, col_ep2, col_ep3 = st.columns(3)
  col_ep1.metric(label="Spesa Schedina", value=f"€{importo_totale_giocata:.2f}")
  col_ep2.metric(label="Quota Totale", value=f"{quota_totale_multipla:.2f}")
  col_ep3.metric(
      label="Vincita Potenziale", value=f"€{vincita_totale_multipla:.2f}"
  )

  st.markdown("---")

  # Pulsante Ricevuta Stampabile
  if st.button("📄 Genera Ricevuta Stampabile", use_container_width=True):
    st.success(
        "Ricevuta generata qui sotto! Clicca sui tre puntini in alto a destra"
        " `⋮` di Streamlit e seleziona **Print** per stamparla o salvarla in"
        " PDF."
    )

    # Testo della ricevuta pulito
    ricevuta_txt = "========================================\n"
    ricevuta_txt += "            ⚽ LA MIA SCHEDINA ⚽          \n"
    ricevuta_txt += "========================================\n\n"
    for idx, bet in enumerate(st.session_state.schedina):
      ricevuta_txt += (
          f"{idx + 1}. [{bet['competizione']}] {bet['partita']}\n"
      )
      ricevuta_txt += (
          f"   Pronostico: {bet['pronostico']} | Quota: {bet['quota']}\n\n"
      )
    ricevuta_txt += "----------------------------------------\n"
    ricevuta_txt += f"Quota Totale Multipla: {quota_totale_multipla:.2f}\n"
    ricevuta_txt += f"Importo Puntato: €{importo_totale_giocata:.2f}\n"
    ricevuta_txt += f"Vincita Potenziale: €{vincita_totale_multipla:.2f}\n"
    ricevuta_txt += "========================================\n"

    st.code(ricevuta_txt, language="text")

  # Configurazione crediti Telegram
  with st.expander("🤖 Configurazione Bot Telegram"):
    telegram_token = st.text_input(
        "Token Bot Telegram",
        type="password",
        value="",
        placeholder="Es. 123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ",
    )
    telegram_chat_id = st.text_input(
        "Chat ID Telegram", value="", placeholder="Es. 987654321"
    )

  # Pulsanti di azione (Telegram e Svuota)
  col_btn1, col_btn2 = st.columns(2)

  with col_btn1:
    if st.button("📤 Invia a Telegram", use_container_width=True):
      if not telegram_token or not telegram_chat_id:
        st.warning(
            "⚠️ Inserisci prima il Token e la Chat ID nel menu 'Configurazione"
            " Bot Telegram'!"
        )
      else:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {
            "chat_id": telegram_chat_id,
            "text": testo_telegram,
            "parse_mode": "Markdown",
        }
        try:
          response = requests.post(url, json=payload)
          if response.status_code == 200:
            st.success("✅ Schedina inviata con successo su Telegram!")
          else:
            st.error(f"❌ Errore nell'invio: {response.text}")
        except Exception as e:
          st.error(f"❌ Connessione fallita: {e}")

  with col_btn2:
    if st.button("🗑️ Svuota Carrello", type="secondary", use_container_width=True):
      st.session_state.schedina = []
      st.rerun()