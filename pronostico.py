import os
from collections import Counter
from datetime import datetime
from math import exp, factorial
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="Pronosticatore Calcio & Gestionale Cassa",
    page_icon="⚽",
    layout="centered",
)

# File CSV per lo storico giocate
HISTORICAL_CSV = "storico_giocate.csv"

# ==================================================
# GESTIONE CSV STORICO E CASSA
# ==================================================
if not os.path.exists(HISTORICAL_CSV):
  df_init = pd.DataFrame(columns=[
      "ID",
      "Data",
      "Dettaglio_Giocata",
      "Quota_Totale",
      "Puntata",
      "Vincita_Potenziale",
      "Esito",
  ])
  df_init.to_csv(HISTORICAL_CSV, index=False)


def load_history():
  if os.path.exists(HISTORICAL_CSV):
    return pd.read_csv(HISTORICAL_CSV)
  return pd.DataFrame(columns=[
      "ID",
      "Data",
      "Dettaglio_Giocata",
      "Quota_Totale",
      "Puntata",
      "Vincita_Potenziale",
      "Esito",
  ])


def save_history(df_hist):
  df_hist.to_csv(HISTORICAL_CSV, index=False)


# ==================================================
# CONFIGURAZIONE COMPETIZIONI
# ==================================================
COMPETIZIONI = {"Serie A": "serieA.csv", "europa League": "europaleague.csv","serieB":"serieB.csv","premier league":"premierleague.csv"}

ELO_START = 1500
K = 20
HOME_ADVANTAGE = 70
SIMULATIONS = 5000

# Inizializzazione del carrello schedina nella memoria di sessione
if "schedina" not in st.session_state:
  st.session_state.schedina = []

# ==================================================
# TITOLO PRINCIPALE
# ==================================================
st.markdown(
    "<h1"
    " style='text-align: center; color: #FFFFFF;'>⚽ Pronostici Calcio &"
    " Schedina di Pietro Marzano</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align: center; color: #A0A0A0;'>Analisi "
    "<b>Over/Under</b>, <b>Goal/No Goal</b> e <b>Parziale/Finale</b></h3>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Menu principale a schede (Tabs principali)
tab1, tab2 = st.tabs(
    ["⚽ Pronosticatore & Schedina", "📊 Cassa & Storico Giocate"]
)

with tab1:
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

  try:
    df = load_data(CSV_FILE)
  except FileNotFoundError:
    st.error(
        f"⚠️ Attenzione: Il file '{CSV_FILE}' per la competizione"
        f" '{competizione_scelta}' non è stato trovato!"
    )
    st.stop()

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
        f"⚠️ Errore nel file '{CSV_FILE}': Colonna risultati non trovata."
    )
    st.stop()

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
  played["gol_trasferta"] = played[RESULT_COL].apply(
      lambda x: estrai_gol(x, 1)
  )

  played = played.dropna(subset=["gol_casa", "gol_trasferta"])
  played["gol_casa"] = played["gol_casa"].astype(int)
  played["gol_trasferta"] = played["gol_trasferta"].astype(int)

  # ELO e Statistiche
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

    hg, ag = row["gol_casa"], row["gol_trasferta"]
    ra, rb = ratings[home], ratings[away]

    ea = expected_score(ra + HOME_ADVANTAGE, rb)
    eb = expected_score(rb, ra + HOME_ADVANTAGE)

    sa, sb = (1, 0) if hg > ag else ((0, 1) if hg < ag else (0.5, 0.5))

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
        home_games["gol_casa"].mean()
        if len(home_games) > 0
        else avg_home_goals
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
        home_games["gol_casa"].mean()
        if len(home_games) > 0
        else avg_home_goals
    )

  def recent_form(team, n=5):
    recent = played[
        (played[HOME_COL] == team) | (played[AWAY_COL] == team)
    ].tail(n)
    points = 0
    for _, row in recent.iterrows():
      hg, ag = row["gol_casa"], row["gol_trasferta"]
      if row[HOME_COL] == team:
        points += 3 if hg > ag else (1 if hg == ag else 0)
      else:
        points += 3 if ag > hg else (1 if ag == hg else 0)
    return points

  def predict_match(home, away):
    form_home = recent_form(home)
    form_away = recent_form(away)
    form_factor = (form_home - form_away) / 15

    lambda_home = (attack_home[home] * defense_away[away]) / avg_home_goals
    lambda_away = (attack_away[away] * defense_home[home]) / avg_away_goals

    lambda_home *= 1 + form_factor * 0.15
    lambda_away *= 1 - form_factor * 0.15

    lambda_home = max(lambda_home, 0.20)
    lambda_away = max(lambda_away, 0.20)

    over25 = 0
    both_score = 0
    exact_scores = Counter()
    ht_ft_counts = Counter()

    for _ in range(SIMULATIONS):
      hg = np.random.poisson(lambda_home)
      ag = np.random.poisson(lambda_away)
      exact_scores[f"{hg}-{ag}"] += 1
      if hg + ag > 2:
        over25 += 1
      if hg > 0 and ag > 0:
        both_score += 1

      hg_ht = min(np.random.poisson(lambda_home * 0.44), hg)
      ag_ht = min(np.random.poisson(lambda_away * 0.44), ag)

      ht_res = "1" if hg_ht > ag_ht else ("2" if hg_ht < ag_ht else "X")
      ft_res = "1" if hg > ag else ("2" if hg < ag else "X")

      ht_ft_counts[f"{ht_res}/{ft_res}"] += 1

    p_over = round(over25 / SIMULATIONS * 100, 2)
    p_under = round(100 - p_over, 2)
    p_goal = round(both_score / SIMULATIONS * 100, 2)
    p_nogoal = round(100 - p_goal, 2)

    q_over = round(100 / p_over, 2) if p_over > 0 else 99.0
    q_under = round(100 / p_under, 2) if p_under > 0 else 99.0
    q_goal = round(100 / p_goal, 2) if p_goal > 0 else 99.0
    q_nogoal = round(100 / p_nogoal, 2) if p_nogoal > 0 else 99.0

    ht_ft_results = {}
    combo_keys = [
        "1/1",
        "1/X",
        "1/2",
        "X/1",
        "X/X",
        "X/2",
        "2/1",
        "2/X",
        "2/2",
    ]
    for combo in combo_keys:
      count = ht_ft_counts[combo]
      perc = round(count / SIMULATIONS * 100, 2)
      quota_teorica = round(100 / perc, 2) if perc > 0 else 99.0
      ht_ft_results[combo] = {"perc": perc, "quota": quota_teorica}

    return {
        "OVER25": p_over,
        "UNDER25": p_under,
        "GOAL": p_goal,
        "NOGOAL": p_nogoal,
        "Q_OVER25": q_over,
        "Q_UNDER25": q_under,
        "Q_GOAL": q_goal,
        "Q_NOGOAL": q_nogoal,
        "HT_FT": ht_ft_results,
        "xG_home": round(lambda_home, 2),
        "xG_away": round(lambda_away, 2),
        "total_xG": round(lambda_home + lambda_away, 2),
        "top_results": exact_scores.most_common(5),
    }

  # Selezione Squadre
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
      teams_casa = teams
    home_team = st.selectbox(
        "Seleziona Casa",
        teams_casa,
        key="home_select",
        label_visibility="collapsed",
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
    st.warning("⚠️ Seleziona due squadre diverse.")
  else:
    if st.button(
        "🔍 Analizza Partita", type="primary", use_container_width=True
    ):
      p = predict_match(home_team, away_team)
      st.session_state.current_analysis = {
          "competizione": competizione_scelta,
          "partita": f"{home_team} vs {away_team}",
          "p": p,
      }

    if (
        "current_analysis" in st.session_state
        and st.session_state.current_analysis["partita"]
        == f"{home_team} vs {away_team}"
    ):
      p = st.session_state.current_analysis["p"]

      # SUB-TABS PER ORGANIZZARE I MERCATI IN SCHEDE PULITE
      subtab1, subtab2, subtab3, subtab4 = st.tabs([
          "📊 Panoramica & xG",
          "⚽ Over / Under 2.5",
          "🥅 Goal / No Goal",
          "🔄 Parziale / Finale",
      ])

      # SUBTAB 1: PANORAMICA & xG
      with subtab1:
        st.markdown("#### ⚽ Gol Attesi (Expected Goals)")
        xg1, xg2, xg3 = st.columns(3)
        xg1.metric(label=f"xG {home_team}", value=p["xG_home"])
        xg2.metric(label=f"xG {away_team}", value=p["xG_away"])
        xg3.metric(label="Totale xG", value=p["total_xG"])

        st.markdown("---")
        st.markdown("#### 🎯 Top 5 Risultati Esatti Più Probabili")
        for score, count in p["top_results"]:
          perc = round(count / SIMULATIONS * 100, 2)
          q_exact = round(100 / perc, 2) if perc > 0 else 99.0
          st.write(
              f"• **Risultato {score}** → `{perc}%` | Quota teorica:"
              f" `{q_exact}`"
          )

      # SUBTAB 2: OVER / UNDER
      with subtab2:
        st.markdown("#### ⚽ Mercato Over / Under 2.5")
        ou1, ou2 = st.columns(2)
        ou1.metric(
            label="Over 2.5",
            value=f"{p['OVER25']}%",
            delta=f"Quota Minima: {p['Q_OVER25']}",
            delta_color="off",
        )
        ou2.metric(
            label="Under 2.5",
            value=f"{p['UNDER25']}%",
            delta=f"Quota Minima: {p['Q_UNDER25']}",
            delta_color="off",
        )

      # SUBTAB 3: GOAL / NO GOAL
      with subtab3:
        st.markdown("#### 🥅 Mercato Goal / No Goal")
        gg1, gg2 = st.columns(2)
        gg1.metric(
            label="Goal",
            value=f"{p['GOAL']}%",
            delta=f"Quota Minima: {p['Q_GOAL']}",
            delta_color="off",
        )
        gg2.metric(
            label="No Goal",
            value=f"{p['NOGOAL']}%",
            delta=f"Quota Minima: {p['Q_NOGOAL']}",
            delta_color="off",
        )

      # SUBTAB 4: PARZIALE / FINALE
      with subtab4:
        st.markdown("#### 🔄 Mercato Parziale / Finale (1T / Finale)")
        ht_ft = p["HT_FT"]

        col_hf1, col_hf2, col_hf3 = st.columns(3)
        with col_hf1:
          st.markdown("**Ribaltamenti Casa**")
          st.write(
              f"• **1/X**: `{ht_ft['1/X']['perc']}%` | Quota:"
              f" `{ht_ft['1/X']['quota']}`"
          )
          st.write(
              f"• **1/1**: `{ht_ft['1/1']['perc']}%` | Quota:"
              f" `{ht_ft['1/1']['quota']}`"
          )
          st.write(
              f"• **1/2**: `{ht_ft['1/2']['perc']}%` | Quota:"
              f" `{ht_ft['1/2']['quota']}`"
          )

        with col_hf2:
          st.markdown("**Ribaltamenti Trasferta**")
          st.write(
              f"• **2/X**: `{ht_ft['2/X']['perc']}%` | Quota:"
              f" `{ht_ft['2/X']['quota']}`"
          )
          st.write(
              f"• **2/2**: `{ht_ft['2/2']['perc']}%` | Quota:"
              f" `{ht_ft['2/2']['quota']}`"
          )
          st.write(
              f"• **2/1**: `{ht_ft['2/1']['perc']}%` | Quota:"
              f" `{ht_ft['2/1']['quota']}`"
          )

        with col_hf3:
          st.markdown("**Pareggio 1° Tempo**")
          st.write(
              f"• **X/X**: `{ht_ft['X/X']['perc']}%` | Quota:"
              f" `{ht_ft['X/X']['quota']}`"
          )
          st.write(
              f"• **X/1**: `{ht_ft['X/1']['perc']}%` | Quota:"
              f" `{ht_ft['X/1']['quota']}`"
          )
          st.write(
              f"• **X/2**: `{ht_ft['X/2']['perc']}%` | Quota:"
              f" `{ht_ft['X/2']['quota']}`"
          )

      st.markdown("---")
      st.markdown("### 🛒 Aggiungi alla Schedina & Verifica Valore")

      mercati_disponibili = [
          "Over 2.5",
          "Under 2.5",
          "Goal",
          "No Goal",
          "1/X (Parz/Fin)",
          "2/X (Parz/Fin)",
          "1/2 (Parz/Fin)",
          "2/1 (Parz/Fin)",
          "1/1 (Parz/Fin)",
          "2/2 (Parz/Fin)",
          "X/X (Parz/Fin)",
          "X/1 (Parz/Fin)",
          "X/2 (Parz/Fin)",
      ]

      col_mercato, col_quota = st.columns(2)
      with col_mercato:
        mercato_scelto = st.selectbox(
            "Seleziona Pronostico", mercati_disponibili, key="sel_mercato"
        )
      with col_quota:
        quota_inserita = st.number_input(
            "Quota Offerta dal Bookmaker",
            min_value=1.01,
            max_value=100.0,
            value=2.00,
            step=0.01,
            key="sel_quota",
        )

      map_q = {
          "Over 2.5": p["Q_OVER25"],
          "Under 2.5": p["Q_UNDER25"],
          "Goal": p["Q_GOAL"],
          "No Goal": p["Q_NOGOAL"],
          "1/X (Parz/Fin)": ht_ft["1/X"]["quota"],
          "2/X (Parz/Fin)": ht_ft["2/X"]["quota"],
          "1/2 (Parz/Fin)": ht_ft["1/2"]["quota"],
          "2/1 (Parz/Fin)": ht_ft["2/1"]["quota"],
          "1/1 (Parz/Fin)": ht_ft["1/1"]["quota"],
          "2/2 (Parz/Fin)": ht_ft["2/2"]["quota"],
          "X/X (Parz/Fin)": ht_ft["X/X"]["quota"],
          "X/1 (Parz/Fin)": ht_ft["X/1"]["quota"],
          "X/2 (Parz/Fin)": ht_ft["X/2"]["quota"],
      }
      q_teorica_scelta = map_q[mercato_scelto]

      if quota_inserita > q_teorica_scelta:
        st.success(
            f"🟢 **VALORE TROVATO (+EV)!** La quota bookmaker ({quota_inserita})"
            f" è SUPERIORE alla quota teorica minima ({q_teorica_scelta})."
        )
      elif quota_inserita == q_teorica_scelta:
        st.info(
            f"⚖️ **QUOTA IN LINEA:** La quota bookmaker ({quota_inserita}) è"
            f" identica alla quota teorica ({q_teorica_scelta})."
        )
      else:
        st.error(
            f"🔴 **NESSUN VALORE (-EV):** La quota bookmaker ({quota_inserita})"
            f" è TROPPO BASSA rispetto alla quota teorica minima"
            f" ({q_teorica_scelta}). Giocata sconsigliata!"
        )

      if st.button(
          "➕ Inserisci nel Carrello Schedina", use_container_width=True
      ):
        item = {
            "competizione": competizione_scelta,
            "partita": f"{home_team} vs {away_team}",
            "pronostico": mercato_scelto,
            "quota": quota_inserita,
        }
        st.session_state.schedina.append(item)
        st.success("✅ Partita aggiunta con successo alla schedina!")

  # Carrello Schedina
  st.markdown("---")
  st.subheader("🛒 Carrello Schedina & Riepilogo Giocate")

  if len(st.session_state.schedina) == 0:
    st.info("ℹ️ Il carrello è vuoto.")
  else:
    for idx, bet in enumerate(st.session_state.schedina):
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
      st.markdown("---")

    st.markdown("### 💶 Gestione Tipologia di Giocata & Puntata")

    tipo_giocata = st.radio(
        "Come vuoi gestire la puntata?",
        [
            "Multipla (Puntata unica sul totale)",
            "Singole (Puntata su ciascuna partita)",
        ],
        horizontal=True,
    )

    importo_input = st.number_input(
        (
            "Importo per la Multipla (€):"
            if "Multipla" in tipo_giocata
            else "Importo da puntare su CIASCUNA Singola (€):"
        ),
        min_value=0.50,
        max_value=1000.0,
        value=5.00,
        step=0.50,
        key="global_stake",
    )

    num_partite = len(st.session_state.schedina)
    testo_telegram = ""

    if "Multipla" in tipo_giocata:
      quota_totale_multipla = 1.0
      for bet in st.session_state.schedina:
        quota_totale_multipla *= bet["quota"]

      spesa_totale = importo_input
      vincita_potenziale_totale = round(spesa_totale * quota_totale_multipla, 2)

      testo_telegram = "⚽ *LA MIA SCHEDINA MULTIPLA* ⚽\n\n"
      for idx, bet in enumerate(st.session_state.schedina):
        testo_telegram += (
            f"*{idx + 1}. {bet['partita']}*\n   └ Pronostico:"
            f" `{bet['pronostico']}` (Quota: `{bet['quota']}`)\n"
        )
      testo_telegram += f"\n📈 *Quota Totale:* `{quota_totale_multipla:.2f}`"
      testo_telegram += f"\n💶 *Importo Puntato:* `€{spesa_totale:.2f}`"
      testo_telegram += (
          f"\n💰 *Vincita Potenziale:* `€{vincita_potenziale_totale:.2f}`"
      )

      col_ep1, col_ep2, col_ep3 = st.columns(3)
      col_ep1.metric(label="Spesa Totale", value=f"€{spesa_totale:.2f}")
      col_ep2.metric(label="Quota Totale Multipla", value=f"{quota_totale_multipla:.2f}")
      col_ep3.metric(
          label="Vincita Potenziale", value=f"€{vincita_potenziale_totale:.2f}"
      )

    else:
      spesa_totale = importo_input * num_partite
      vincita_potenziale_totale = sum(
          round(importo_input * bet["quota"], 2)
          for bet in st.session_state.schedina
      )

      testo_telegram = f"⚽ *LE MIE GIOCATE SINGOLE ({num_partite})* ⚽\n\n"
      for idx, bet in enumerate(st.session_state.schedina):
        v_pot = round(importo_input * bet["quota"], 2)
        testo_telegram += (
            f"*{idx + 1}. {bet['partita']}*\n   └ Pronostico:"
            f" `{bet['pronostico']}` | Quota: `{bet['quota']}` | Puntata:"
            f" `€{importo_input:.2f}` | Vincita: `€{v_pot:.2f}`\n"
        )
      testo_telegram += f"\n💶 *Spesa Totale ({num_partite} singole):* `€{spesa_totale:.2f}`"
      testo_telegram += f"\n💰 *Vincita Potenziale Max:* `€{vincita_potenziale_totale:.2f}`"

      col_ep1, col_ep2, col_ep3 = st.columns(3)
      col_ep1.metric(
          label="Spesa Totale",
          value=f"€{spesa_totale:.2f}",
          delta=f"{num_partite} x €{importo_input:.2f}",
          delta_color="off",
      )
      col_ep2.metric(label="Numero Singole", value=f"{num_partite}")
      col_ep3.metric(
          label="Vincita Potenziale Max",
          value=f"€{vincita_potenziale_totale:.2f}",
      )

    st.markdown("---")

    if st.button(
        "💾 Salva e Registra Schedina nello Storico CSV",
        type="primary",
        use_container_width=True,
    ):
      df_h = load_history()
      data_ora = datetime.now().strftime("%Y-%m-%d %H:%M")

      if "Multipla" in tipo_giocata:
        nuovo_id = len(df_h) + 1
        dettaglio_multipla = " | ".join(
            f"{b['partita']} ({b['pronostico']} @ {b['quota']})"
            for b in st.session_state.schedina
        )

        nuova_riga = pd.DataFrame([{
            "ID": nuovo_id,
            "Data": data_ora,
            "Dettaglio_Giocata": f"[MULTIPLE] {dettaglio_multipla}",
            "Quota_Totale": round(quota_totale_multipla, 2),
            "Puntata": spesa_totale,
            "Vincita_Potenziale": vincita_potenziale_totale,
            "Esito": "In corso",
        }])
        df_h = pd.concat([df_h, nuova_riga], ignore_index=True)

      else:
        nuove_righe = []
        for bet in st.session_state.schedina:
          nuovo_id = len(df_h) + len(nuove_righe) + 1
          vincita_singola = round(importo_input * bet["quota"], 2)
          nuove_righe.append({
              "ID": nuovo_id,
              "Data": data_ora,
              "Dettaglio_Giocata": (
                  f"[{bet['competizione']}] {bet['partita']} -"
                  f" {bet['pronostico']}"
              ),
              "Quota_Totale": bet["quota"],
              "Puntata": importo_input,
              "Vincita_Potenziale": vincita_singola,
              "Esito": "In corso",
          })
        df_h = pd.concat([df_h, pd.DataFrame(nuove_righe)], ignore_index=True)

      save_history(df_h)
      st.session_state.schedina = []
      st.success("✅ Giocate registrate nello storico CSV! Carrello svuotato.")
      st.rerun()

    with st.expander("🤖 Configurazione Bot Telegram"):
      telegram_token = st.text_input("Token Bot Telegram", type="password")
      telegram_chat_id = st.text_input("Chat ID Telegram")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
      if st.button("📤 Invia a Telegram", use_container_width=True):
        if not telegram_token or not telegram_chat_id:
          st.warning("⚠️ Inserisci prima Token e Chat ID!")
        else:
          url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
          payload = {
              "chat_id": telegram_chat_id,
              "text": testo_telegram,
              "parse_mode": "Markdown",
          }
          try:
            res = requests.post(url, json=payload)
            if res.status_code == 200:
              st.success("✅ Inviato su Telegram!")
            else:
              st.error(f"❌ Errore: {res.text}")
          except Exception as e:
            st.error(f"❌ Connessione fallita: {e}")

    with col_btn2:
      if st.button(
          "🗑️ Svuota Carrello", type="secondary", use_container_width=True
      ):
        st.session_state.schedina = []
        st.rerun()

# ==================================================
# TAB 2: CASSA & STORICO GIOCATE
# ==================================================
with tab2:
  st.subheader("💰 Gestionale Cassa & Profilo Profitti / Perdite")

  cassa_iniziale = st.number_input(
      "Bankroll / Cassa Iniziale (€):",
      min_value=10.0,
      max_value=10000.0,
      value=100.0,
      step=10.0,
  )

  df_history = load_history()

  if df_history.empty:
    st.info("ℹ️ Nessuna scommessa salvata nello storico CSV.")
  else:
    spesa_totale = df_history["Puntata"].sum()
    vinte = df_history[df_history["Esito"] == "Vinta"]
    incasso_totale = vinte["Vincita_Potenziale"].sum()

    profitto_netto = incasso_totale - spesa_totale
    cassa_attuale = cassa_iniziale + profitto_netto
    yield_perc = (
        round((profitto_netto / spesa_totale) * 100, 2)
        if spesa_totale > 0
        else 0.0
    )

    totale_chiuse = len(
        df_history[df_history["Esito"].isin(["Vinta", "Persa"])]
    )
    win_rate = (
        round((len(vinte) / totale_chiuse) * 100, 1) if totale_chiuse > 0 else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cassa Attuale", f"€{cassa_attuale:.2f}")
    c2.metric("Profitto / Perdita Netto", f"€{profitto_netto:.2f}")
    c3.metric("Yield / ROI", f"{yield_perc}%")
    c4.metric("Win Rate", f"{win_rate}%")

    st.markdown("---")
    st.markdown("### 📑 Gestisci le Giocate nello Storico CSV")

    for i, row in df_history.iterrows():
      with st.expander(
          f"Giocata #{row['ID']} - {row['Data']} | Quota:"
          f" {row['Quota_Totale']} | Puntata: €{row['Puntata']} | Status:"
          f" {row['Esito']}"
      ):
        st.write(f"**Dettaglio:** {row['Dettaglio_Giocata']}")
        st.write(f"**Vincita Potenziale:** €{row['Vincita_Potenziale']}")

        c_status, c_del = st.columns([3, 1])

        with c_status:
          nuovo_esito = st.selectbox(
              "Aggiorna Esito",
              ["In corso", "Vinta", "Persa"],
              index=["In corso", "Vinta", "Persa"].index(row["Esito"]),
              key=f"esito_{row['ID']}",
          )
          if nuovo_esito != row["Esito"]:
            df_history.at[i, "Esito"] = nuovo_esito
            save_history(df_history)
            st.success("Esito aggiornato!")
            st.rerun()

        with c_del:
          if st.button("🗑️ Elimina", key=f"del_hist_{row['ID']}"):
            df_history = df_history.drop(i).reset_index(drop=True)
            save_history(df_history)
            st.warning("Giocata rimossa dallo storico!")
            st.rerun()

    st.markdown("---")
    if st.button("⚠️ Svuota Interamente lo Storico CSV"):
      save_history(
          pd.DataFrame(columns=[
              "ID",
              "Data",
              "Dettaglio_Giocata",
              "Quota_Totale",
              "Puntata",
              "Vincita_Potenziale",
              "Esito",
          ])
      )
      st.success("Storico CSV azzerato!")
      st.rerun()