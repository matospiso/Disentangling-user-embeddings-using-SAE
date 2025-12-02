import html
import polars as pl
import streamlit as st

from demo_core import (
    load_artifacts,
    all_user_ids,
    all_neuron_labels,
    user_interactions_df,
    recommend_unsteered,
    recommend_steered,
)

# ---------- Cached state (models + ids/labels) ----------


@st.cache_resource
def get_state():
    # heavy stuff: load models, data, mappings
    artifacts = load_artifacts()

    user_ids = list(map(str, all_user_ids(artifacts)))
    neuron_labels = list(map(str, all_neuron_labels(artifacts)))

    return artifacts, user_ids, neuron_labels


# ---------- Rendering helpers ----------


def render_item_list(df: pl.DataFrame, title: str):
    st.subheader(title)

    if df is None or df.height == 0:
        st.caption("_No items to show._")
        return

    # CSS for horizontal scroller
    st.markdown(
        """
        <style>
        .scroll-container {
            display: flex;
            overflow-x: auto;
            padding: 12px 0;
            gap: 16px;
            scroll-behavior: smooth;
            white-space: nowrap;
        }
        .item-card {
            flex: 0 0 auto;
            width: 180px;
            text-align: left;
            white-space: normal;
        }
        .item-card img {
            width: 100%;
            border-radius: 6px;
        }
        .item-caption {
            font-size: 0.83rem;
            color: #ddd;
            line-height: 1.25rem;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cards = []

    for row in df.iter_rows(named=True):
        img = row.get("image", "")
        img = f"https://image.tmdb.org/t/p/w500/{img}.jpg" if img else ""

        # --- Title ---
        title_val = html.escape(row.get("title") or "")

        # --- Genres ---
        genres_val = html.escape(row.get("genres") or "")

        # --- Item ID ---
        item_id_val = row.get("item_id") or ""

        # --- Score formatting ---
        score_val = row.get("score")
        if score_val is not None:
            try:
                score_str = f"{float(score_val):.3f}"
            except Exception:
                score_str = str(score_val)
            score_html = f"<br><b>Score:</b> {score_str}"
        else:
            score_html = ""

        caption_html = f"<b>{title_val}</b>" f"<br><i>{genres_val}</i>" f"<br><b>ID:</b> {item_id_val}" f"{score_html}"

        img_html = f"<img src='{img}' />" if img else ""

        card_html = f"<div class='item-card'>" f"{img_html}" f"<div class='item-caption'>{caption_html}</div>" f"</div>"

        cards.append(card_html)

    all_cards_html = "<div class='scroll-container'>" + "".join(cards) + "</div>"
    st.markdown(all_cards_html, unsafe_allow_html=True)


# ---------- Main app ----------


def main():
    st.set_page_config(page_title="ELSA + SAE Demo", layout="wide")

    st.title("ELSA + SAE Neuron Steering Demo")

    st.write("Type a **user id** and a **neuron/tag name**, then compare:")
    st.markdown("- User’s interacted items\n" "- Original recommendations\n" "- Neuron-boosted recommendations")

    with st.spinner("Loading models and data…"):
        artifacts, user_ids, neuron_labels = get_state()

    # ---- initialise session_state for results ----
    if "results" not in st.session_state:
        st.session_state.results = {
            "user": None,
            "neuron": None,
            "interacted": None,
            "unsteered": None,
            "steered": None,
        }

    # ---------- Two side-by-side searchable selectboxes ----------

    user_col, neuron_col, alpha_col = st.columns([2, 2, 1])

    # default user id
    default_user_id = "113638"
    if default_user_id in user_ids:
        user_index = user_ids.index(default_user_id)
    else:
        user_index = None  # falls back to "no selection"

    with user_col:
        final_user = st.selectbox(
            "User ID (searchable)",
            options=user_ids,
            index=user_index,
            placeholder="Start typing a user ID…",
            key="user_select",
        )

    with neuron_col:
        final_neuron = st.selectbox(
            "Neuron/tag (searchable)",
            options=neuron_labels,
            index=None,
            placeholder="Start typing a neuron/tag…",
            key="neuron_select",
        )

    with alpha_col:
        alpha = st.slider(
            "Steering strength (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help="Increase to boost the selected tag.",
            key="alpha_slider",
        )

    # ---------- Trigger computation only when user is "done" ----------

    run_clicked = st.button("Show recommendations")

    if run_clicked and final_user and final_neuron:
        interacted = user_interactions_df(artifacts, final_user)
        unsteered = recommend_unsteered(artifacts, final_user)
        steered = recommend_steered(artifacts, final_user, final_neuron, alpha)

        st.session_state.results = {
            "user": final_user,
            "neuron": final_neuron,
            "interacted": interacted,
            "unsteered": unsteered,
            "steered": steered,
        }

    # ---------- Display last computed results (if any) ----------

    res = st.session_state.results
    if res["user"] is not None and res["neuron"] is not None:
        st.markdown("---")
        st.write(f"Showing results for **user {res['user']}** " f"and neuron/tag **{res['neuron']}**")

        render_item_list(res["interacted"], "1. User’s interacted items")
        render_item_list(res["unsteered"], "2. Original model predictions")
        render_item_list(res["steered"], "3. Neuron-boosted predictions")


if __name__ == "__main__":
    main()
