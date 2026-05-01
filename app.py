import streamlit as st
import pandas as pd
import io
import time

from src.predict import predict_clickbait, get_sentiment, suggest_headlines, annotate_headline, generate_headlines, is_serious_headline
from src.batch_processor import load_file_to_df, process_batch
from src.url_processor import extract_headlines_from_url
from src.visualization import plot_clickbait_distribution, plot_sentiment_distribution, plot_confidence_histogram, plot_wordcloud
from src.explainability import highlight_important_words, top_words_from_batch

st.set_page_config(layout='wide', page_title='DeClickify')
st.title('DeClickify — Clickbait Detector & Sentiment Analyzer 🔎')

# Initialize session state keys
if 'headlines' not in st.session_state:
    st.session_state['headlines'] = []
if 'single_suggestions' not in st.session_state:
    st.session_state['single_suggestions'] = []
if 'url_headlines' not in st.session_state:
    st.session_state['url_headlines'] = []
if 'url_suggestions' not in st.session_state:
    st.session_state['url_suggestions'] = []
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = None
if 'batch_selected_suggestions' not in st.session_state:
    st.session_state['batch_selected_suggestions'] = []

# Debounce helper to avoid rapid repeated generator calls
def _allow_generation(namespace: str, cooldown: float = 2.5) -> bool:
    key = f'last_gen_{namespace}'
    now = time.time()
    last = st.session_state.get(key, 0)
    if now - last < cooldown:
        return False
    st.session_state[key] = now
    return True

menu = ['Single Headline', 'URL', 'Batch']
choice = st.sidebar.selectbox('Choose input type:', menu)

# ---- SINGLE HEADLINE ----
if choice == 'Single Headline':
    st.subheader('Analyze a single headline')
    text = st.text_input('Enter headline:')
    # show previous suggestions if any
    if st.session_state.get('single_suggestions'):
        st.markdown('**Suggested Headlines:**')
        for s in st.session_state['single_suggestions']:
            st.write('-', s)

    if st.button('Analyze Headline'):
        if not text or not text.strip():
            st.warning('Please enter a headline to analyze.')
        else:
            label, conf = predict_clickbait(text)
            sentiment, sent_score = get_sentiment(text)
            st.markdown(f"**Clickbait:** {label} — {conf:.2f}")
            st.markdown(f"**Sentiment:** {sentiment} — {sent_score:.2f}")
            st.markdown('**Highlighted Words:**')
            st.markdown(highlight_important_words(text), unsafe_allow_html=False)
            st.markdown('**Headline Suggestions:**')
            for s in suggest_headlines(text):
                st.write('-', s)


    if st.button('Generate Variations'):
        if not text or not text.strip():
            st.warning('Enter a headline to generate suggestions')
        else:
            # determine seriousness and whether AI is allowed (internal only)
            serious = is_serious_headline(text)
            allow_ai = (len(text.split()) < 15) and (not serious)
            # cooldown check
            if not _allow_generation('single'):
                st.info('Please wait a moment before generating again.')
            else:
                with st.spinner('Generating variations...'):
                    gens = generate_headlines(text, n=5, max_words=15, rewrite_only=True, allow_ai=allow_ai, force_news_style=serious)
                # gens will be a silent template fallback if AI fails or is disallowed
                st.session_state['single_suggestions'] = gens[:5]

    # show previous suggestions if any
    if st.session_state.get('single_suggestions'):
        st.markdown('**Suggested Headlines:**')
        for s in st.session_state['single_suggestions']:
            st.write('-', s)

# ---- URL MODE ----
elif choice == 'URL':
    st.subheader('Analyze headlines from a URL')
    url = st.text_input('Enter URL:')
    if st.button('Extract Headings'):
        if not url:
            st.warning('Enter a valid URL')
        else:
            headings = extract_headlines_from_url(url)
            st.session_state['url_headlines'] = headings
            st.success(f'Found {len(headings)} headings')
    if st.session_state.get('url_headlines'):
        sel = st.selectbox('Pick a headline to analyze', st.session_state['url_headlines'])
        if st.button('Analyze Selected Heading'):
            label, conf = predict_clickbait(sel)
            sent, sconf = get_sentiment(sel)
            st.markdown(f"**Clickbait:** {label} — {conf:.2f}")
            st.markdown(f"**Sentiment:** {sent} — {sconf:.2f}")
            st.markdown('**Highlighted Words:**')
            st.markdown(highlight_important_words(sel), unsafe_allow_html=False)
            st.markdown('**Headline Suggestions:**')
            for s in suggest_headlines(sel):
                st.write('-', s)


        # Generate variations for selected heading
        if st.button('Generate Variations for Selected Heading'):
            # determine seriousness and whether AI is allowed (internal only)
            serious = is_serious_headline(sel)
            allow_ai = (len(sel.split()) < 15) and (not serious)
            if not _allow_generation('url'):
                st.info('Please wait a moment before generating again.')
            else:
                with st.spinner('Generating variations...'):
                    gens = generate_headlines(sel, n=5, max_words=15, rewrite_only=True, allow_ai=allow_ai, force_news_style=serious)
                st.session_state['url_suggestions'] = gens[:5]
        if st.session_state.get('url_suggestions'):
            st.markdown('**Suggested Headlines:**')
            for s in st.session_state['url_suggestions']:
                st.write('-', s)


# ---- BATCH MODE ----
elif choice == 'Batch':
    st.subheader('Upload batch file (CSV, JSON, XML, XLSX, TXT, PDF)')
    uploaded_file = st.file_uploader('Upload file', type=['csv', 'json', 'xml', 'xlsx', 'txt', 'pdf'])
    if uploaded_file is not None:
        df, guessed, file_hash = load_file_to_df(uploaded_file)
        if df.empty:
            st.warning('Could not parse file, or file is empty')
        else:
            st.write('Preview:')
            st.dataframe(df.head())
            st.caption(f"File hash: {file_hash[:10]}...")
            col = st.selectbox('Select column containing headlines', options=df.columns.tolist(), index=0)
            if st.button('Process Batch'):
                with st.spinner('Processing...'):
                    results = process_batch(df, col)
                    st.session_state['batch_results'] = results
                    st.success('Batch processed')
    if st.session_state.get('batch_results') is not None:
        results = st.session_state['batch_results']
        st.subheader('Results')
        st.dataframe(results.head(200))

        # Allow analyzing a single headline from batch results
        pick_options = results['headline'].astype(str).tolist()
        if pick_options:
            pick = st.selectbox('Pick a headline from results to analyze', options=pick_options)
            if st.button('Analyze Selected from Results'):
                label, conf = predict_clickbait(pick)
                sent, sconf = get_sentiment(pick)
                st.markdown(f"**Clickbait:** {label} — {conf:.2f}")
                st.markdown(f"**Sentiment:** {sent} — {sconf:.2f}")
                st.markdown('**Highlighted Words:**')
                st.markdown(highlight_important_words(pick), unsafe_allow_html=False)
                st.markdown('**Headline Suggestions:**')
                for s in suggest_headlines(pick):
                    st.write('-', s)


            if st.button('Generate Variations for Selected Result'):
                serious = is_serious_headline(pick)
                allow_ai = (len(pick.split()) < 15) and (not serious)
                if not _allow_generation('batch'):
                    st.info('Please wait a moment before generating again.')
                else:
                    with st.spinner('Generating variations...'):
                        gens = generate_headlines(pick, n=5, max_words=15, rewrite_only=True, allow_ai=allow_ai, force_news_style=serious)
                    st.session_state['batch_selected_suggestions'] = gens[:5]
            if st.session_state.get('batch_selected_suggestions'):
                st.markdown('**Suggested Headlines:**')
                for s in st.session_state['batch_selected_suggestions']:
                    st.write('-', s)

        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button('Download CSV', csv, file_name='declickify_results.csv', mime='text/csv')
        # Excel
        towrite = io.BytesIO()
        results.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button('Download XLSX', towrite, file_name='declickify_results.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.subheader('Visualizations')
        fig1 = plot_clickbait_distribution(results)
        st.pyplot(fig1)
        fig2 = plot_sentiment_distribution(results)
        st.pyplot(fig2)
        fig3 = plot_confidence_histogram(results)
        st.pyplot(fig3)
        fig4 = plot_wordcloud(results)
        st.pyplot(fig4)
        st.subheader('Top words')
        top = top_words_from_batch(results['headline'].astype(str).tolist(), top_k=20)
        st.bar_chart(dict(top))

# Footer
st.markdown('---')
st.caption('DeClickify — clickbait detector + sentiment.')
