This folder will contain the trained Bi-LSTM model and tokenizer after running:

python src/train_model.py

It will produce:
- clickbait_bilstm.h5
- tokenizer.pkl

If these files exist, the Streamlit app will load them for inference; otherwise a small fallback model will be trained in-memory automatically.