import os
import telebot
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import ta
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate
from transformers import pipeline

# --- CONFIGURAZIONE ---
TELEGRAM_TOKEN = "IL_TUO_TELEGRAM_TOKEN"
NEWS_API_KEY = "LA_TUA_NEWS_API_KEY"

# Inizializzazione bot e IA
bot = telebot.TeleBot(TELEGRAM_TOKEN)
print("Caricamento FinBERT...")
sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")

# Dizionario per mappare i nomi che scrivi tu ai ticker di Yahoo/TradingView
SYMBOLS = {
    "EURUSD": {"ticker": "EURUSD=X", "name": "Euro/Dollaro", "news": "Euro Dollar"},
    "GBPUSD": {"ticker": "GBPUSD=X", "name": "Sterlina/Dollaro", "news": "British Pound Dollar"},
    "USDJPY": {"ticker": "USDJPY=X", "name": "Dollaro/Yen", "news": "US Dollar Yen"},
    "XAUUSD": {"ticker": "GC=F", "name": "Oro", "news": "Gold Price"},
    "XAGUSD": {"ticker": "SI=F", "name": "Argento", "news": "Silver Price"},
    "BTCUSD": {"ticker": "BTC-USD", "name": "Bitcoin", "news": "Bitcoin"},
    "SPX500": {"ticker": "^GSPC", "name": "S&P 500", "news": "S&P 500"},
    "NAS100": {"ticker": "^IXIC", "name": "NASDAQ 100", "news": "NASDAQ"},
    "US30":   {"ticker": "^DJI", "name": "Dow Jones", "news": "Dow Jones"}
}

def get_sentiment_score(ticker_name):
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker_name}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
        response = requests.get(url).json()
        if response.get('status') != 'ok' or not response.get('articles'): return 0
        score = 0
        for art in response['articles']:
            res = sentiment_pipeline(art['title'])[0]
            if res['label'] == 'positive': score += res['score']
            elif res['label'] == 'negative': score -= res['score']
        return score / len(response['articles'])
    except: return 0

def analyze_asset(symbol_key):
    info = SYMBOLS[symbol_key]
    try:
        df = yf.download(info['ticker'], period="60d", interval="1h", progress=False, multi_level_index=False)
        if len(df) < 100: return None
    except: return None

    atr = ta.volatility.average_true_range(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze(), window=14)
    atr_current = float(atr.iloc[-1])
    
    df_feat = pd.DataFrame(index=df.index)
    df_feat['Close'] = df['Close'].squeeze()
    df_feat['RSI'] = ta.momentum.rsi(df_feat['Close'], window=14)
    df_feat['MACD'] = ta.trend.MACD(df_feat['Close']).macd_diff()
    df_feat['Volume'] = df['Volume'].squeeze()
    df_feat['ATR'] = atr
    df_feat = df_feat.dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_feat)
    
    X, y = [], []
    for i in range(len(scaled_data) - 60):
        X.append(scaled_data[i:(i + 60)])
        y.append(scaled_data[i + 60, 0])
    X, y = np.array(X), np.array(y)

    input_layer = Input(shape=(X.shape[1], X.shape[2]))
    merged = concatenate([LSTM(32)(input_layer), LSTM(32)(input_layer), LSTM(32)(Dropout(0.2)(input_layer))])
    model = Model(inputs=input_layer, outputs=Dense(1)(merged))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_60 = scaled_data[-60:]
    pred_scaled = model.predict(np.reshape(last_60, (1, 60, X.shape[2])), verbose=0)
    dummy = np.zeros((1, X.shape[2]))
    dummy[0, 0] = pred_scaled[0, 0]
    pred_price = float(scaler.inverse_transform(dummy)[0, 0])
    current_price = float(df_feat['Close'].iloc[-1])
    
    sentiment = get_sentiment_score(info['news'])
    price_direction = 1 if pred_price > current_price else -1
    
    return {
        "name": info['name'], "current": current_price, "pred": pred_price,
        "sentiment": sentiment, "direction": price_direction, "atr": atr_current
    }

# --- COMANDI TELEGRAM ---
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "👋 Ciao! Sono il tuo AI Consultant.\nScrivimi il segnale che vedi sul grafico.\nEsempio: `/check XAUUSD LONG` oppure `/check EURUSD SHORT`")

@bot.message_handler(commands=['check'])
def handle_check(message):
    try:
        # Estrae simbolo e direzione dal messaggio
        parts = message.text.split()
        if len(parts) < 3:
            bot.reply_to(message, "⚠️ Formato errato. Usa: `/check SIMBOLO DIREZIONE`\nEs: `/check XAUUSD LONG`")
            return
            
        symbol = parts[1].upper()
        user_direction = parts[2].upper() # LONG o SHORT
        
        if symbol not in SYMBOLS:
            bot.reply_to(message, f"❌ Simbolo {symbol} non riconosciuto. Usa uno di questi: {', '.join(SYMBOLS.keys())}")
            return
            
        bot.reply_to(message, f"🔍 Analisi di {SYMBOLS[symbol]['name']} in corso...")
        
        res = analyze_asset(symbol)
        if not res:
            bot.reply_to(message, "❌ Errore nel recupero dei dati. Riprova.")
            return
            
        ai_dir = "LONG" if res['direction'] == 1 else "SHORT"
        
        # Calcolo SL e TP per la direzione richiesta
        if user_direction == "LONG":
            sl = res['current'] - (res['atr'] * 2)
            tp = res['current'] + (res['atr'] * 4)
        else:
            sl = res['current'] + (res['atr'] * 2)
            tp = res['current'] - (res['atr'] * 4)
            
        # Costruzione risposta
        msg = f"📊 <b>{res['name']}</b>\n"
        msg += f"💰 Prezzo attuale: {res['current']:.5f}\n"
        msg += f"📰 Sentiment Notizie: {res['sentiment']:.2f}\n"
        msg += f"🤖 LSTM dice: <b>{ai_dir}</b>\n\n"
        
        # Logica di conferma
        if user_direction == ai_dir:
            if abs(res['sentiment']) > 0.10:
                msg += f"✅ <b>SEGNALE CONFERMATO!</b>\nPrevisione prezzi e Notizie sono allineate.\n\n"
                msg += f"🛑 <b>Stop Loss:</b> {sl:.5f}\n"
                msg += f"✅ <b>Take Profit:</b> {tp:.5f}\n"
                msg += f"⚖️ <b>R:R:</b> 1:2"
            else:
                msg += f"⚠️ <b>CONFERMA DEBOLE</b>\nI prezzi confermano, ma le notizie sono incerte.\nRiduci la size del 50%."
        else:
            msg += f"🛑 <b>ATTENZIONE! CONFLITTO!</b>\nTu vuoi fare {user_direction} ma l'IA dice {ai_dir}.\n"
            msg += f"❌ <b>NON ENTRARE.</b> Aspetta un setup migliore."
            
        bot.reply_to(message, msg, parse_mode='HTML')
        
    except Exception as e:
        bot.reply_to(message, f"Errore: {e}")

print("Bot in ascolto...")
bot.polling()
