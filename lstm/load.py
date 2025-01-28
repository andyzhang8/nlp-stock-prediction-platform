import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_sequences(features, labels, sequence_length):
    xs, ys = [], []
    for i in range(len(features) - sequence_length):
        x = features[i : i + sequence_length]
        y = labels[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_all_data(
    directory_path,
    sequence_length=48,
    test_size=0.2,
    random_state=42
):
    all_dfs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            csv_path = os.path.join(directory_path, filename)
            df = pd.read_csv(csv_path)
            if all(col in df.columns for col in ['date','open','high','low','close','volume']):
                all_dfs.append(df)

    big_df = pd.concat(all_dfs, ignore_index=True)

    big_df['date'] = pd.to_datetime(big_df['date'], utc=True)
    big_df['date'] = big_df['date'].dt.tz_convert(None)

    big_df = big_df.sort_values('date', ascending=True)
    big_df.reset_index(drop=True, inplace=True)


    big_df['next_close'] = big_df['close'].shift(-1)
    big_df.dropna(inplace=True)
    big_df['label'] = (big_df['next_close'] > big_df['close']).astype(int)

    big_df['ma5'] = big_df['close'].rolling(window=5).mean()
    big_df['ma20'] = big_df['close'].rolling(window=20).mean()
    big_df['std20'] = big_df['close'].rolling(window=20).std()
    big_df['bol_upper'] = big_df['ma20'] + (2 * big_df['std20'])
    big_df['bol_lower'] = big_df['ma20'] - (2 * big_df['std20'])

    big_df['close_lag1'] = big_df['close'].shift(1)
    big_df['close_lag2'] = big_df['close'].shift(2)
    big_df['close_lag3'] = big_df['close'].shift(3)

    big_df.dropna(inplace=True)

    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'ma5', 'ma20', 'std20', 'bol_upper', 'bol_lower',
        'close_lag1', 'close_lag2', 'close_lag3'
    ]
    features = big_df[feature_cols].values
    labels = big_df['label'].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X, y = create_sequences(features, labels, sequence_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
