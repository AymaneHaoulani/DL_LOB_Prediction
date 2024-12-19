
# --- Feature Engineering ---

def add_features(data, k_levels=[5, 10], rolling_windows=[10, 20, 50, 100]):

    # --- Order Book Aggregation Features ---

    data['Mid Price'] = (data['Ask Price 1'] + data['Bid Price 1']) / 2
    data['Relative Spread'] = (data['Ask Price 1'] - data['Bid Price 1']) / data['Mid Price']

    bid_volume_columns = [f'Bid Volume {i}' for i in range(1, 11)]
    ask_volume_columns = [f'Ask Volume {i}' for i in range(1, 11)]
    bid_price_columns = [f'Bid Price {i}' for i in range(1, 11)]
    ask_price_columns = [f'Ask Price {i}' for i in range(1, 11)]

    for k in k_levels:
        data[f'Cumulative Bid Volume {k}'] = data[bid_volume_columns[:k]].sum(axis=1)
        data[f'Cumulative Ask Volume {k}'] = data[ask_volume_columns[:k]].sum(axis=1)
        weighted_sum = (
            data[bid_price_columns[:k]].mul(data[bid_volume_columns[:k]]).sum(axis=1) +
            data[ask_price_columns[:k]].mul(data[ask_volume_columns[:k]]).sum(axis=1)
        )
        total_volume = (
            data[bid_volume_columns[:k]].sum(axis=1) + data[ask_volume_columns[:k]].sum(axis=1)
        )
        data[f'Weighted Price {k}'] = weighted_sum / total_volume

    # Compute Volume Imbalance for k = 1, 3, 5, 10.
    for k in [1, 3, 5, 10]:
        data[f'Volume Imbalance {k}'] = (
            (data[bid_volume_columns[:k]].sum(axis=1) - data[ask_volume_columns[:k]].sum(axis=1)) /
            (data[bid_volume_columns[:k]].sum(axis=1) + data[ask_volume_columns[:k]].sum(axis=1))
        )

    # --- Temporal Features ---

    for T in rolling_windows:
        data[f'Rolling Std {T}'] = data['Mid Price'].rolling(window=T).std()
        data[f'Rolling Mean {T}'] = data['Mid Price'].rolling(window=T).mean()

    data = data.dropna()

    return data
