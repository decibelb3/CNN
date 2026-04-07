import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os  

# Define attack categories (Maintains the 2, 6, and 19-class logic)
ATTACK_CATEGORIES_19 = { 
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_6 = {  
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT',
    'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT',
    'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon',
    'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon',
    'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS',
    'DDoS-SYN': 'DDoS',
    'DDoS-TCP': 'DDoS',
    'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS',
    'DoS-SYN': 'DoS',
    'DoS-TCP': 'DoS',
    'DoS-UDP': 'DoS',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_2 = {  
    'ARP_Spoofing': 'attack',
    'MQTT-DDoS-Connect_Flood': 'attack',
    'MQTT-DDoS-Publish_Flood': 'attack',
    'MQTT-DoS-Connect_Flood': 'attack',
    'MQTT-DoS-Publish_Flood': 'attack',
    'MQTT-Malformed_Data': 'attack',
    'Recon-OS_Scan': 'attack',
    'Recon-Ping_Sweep': 'attack',
    'Recon-Port_Scan': 'attack',
    'Recon-VulScan': 'attack',
    'TCP_IP-DDoS-ICMP': 'attack',
    'TCP_IP-DDoS-SYN': 'attack',
    'TCP_IP-DDoS-TCP': 'attack',
    'TCP_IP-DDoS-UDP': 'attack',
    'TCP_IP-DoS-ICMP': 'attack',
    'TCP_IP-DoS-SYN': 'attack',
    'TCP_IP-DoS-TCP': 'attack',
    'TCP_IP-DoS-UDP': 'attack',
    'Benign': 'Benign'
}

def get_attack_category(file_name, class_config): 
    """Maps file names to their respective attack labels."""
    if class_config == 2:
        categories = ATTACK_CATEGORIES_2
    elif class_config == 6:
        categories = ATTACK_CATEGORIES_6
    else:
        categories = ATTACK_CATEGORIES_19  

    for key in categories:
        if key in file_name:
            return categories[key]
    return "Unknown"

def inject_adversarial_noise(df, noise_level=0.10):
    """TASK 3: Injects Gaussian noise to simulate malformed data attacks."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        std = df[col].std()
        noise = np.random.normal(0, std * noise_level, size=len(df))
        df[col] = df[col] + noise
    print(f"DEBUG: Adversarial noise ({noise_level*100}%) injected into {len(numeric_cols)} features.")
    return df

def load_and_preprocess_data(data_dir, class_config, sample_fraction=0.1):
    """Loads sampled 'Toy' sets, applies noise, and reshapes data for the 1D CNN."""
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    
    train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.csv')]
    test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.csv')]

    # Step 1: Create sampled 'Toy' datasets to manage memory and speed [cite: 38]
    train_df = pd.concat([
        pd.read_csv(f, dtype=object).assign(file=f).sample(frac=sample_fraction, random_state=42) 
        for f in train_files
    ], ignore_index=True)
    
    test_df = pd.concat([
        pd.read_csv(f, dtype=object).assign(file=f).sample(frac=sample_fraction, random_state=42) 
        for f in test_files
    ], ignore_index=True)

    # Step 2: Convert objects to numeric for mathematical operations
    cols = train_df.columns.drop('file')
    train_df[cols] = train_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    test_df[cols] = test_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Step 3: TASK 3 - Inject noise into the test set only to test robustness [cite: 42]
    test_df = inject_adversarial_noise(test_df, noise_level=0.50) 

    # Step 4: Label Assignment
    train_df['Attack_Type'] = train_df['file'].apply(lambda x: get_attack_category(x, class_config))
    test_df['Attack_Type'] = test_df['file'].apply(lambda x: get_attack_category(x, class_config))

    X_train = train_df.drop(['Attack_Type', 'file'], axis=1)
    y_train = train_df['Attack_Type']
    X_test = test_df.drop(['Attack_Type', 'file'], axis=1)
    y_test = test_df['Attack_Type']

    # Step 5: Encoding and Train/Val Split
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical = to_categorical(y_test_encoded)

    X_train, X_val, y_train_categorical, y_val_categorical = train_test_split(
        X_train, y_train_categorical, test_size=0.2, random_state=42
    )

    # Step 6: Scaling and Reshaping for 1D CNN Architecture
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Step 7: Final Return for main.py unpacking
    return X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder
