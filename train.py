# The Anthill AI+ Evaluation Tool
# ONE-TIME TRAINING SCRIPT

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from transformers import BertTokenizer, BertModel
from faker import Faker
import random
import warnings
import pickle

warnings.filterwarnings("ignore")
fake = Faker()
print("Anthill AI+ Tool Initialized for Training.")
print(f"Using PyTorch version: {torch.__version__}")
print("-" * 50)

def create_startup_dataset(num_samples=500):
    data = []
    sectors = [
        'AI & Big Data', 'AgriTech', 'CleanTech & EV', 'Consumer Services', 'Cybersecurity',
        'Deep Tech & Robotics', 'DevTools', 'E-commerce & D2C', 'EdTech', 'FinTech',
        'Gaming', 'GovTech & Social Enterprise', 'HealthTech', 'Logistics & Supply Chain',
        'Media & Entertainment', 'PropTech', 'SaaS', 'Travel & Hospitality'
    ]
    locations = ['Metro', 'Tier-2/3']
    founder_colleges = ['an IIT', 'an IIM', 'BITS Pilani', 'a top US university', 'a state university']
    previous_employers = ['Google', 'Microsoft', 'a unicorn startup', 'a major consulting firm', 'another startup']

    for i in range(num_samples):
        founded_year = random.randint(2015, 2024)
        age = 2025 - founded_year
        total_funding = random.uniform(50000, 50e6) if age > 0 else 0
        num_investors = random.randint(0, 15)
        team_size = random.randint(1, 500)
        
        success_prob = 0.1 + (total_funding / 1e7) + (age / 20) - (random.uniform(0, 0.3))
        status = 1 if random.random() < success_prob else 0
        next_valuation = total_funding * random.uniform(2, 5) * (1 + success_prob) if status == 1 else total_funding * random.uniform(0.5, 1.5)
        
        founder_bio = f"{fake.name()}, a graduate from {random.choice(founder_colleges)}, previously worked at {random.choice(previous_employers)}. With {random.randint(2, 15)} years of experience, they are passionate about solving problems in the {random.choice(sectors)} space."
        product_desc = f"A revolutionary {random.choice(['AI-powered', 'blockchain-based', 'sustainable'])} platform for the {random.choice(sectors)} sector in India. Our solution leverages {random.choice(['deep learning', 'big data analytics'])} to optimize {random.choice(['supply chains', 'customer engagement', 'unit economics'])}."
        monthly_web_traffic = np.sort(np.random.randint(1000, 100000, 12).astype(float)) * (1 + i/num_samples)

        data.append({
            'founded_year': founded_year, 'age': age, 'total_funding_usd': total_funding,
            'num_investors': num_investors, 'team_size': team_size, 'sector': random.choice(sectors),
            'location': random.choice(locations), 'is_dpiit_recognized': random.choice([0, 1]),
            'founder_bio': founder_bio, 'product_desc': product_desc,
            'monthly_web_traffic': monthly_web_traffic,
            'status_success': status, 'predicted_next_valuation': next_valuation,
        })
    
    return pd.DataFrame(data).sort_values('founded_year').reset_index(drop=True)

print("Simulating multi-modal startup dataset...")
startup_df = create_startup_dataset()
print("-" * 50)

def feature_engineer(df):
    df['funding_per_investor'] = df['total_funding_usd'] / (df['num_investors'] + 1e-6)
    df['funding_per_employee'] = df['total_funding_usd'] / (df['team_size'] + 1e-6)
    df['founder_has_iit_iim_exp'] = df['founder_bio'].str.contains('IIT|IIM').astype(int)
    df['avg_web_traffic'] = df['monthly_web_traffic'].apply(np.mean)
    df['web_traffic_growth'] = df['monthly_web_traffic'].apply(lambda x: (x[-1] - x[0]) / (x[0] + 1e-6) if x[0] > 0 else 0)
    return df

print("Performing feature engineering...")
startup_df_featured = feature_engineer(startup_df.copy())
print("-" * 50)

class StartupDataset(Dataset):
    def __init__(self, dataframe, tokenizer, numeric_cols, text_cols, time_series_cols, target_cols):
        self.dataframe, self.tokenizer, self.numeric_cols, self.text_cols, self.time_series_cols, self.target_cols = dataframe, tokenizer, numeric_cols, text_cols, time_series_cols, target_cols
    def __len__(self): return len(self.dataframe)
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        numeric_data = torch.tensor(row[self.numeric_cols].values.astype(np.float32))
        text = row['founder_bio'] + " " + row['product_desc']
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        time_series_data = torch.tensor(row['monthly_web_traffic'], dtype=torch.float32).unsqueeze(1)
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
        return {'numeric_data': numeric_data, 'input_ids': inputs['input_ids'].squeeze(0), 'attention_mask': inputs['attention_mask'].squeeze(0), 'time_series_data': time_series_data, 'targets': targets}

class TabularEncoder(nn.Module):
    def __init__(self, i, o): super().__init__(); self.net = nn.Sequential(nn.Linear(i, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3), nn.Linear(128, o))
    def forward(self, x): return self.net(x)

class TextEncoder(nn.Module):
    def __init__(self, o): super().__init__(); self.bert = BertModel.from_pretrained('bert-base-uncased'); self.fc = nn.Linear(self.bert.config.hidden_size, o)
    def forward(self, i, a): return self.fc(self.bert(input_ids=i, attention_mask=a).pooler_output)

class TimeSeriesEncoder(nn.Module):
    def __init__(self, i, e, n, o):
        super().__init__(); self.embedding = nn.Linear(i, e)
        el = nn.TransformerEncoderLayer(d_model=e, nhead=n, batch_first=True); self.transformer_encoder = nn.TransformerEncoder(el, num_layers=2); self.fc = nn.Linear(e, o)
    def forward(self, x): return self.fc(self.transformer_encoder(self.embedding(x)).mean(dim=1))

class AIPlusModel(nn.Module):
    def __init__(self, num_numeric_features, tab_e=64, txt_e=64, ts_e=32):
        super().__init__()
        self.tabular_encoder = TabularEncoder(num_numeric_features, tab_e)
        self.text_encoder = TextEncoder(txt_e)
        self.ts_encoder = TimeSeriesEncoder(1, 64, 4, ts_e)
        self.fusion_mlp = nn.Sequential(nn.Linear(tab_e + txt_e + ts_e, 256), nn.ReLU(), nn.Dropout(0.5))
        self.success_head = nn.Linear(256, 1)
        self.valuation_head = nn.Linear(256, 1)
    def forward(self, n, i, a, t):
        f = torch.cat([self.tabular_encoder(n), self.text_encoder(i, a), self.ts_encoder(t)], dim=1)
        fused = self.fusion_mlp(f)
        return torch.cat([self.success_head(fused), self.valuation_head(fused)], dim=1)

print("\nStarting final AI+ model training...")
numeric_features = ['age', 'total_funding_usd', 'num_investors', 'team_size', 'is_dpiit_recognized', 'funding_per_investor', 'funding_per_employee', 'founder_has_iit_iim_exp', 'avg_web_traffic', 'web_traffic_growth']
categorical_features = ['sector', 'location']
text_features, ts_features, target_features = ['founder_bio', 'product_desc'], ['monthly_web_traffic'], ['status_success', 'predicted_next_valuation']

preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features),('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
df_to_transform = startup_df_featured[numeric_features + categorical_features]
df_to_keep = startup_df_featured.drop(columns=numeric_features + categorical_features)
processed_data = preprocessor.fit_transform(df_to_transform)
new_col_names = preprocessor.get_feature_names_out()
df_transformed = pd.DataFrame(processed_data, columns=new_col_names, index=startup_df_featured.index)
df_processed = pd.concat([df_transformed, df_to_keep], axis=1)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
processed_numeric_cols = list(preprocessor.get_feature_names_out())
train_dataset = StartupDataset(df_processed, tokenizer, processed_numeric_cols, text_features, ts_features, target_features)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = AIPlusModel(num_numeric_features=len(processed_numeric_cols))
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion_bce, criterion_mse = nn.BCEWithLogitsLoss(), nn.MSELoss()

for epoch in range(10):
    for batch in train_loader:
        optimizer.zero_grad()
        o = model(n=batch['numeric_data'], i=batch['input_ids'], a=batch['attention_mask'], t=batch['time_series_data'])
        loss = criterion_bce(o[:, 0], batch['targets'][:, 0]) + criterion_mse(o[:, 1], batch['targets'][:, 1])
        loss.backward(); optimizer.step()
    print(f"Epoch {epoch+1}/10, Train Loss: {loss.item():.4f}")

print("\nTraining complete. Saving model and preprocessor...")
torch.save(model.state_dict(), 'ai_plus_model.pth')
with open('preprocessor.pkl', 'wb') as f: pickle.dump(preprocessor, f)
print("Artifacts saved: ai_plus_model.pth, preprocessor.pkl")
print("-" * 50)