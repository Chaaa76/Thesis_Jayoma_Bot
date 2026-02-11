# Thesis Jayoma Bot

**Integrating Legal-BERT and Cosine Similarity for Enhanced Ordinance Retrieval**

This repository contains the code and assets for *Jayoma Bot*, a Python-based ordinance retrieval chatbot built as part of a thesis project. The system combines a Legal-BERT language model with cosine similarity search to help users query and retrieve relevant ordinance text from a structured dataset.

##  Features

-  **Ordinance Retrieval** – Finds relevant ordinances based on user queries using semantic embedding.
-  **Chatbot Interface** – A simple conversational interface for users to interact with the bot.
-  **Cosine Similarity Ranking** – Ranks ordinance text based on similarity to user queries.
-  **Data-Driven** – Uses a CSV dataset (`ordinance_data.csv`) for searchable ordinance content.

##  Repository Structure

.
├── app.py # Main application / chatbot runner
├── models.py # Custom model handlers (e.g., embedding + retrieval)
├── ordinance_data.csv # Ordinance dataset used for search
├── chatbot_memory.json # Memory / context storage for the bot
├── requirements.txt # Python dependencies
├── vercel.json # Deployment config (for Vercel)
├── .gitignore
├── .gitattributes
└── ordinance_retrieval.log # Log output from retrieval runs


##  Getting Started

### Requirements

- Python **3.8+**
- A Legal-BERT model (e.g., from Hugging Face or locally hosted)
- (Optional) Deployment platform like Vercel or similar

### Install Dependencies

1. Clone the repo:
   ```bash
   git clone https://github.com/Chaaa76/Thesis_Jayoma_Bot.git
   cd Thesis_Jayoma_Bot
Create a Python virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
Install requirements:

pip install -r requirements.txt
🛠 Usage
Update or confirm your ordinance dataset in ordinance_data.csv.

Configure or load your Legal-BERT model inside models.py.

Run the chatbot:

python app.py
Interact with the bot and submit queries to retrieve ordinance text semantically.

 Logging
The bot logs retrieval activity and results into:

ordinance_retrieval.log
This is useful for analysis and debugging.

Deployment
This project includes a vercel.json config for simple deployment on Vercel (or similar platforms). Be sure to set any environment variables required for model loading or external services before deploying.

Credits
Built as part of a thesis project focusing on the application of transformer-based legal language models and semantic search.
