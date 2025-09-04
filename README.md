# 🌐 NovaPress – AI-Powered News Engine

NovaPress is an **AI-powered news platform** that scrapes, stores, and re-publishes online news with **smart AI-generated summaries, titles, and tags**.  
Built for developers, researchers, and enthusiasts who want to explore news in a **smarter, cleaner, and structured way**.  

---

## ✨ Features  

- ✅ **Scrapes articles** from top sources: *Indian Express, Times of India, Hindustan Times*  
- ✅ **Stores data** in **MongoDB** for structured access  
- ✅ **AI-powered enrichment** using:  
  - 🧠 **BART Summarizer** → generates concise article summaries  
  - 🏷️ **KeyBERT** → extracts keywords & tags  
  - 🤖 AI-generated **titles** for better readability  
- ✅ **Backend** built with **FastAPI** (REST API for serving articles)  
- ✅ **Frontend** powered by **JSX** for an interactive UI  
- ✅ **Web scraping** handled via **Selenium + BeautifulSoup**  

---

## 🛠️ Tech Stack  

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)  
- **Frontend**: JSX (React-like components)  
- **Database**: [MongoDB](https://www.mongodb.com/)  
- **Scraping**: [Selenium](https://www.selenium.dev/), [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)  
- **AI/ML**: [Transformers (BART)](https://huggingface.co/facebook/bart-large-cnn), [KeyBERT](https://github.com/MaartenGr/KeyBERT)  
- **Other**: Python, ThreadPoolExecutor, Logging  

---

## 🚀 Getting Started  

### 1️⃣ Clone the repo  
```bash
git clone https://github.com/yourusername/novapress.git
cd novapress
```

### 2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Setup environment variables  
Create a `.env` file in the root directory:  
```env
MONGODB_URL=mongodb://localhost:27017
```

### 4️⃣ Run the backend (FastAPI)  
```bash
uvicorn Scraper:app --reload
```

The API will be live at:  
👉 `http://127.0.0.1:8000`  

### 5️⃣ Access Endpoints  
- `/articles` → Get latest articles  
- `/articles/category/{category_name}` → Get articles by category  
- `/articles/id/{article_id}` → Get single article by ID  
- `/debug/collections` → Debug Mongo collections  
- `/debug/recent-articles` → Debug latest articles  

---

## 📸 Screenshots (Optional)  
_Add UI previews here once frontend is running._  

---

## 📂 Project Structure  

```
📦 NovaPress
 ┣ 📜 Scraper.py      # Core backend scraper + API
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┣ 📂 frontend/       # JSX-based frontend
 ┣ 📂 logs/           # Log files
 ┗ 📂 .env            # Environment variables
```

---

## 🤝 Contributing  

Contributions are welcome! 🚀  
- Fork the repo  
- Create a new branch (`feature/my-feature`)  
- Commit your changes  
- Submit a PR  

---

## 📜 License  

This project is licensed under the **MIT License** – feel free to use and modify it.  

---

## 💡 Future Scope  

- 🔎 Full-text search with ElasticSearch  
- 📊 Personalized news feed (recommendation system)  
- 🌍 Multi-language support  
- 📱 Mobile-friendly UI  
