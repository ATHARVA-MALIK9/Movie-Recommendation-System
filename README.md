# 🎬 Movie Recommendation System

A beginner-friendly movie recommendation system built in Python that implements two popular recommendation techniques:

- **Content-Based Filtering** — recommends movies similar to one you like, based on genres
- **Collaborative Filtering** — recommends movies based on what similar users have rated

---

## 📸 Features

- Clean interactive web UI built with **Streamlit**
- Two recommendation algorithms from scratch using **scikit-learn**
- TF-IDF vectorization + Cosine Similarity for content filtering
- User-based collaborative filtering with weighted ratings
- Easily extendable dataset

---

## 🗂️ Project Structure

```
movie-recommender/
│
├── recommender.py      # Core logic: both recommendation algorithms
├── app.py              # Streamlit web application (UI)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the terminal demo (no UI)
```bash
python recommender.py
```

### 4. Launch the web app
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## 🧠 How It Works

### Content-Based Filtering
1. Movie genres are converted into numerical vectors using **TF-IDF** (Term Frequency–Inverse Document Frequency).
2. **Cosine Similarity** is computed between all movie vectors.
3. Movies with the highest similarity score to the selected movie are returned.

```
Genre String  →  TF-IDF Vector  →  Cosine Similarity  →  Top-N Similar Movies
```

### Collaborative Filtering
1. A user-movie ratings matrix is built.
2. **Cosine Similarity** is computed between all user rating vectors.
3. For movies the target user hasn't rated, a predicted score is calculated as a **weighted average** of similar users' ratings.
4. Movies with the highest predicted scores are recommended.

```
User Ratings  →  User Similarity  →  Weighted Prediction  →  Top-N Recommendations
```

---

## 📊 Sample Dataset

The project includes a built-in dataset of **10 movies** and **5 users** for demonstration.

| Movie              | Genres                    |
|--------------------|---------------------------|
| The Dark Knight    | Action Crime Drama        |
| Inception          | Action Sci-Fi Thriller    |
| Interstellar       | Sci-Fi Drama Adventure    |
| The Notebook       | Romance Drama             |
| Toy Story          | Animation Comedy Family   |
| ...                | ...                       |

---

## 🛠️ Tech Stack

| Tool           | Purpose                        |
|----------------|--------------------------------|
| Python 3.x     | Core language                  |
| Pandas         | Data manipulation              |
| NumPy          | Numerical computations         |
| scikit-learn   | TF-IDF & Cosine Similarity     |
| Streamlit      | Interactive web UI             |

---

## 🚀 Future Improvements

- [ ] Load real datasets (e.g., MovieLens 100K)
- [ ] Add matrix factorization (SVD)
- [ ] Add user registration & rating input
- [ ] Deploy on Streamlit Cloud

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Your Name**  
Internship Project — Movie Recommendation System  
[GitHub](https://github.com/YOUR_USERNAME) • [LinkedIn](https://linkedin.com/in/YOUR_USERNAME)
