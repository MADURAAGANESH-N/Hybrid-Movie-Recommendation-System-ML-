# 🎬 Hybrid Movie Recommendation System Using Machine Learning

A machine learning-based movie recommendation system that combines **Content-Based Filtering** and **Collaborative Filtering** to recommend movies similar to a user's selected movie.

The system uses a weighted hybrid approach and provides an interactive **Streamlit web application**.

---

## 📌 Project Overview

Movie recommendation systems help users discover movies based on their interests and previous viewing preferences.

This project implements a **Hybrid Movie Recommendation System** using two recommendation techniques:

- **Content-Based Filtering** – recommends movies based on similar genres.
- **Collaborative Filtering** – recommends movies based on similarities in user-rating behavior.

The final recommendation score is calculated as:

```text
Hybrid Score =
0.6 × Content Score +
0.4 × Collaborative Score
```

---

## ✨ Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Content-Based Movie Recommendation
- Collaborative Filtering
- Hybrid Recommendation System
- Precision@10 evaluation
- Recall@10 evaluation
- Interactive Streamlit web application
- TF-IDF based movie similarity
- Sparse matrix based collaborative filtering
- Saved machine learning model files
- Jupyter Notebook implementation

---

## 🧠 Recommendation Methods

### 1. Content-Based Filtering

The content-based recommender uses movie genre information to identify similar movies.

The process is:

```text
Movie Genres
     ↓
Text Processing
     ↓
TF-IDF Vectorization
     ↓
Cosine Similarity
     ↓
Similar Movies
```

TF-IDF converts movie genre information into numerical representations.

Cosine similarity is then used to calculate the similarity between the selected movie and other movies.

---

### 2. Collaborative Filtering

Collaborative filtering uses user-rating behavior to identify movies that have similar rating patterns.

The process is:

```text
User Ratings
     ↓
User-Movie Sparse Matrix
     ↓
Movie Similarity
     ↓
Collaborative Recommendations
```

A sparse matrix is used to handle the large number of users and ratings efficiently.

---

### 3. Hybrid Recommendation

The final recommendation combines both approaches.

```text
Content Score × 0.6
        +
Collaborative Score × 0.4
        ↓
   Hybrid Score
        ↓
 Top Recommendations
```

Movies with higher hybrid scores are ranked higher in the recommendation list.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-learn
- TF-IDF
- Cosine Similarity

### Data Visualization

- Matplotlib
- Seaborn

### Model Storage

- Joblib

### Web Application

- Streamlit

### Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

## 📂 Project Structure

```text
Hybrid Movie Recommendation System Using Machine Learning/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── movies.csv
│   │   └── ratings.csv
│   │
│   └── processed/
│       ├── movies_clean.csv
│       └── ratings_clean.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── movie_indices.pkl
│   ├── movies.pkl
│   ├── user_movie_sparse.npz
│   ├── user_to_index.pkl
│   ├── movie_to_index.pkl
│   ├── index_to_movie.pkl
│   └── hybrid_config.pkl
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_content_based_recommender.ipynb
│   ├── 04_collaborative_filtering.ipynb
│   ├── 05_hybrid_recommender.ipynb
│   └── 06_evaluation.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── rating_distribution.png
│   │   ├── most_rated_movies.png
│   │   ├── average_rating_distribution.png
│   │   ├── genre_distribution.png
│   │   ├── user_activity.png
│   │   └── movies_by_year.png
│   │
│   └── evaluation_results.csv
│
├── screenshots/
│
├── report/
│
├── src/
│   ├── __init__.py
│   ├── content_recommender.py
│   ├── collaborative_recommender.py
│   ├── hybrid_recommender.py
│   └── evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

The system uses movie and user-rating data containing information such as:

- Movie ID
- Movie title
- Movie genres
- User ID
- User ratings

The dataset is cleaned and processed before being used by the recommendation models.

> **Note:** Dataset files may be excluded from the GitHub repository because of their size.

---

## 🔄 Project Workflow

```text
                 Movie Dataset
                      │
                      ▼
              Data Preprocessing
                      │
                      ▼
             Exploratory Data Analysis
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
 Content-Based Model      Collaborative Model
          │                       │
          ▼                       ▼
   TF-IDF + Cosine        Sparse User-Movie
      Similarity              Matrix
          │                       │
          └───────────┬───────────┘
                      ▼
              Hybrid Recommendation
                      │
                      ▼
                  Evaluation
                      │
                      ▼
              Streamlit Application
```

---

## 📈 Exploratory Data Analysis

The project performs exploratory analysis of the movie-rating dataset.

The generated visualizations include:

1. Rating Distribution
2. Most Rated Movies
3. Average Rating Distribution
4. Genre Distribution
5. User Activity
6. Movies by Year

The figures are stored inside:

```text
outputs/figures/
```

---

## 📏 Model Evaluation

The recommendation system is evaluated using:

### Precision@10

Precision@10 measures the proportion of relevant movies in the top 10 recommendations.

```text
Precision@10 =
Relevant Recommended Movies
---------------------------
Total Recommended Movies
```

### Recall@10

Recall@10 measures the proportion of relevant movies that were successfully recommended.

```text
Recall@10 =
Relevant Recommended Movies
---------------------------
Total Relevant Movies
```

Evaluation results are saved in:

```text
outputs/evaluation_results.csv
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/MADURAAGANESH-N/Hybrid-Movie-Recommendation-System-ML-.git
```

Move into the project directory:

```bash
cd Hybrid-Movie-Recommendation-System-ML-
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

---

### 3. Activate the virtual environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 🎯 How to Use

1. Open the Streamlit application.
2. Select a movie from the dropdown.
3. Choose the number of recommendations.
4. Click **Recommend Movies**.
5. The system generates movie recommendations.
6. Each recommendation displays:
   - Movie title
   - Genres
   - Content similarity score
   - Collaborative similarity score
   - Hybrid score

---

## 🖥️ Application Screenshots

After running the application, screenshots can be added to the repository.

Example:

```markdown
![Home Page](screenshots/01_homepage.png)

![Movie Selection](screenshots/02_movie_selection.png)

![Recommendations](screenshots/03_recommendations.png)

![Scores](screenshots/04_scores.png)

![EDA](screenshots/05_eda.png)

![Evaluation](screenshots/06_evaluation.png)
```

---

## 🎯 Example

A user can select:

```text
Toy Story (1995)
```

The system then calculates:

```text
Content Similarity
        +
Collaborative Similarity
        ↓
Hybrid Score
        ↓
Top Movie Recommendations
```

The recommended movies are sorted based on the final hybrid score.

---

## ⚡ Performance Considerations

The collaborative filtering dataset contains a large number of users and ratings.

To reduce memory usage, the project uses a **sparse user-movie matrix** instead of a large dense matrix.

The system also calculates similarity for the selected movie instead of creating a complete movie-to-movie similarity matrix.

This approach helps the system work with large rating datasets while reducing unnecessary memory consumption.

---

## 📁 Generated Model Files

The recommendation system uses saved model artifacts such as:

```text
models/
├── tfidf_vectorizer.pkl
├── tfidf_matrix.pkl
├── movie_indices.pkl
├── movies.pkl
├── user_movie_sparse.npz
├── user_to_index.pkl
├── movie_to_index.pkl
├── index_to_movie.pkl
└── hybrid_config.pkl
```

These files allow the application to load the trained recommendation components without rebuilding them every time.

---

## 🔮 Future Improvements

Possible future improvements include:

- Movie poster integration
- Personalized recommendations for individual users
- User login and profile management
- Advanced recommendation algorithms
- Matrix factorization
- Better movie search
- Personalized ranking
- Deployment to Streamlit Community Cloud
- Interactive recommendation analytics
- Improved evaluation using larger test sets

---

## 📚 Project Deliverables

The project includes:

- Data preprocessing notebook
- Exploratory Data Analysis notebook
- Content-based recommender
- Collaborative filtering recommender
- Hybrid recommender
- Evaluation notebook
- Streamlit application
- Source code
- Model files
- Screenshots
- Project report

---

## 👨‍💻 Author

**Maduraa Ganesh**

B.Tech Information Technology

---

## ⭐ Acknowledgement

This project was developed as an academic machine learning project to demonstrate the implementation of recommendation system techniques using Python, machine learning libraries, and Streamlit.

---

## 📜 License

This project is intended for educational and academic purposes.
