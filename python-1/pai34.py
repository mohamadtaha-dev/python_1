import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Load data
df = pd.read_csv('C:\\Users\\Acer\\Desktop\\New folder (2)\\movies-withnull.csv')

# Clean data
df.dropna(subset=['Rating'], inplace=True)
df['Revenue (Millions)'].fillna(0, inplace=True)
df['Metascore'].fillna(0, inplace=True)
df['Votes'].fillna(0, inplace=True)

df['MainGenre'] = df['Genre'].apply(lambda x: x.split(',')[0] if pd.notna(x) else 'Unknown')
le_genre = LabelEncoder()
df['Genre_encoded'] = le_genre.fit_transform(df['MainGenre'])

# KNN classifier
knn_features = ['Genre_encoded', 'Votes', 'Rating']
df_knn = df.dropna(subset=knn_features).copy()
X_knn = df_knn[knn_features]
scaler = StandardScaler()
X_knn_scaled = scaler.fit_transform(X_knn)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_knn_scaled, df_knn['Title'])

# Models for rating prediction
features = ['Genre_encoded', 'Votes', 'Revenue (Millions)', 'Metascore']
target = 'Rating'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)

# Recommendation by title
def movie_recommend_by_title(movie_name):
    titles_lower = df['Title'].str.strip().str.lower()
    input_clean = movie_name.strip().lower()

    if input_clean not in titles_lower.values:
        return ["Movie not found."]
    
    movie_index = titles_lower[titles_lower == input_clean].index[0]
    if movie_index not in X_knn.index:
        return ["Insufficient data for recommendations."]
    
    movie_features = X_knn.loc[[movie_index]]
    movie_scaled = scaler.transform(movie_features)
    distances, indices = knn_classifier.kneighbors(movie_scaled)
    recommended_indices = indices[0][1:]

    info = []
    for idx in recommended_indices:
        row = df_knn.iloc[idx]
        text = f"🎬 {row['Title']}\nGenre: {row['Genre']}\nRating: {row['Rating']}\nVotes: {row['Votes']}\nRevenue: ${row['Revenue (Millions)']}M\n"
        info.append(text)
    return info

# Recommendation by rating
def movie_recommend_by_rating(input_rating):
    try:
        r = float(input_rating)
        similar = df[(df['Rating'] >= r - 0.3) & (df['Rating'] <= r + 0.3)].head(10)
        if similar.empty:
            return ["No movies found with similar rating."]
        info = []
        for _, row in similar.iterrows():
            text = f"🎬 {row['Title']}\nGenre: {row['Genre']}\nRating: {row['Rating']}\nVotes: {row['Votes']}\nRevenue: ${row['Revenue (Millions)']}M\n"
            info.append(text)
        return info
    except:
        return ["Invalid rating input."]

# GUI
root = tk.Tk()
root.title("🎬 Movie Recommendation System")
root.geometry("700x600")
root.configure(bg="#eaf2f8")

tk.Label(root, text="Enter Movie Title:", font=("Helvetica", 13), bg="#eaf2f8").pack(pady=5)
title_entry = tk.Entry(root, width=50, font=("Helvetica", 12))
title_entry.pack(pady=5)

tk.Label(root, text="Or Enter Rating (e.g. 7.5):", font=("Helvetica", 13), bg="#eaf2f8").pack(pady=5)
rating_entry = tk.Entry(root, width=50, font=("Helvetica", 12))
rating_entry.pack(pady=5)

output_box = tk.Text(root, width=80, height=20, font=("Helvetica", 11), bg="#fdfefe")
output_box.pack(pady=10)

def show_recommendations():
    title_input = title_entry.get().strip()
    rating_input = rating_entry.get().strip()
    output_box.delete("1.0", tk.END)

    if title_input:
        results = movie_recommend_by_title(title_input)
    elif rating_input:
        results = movie_recommend_by_rating(rating_input)
    else:
        messagebox.showwarning("Input Needed", "Please enter a movie title or a rating.")
        return

    output_box.insert(tk.END, "\n\n".join(results))

# Stylish Button
recommend_btn = tk.Button(root, text="🎯 Recommend", command=show_recommendations, font=("Helvetica", 13), bg="#2e86c1", fg="white", width=20)
recommend_btn.pack(pady=10)

root.mainloop()
