import tkinter as tk
import pandas as pd
import nltk
import re
import gensim
from gensim import corpora
import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load Datasets
plot_summaries_path = "plot_summaries.txt"
movie_metadata_path = "movie.metadata.tsv"
character_metadata_path = "character.metadata.tsv"

# Load and preprocess the datasets
plot_summaries = pd.read_csv(plot_summaries_path, sep="\t", header=None, names=["movie_id", "summary"])
movie_metadata = pd.read_csv(movie_metadata_path, sep="\t", header=None, names=[
    "movie_id", "freebase_id", "movie_name", "release_date", "box_office_revenue",
    "runtime", "languages", "countries", "genres"
])
character_metadata = pd.read_csv(character_metadata_path, sep="\t", header=None, names=[
    "movie_id", "freebase_id", "release_date", "character_name", "actor_birth_date",
    "actor_gender", "actor_height", "actor_ethnicity", "actor_name", "actor_age_at_release",
    "freebase_char_actor_map_id", "freebase_char_id", "freebase_actor_id"
])

# Merge datasets for easier querying
movie_data = pd.merge(movie_metadata, character_metadata, on="movie_id", how="left")
movie_data = pd.merge(movie_data, plot_summaries, on="movie_id", how="left")

# NLTK and Regex for text processing
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

# Preprocess the plot summaries
movie_data['summary_clean'] = movie_data['summary'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
movie_data['summary_tokens'] = movie_data['summary_clean'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(movie_data['summary_tokens'])
corpus = [dictionary.doc2bow(text) for text in movie_data['summary_tokens']]

# Build LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Define the chatbot application class
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Movie Chatbot")
        self.text_display = tk.Text(self.root, wrap=tk.WORD, height=20, width=80)
        self.text_display.pack(pady=10)

        self.entry = tk.Entry(self.root, width=80)
        self.entry.pack(pady=5)

        self.send_button = tk.Button(self.root, text="Send", command=self.process_input)
        self.send_button.pack(pady=5)

    def process_input(self):
        user_input = self.entry.get().strip().lower()
        response = self.generate_response(user_input)
        self.display_response(response)

    def generate_response(self, user_input):
        # Search for queries related to genre
        if "genre" in user_input:
            genre = re.search(r"genre of ([\w\s]+)", user_input)
            if genre:
                genre = genre.group(1).strip().title()
                movies = movie_data[movie_data['genres'].str.contains(genre, na=False, case=False)]
                movie_titles = movies['movie_name'].dropna().unique()
                if movie_titles.any():
                    return f"Movies with the genre '{genre}':\n" + "\n".join(movie_titles[:10])
                else:
                    return f"No movies found with the genre '{genre}'."

        # Search for queries related to actors
        elif "actor" in user_input:
            actor = re.search(r"movies that (.+) is in", user_input)
            if actor:
                actor_name = actor.group(1).strip().title()
                movies = movie_data[movie_data['actor_name'].str.contains(actor_name, na=False, case=False)]
                movie_titles = movies['movie_name'].dropna().unique()
                if movie_titles.any():
                    return f"Movies that {actor_name} is in:\n" + "\n".join(movie_titles[:10])
                else:
                    return f"No movies found for the actor {actor_name}."

        # Search for movies based on multiple attributes like actor and genre
        elif "movies" in user_input and "with" in user_input:
            actor_search = re.search(r"movies with (.+) in (.+) genre", user_input)
            if actor_search:
                actor_name = actor_search.group(1).strip().title()
                genre_name = actor_search.group(2).strip().title()
                movies = movie_data[
                    (movie_data['actor_name'].str.contains(actor_name, na=False, case=False)) &
                    (movie_data['genres'].str.contains(genre_name, na=False, case=False))
                ]
                movie_titles = movies['movie_name'].dropna().unique()
                if movie_titles.any():
                    return f"Movies with {actor_name} in the '{genre_name}' genre:\n" + "\n".join(movie_titles[:10])
                else:
                    return f"No movies found for {actor_name} in the genre '{genre_name}'."

        # Search for movies with specific attributes like language, country, actor gender, and age at release
        elif "movies" in user_input and ("language" in user_input or "country" in user_input or "actor gender" in user_input or "actor age" in user_input):
            if "language" in user_input:
                language = re.search(r"movies in (.+) language", user_input)
                if language:
                    language_name = language.group(1).strip().title()
                    movies = movie_data[movie_data['languages'].str.contains(language_name, na=False, case=False)]
                    movie_titles = movies['movie_name'].dropna().unique()
                    if movie_titles.any():
                        return f"Movies in {language_name} language:\n" + "\n".join(movie_titles[:10])
                    else:
                        return f"No movies found in the {language_name} language."

            elif "country" in user_input:
                country = re.search(r"movies from (.+) country", user_input)
                if country:
                    country_name = country.group(1).strip().title()
                    movies = movie_data[movie_data['countries'].str.contains(country_name, na=False, case=False)]
                    movie_titles = movies['movie_name'].dropna().unique()
                    if movie_titles.any():
                        return f"Movies from {country_name}:\n" + "\n".join(movie_titles[:10])
                    else:
                        return f"No movies found from the country {country_name}."

            elif "actor gender" in user_input:
                gender = re.search(r"movies with (.+) gender", user_input)
                if gender:
                    actor_gender = gender.group(1).strip().title()
                    movies = movie_data[movie_data['actor_gender'].str.contains(actor_gender, na=False, case=False)]
                    movie_titles = movies['movie_name'].dropna().unique()
                    if movie_titles.any():
                        return f"Movies with actors of {actor_gender} gender:\n" + "\n".join(movie_titles[:10])
                    else:
                        return f"No movies found with actors of {actor_gender} gender."

            elif "actor age" in user_input:
                age = re.search(r"movies with actor age at release (\d+)", user_input)
                if age:
                    age_value = int(age.group(1))
                    movies = movie_data[movie_data['actor_age_at_release'] == age_value]
                    movie_titles = movies['movie_name'].dropna().unique()
                    if movie_titles.any():
                        return f"Movies with actors aged {age_value} at release:\n" + "\n".join(movie_titles[:10])
                    else:
                        return f"No movies found with actors aged {age_value} at release."

        # Search for movies based on release year
        elif "released in" in user_input:
            year_search = re.search(r"released in (\d{4})", user_input)
            if year_search:
                year = year_search.group(1)
                movies = movie_data[movie_data['release_date'].astype(str).str.startswith(year, na=False)]
                movie_titles = movies['movie_name'].dropna().unique()
                if movie_titles.any():
                    return f"Movies released in {year}:\n" + "\n".join(movie_titles[:10])
                else:
                    return f"No movies found released in {year}."

        # Search for movie summaries
        elif "summary" in user_input:
            title = re.search(r"summary of (.+)", user_input)
            if title:
                movie_name = title.group(1).strip().title()
                movie = movie_data[movie_data['movie_name'].str.contains(movie_name, na=False, case=False)]
                if not movie.empty:
                    return f"Summary of {movie_name}:\n" + movie['summary'].values[0]
                else:
                    return f"No movie found with the title '{movie_name}'."
        
        # Search for topics in plot summaries using LDA
        elif "topics" in user_input:
            topics = lda_model.show_topics(formatted=True, num_words=5)
            return "Top topics in movie plots:\n" + "\n".join([f"{i}: {topic}" for i, topic in enumerate(topics)])

        # Search for movies based on shared topics using networkx
        elif "network" in user_input:
            G = nx.Graph()
            for idx, row in movie_data.iterrows():
                if pd.notnull(row['summary']):
                    topic_distribution = lda_model[dictionary.doc2bow(row['summary_tokens'])]
                    for topic_num, _ in topic_distribution:
                        G.add_node(row['movie_name'], topic=topic_num)
                        for neighbor_idx, neighbor_row in movie_data.iterrows():
                            if idx != neighbor_idx and pd.notnull(neighbor_row['summary']):
                                neighbor_topic_distribution = lda_model[dictionary.doc2bow(neighbor_row['summary_tokens'])]
                                for neighbor_topic_num, _ in neighbor_topic_distribution:
                                    if topic_num == neighbor_topic_num:
                                        G.add_edge(row['movie_name'], neighbor_row['movie_name'])

            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
            plt.title("Movie Network Based on Shared Topics")
            plt.show()
            return "Network visualization generated. Check the plot for details."

        # General response if the query is not recognized
        else:
            return "I can help with finding movies by genre, actor, language, country, and more. Try asking something like 'Give me movies that Johnny Depp is in', 'List movies in French language', or 'What movies were released in 2020?'."

    def display_response(self, response):
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, response)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
