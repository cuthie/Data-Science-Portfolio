This project implements a movie recommender system using the MovieLens dataset (available at GroupLens￼) and incorporates features extracted from IMDb. The system includes item-based and user-based recommendation approaches.

⸻

Project Structure

Part 0 – Reading and Importing Data
	•	Datasets are imported from MovieLens, including movies, ratings, and links.
	•	Using a web-scraping function, an additional CSV file is created containing actors/stars lists from IMDb.
	•	All datasets are prepared for further processing and analysis.

⸻

Part 1 – Data Cleaning, Feature Extraction, and Preliminary Analysis

	•	Datasets are trimmed and cleaned to ensure compatibility across all files.
	•	Key features are extracted:
	•	Movie genres
	•	Release year/decade
	•	Stars/actors list
	•	Basic exploratory analysis is performed using word clouds to visualize popular genres, decades, and actors.
	•	The similarity metric is defined using an adjusted cosine similarity:
	•	Adjusted cosine similarity accounts for individual user rating biases by deducting the user’s average rating from all ratings.
	•	This metric provides more realistic similarity measurements compared to alternatives like Euclidean, Hamming, or Jaccard distance.

⸻

Part 2 – Item-Based Recommender System
	•	Recommendations are generated based on similarities between movies.
	•	Movies that a user has not yet rated are predicted using a weighted average of ratings from similar movies.
	•	This approach identifies items a user may like based on patterns across movies.

⸻

Part 3 – User-Based Recommender System
	•	Recommendations are generated based on similarities between users.
	•	Movies that a user has not yet rated are predicted using a weighted average of ratings from similar users.
	•	This approach identifies items a user may like based on patterns across other users with similar tastes.



