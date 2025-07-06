# Book Recommender System

This project is a book recommender system that uses natural language processing and machine learning to provide book recommendations based on semantic search, category filtering, and emotional tone.

## Features

- **Semantic Search:** Find books based on a natural language query.
- **Category Filtering:** Filter recommendations by categories like "Fiction" and "Nonfiction".
- **Emotional Tone:** Sort recommendations based on emotional tones like "Happy," "Sad," "Suspenseful," etc.
- **Interactive Dashboard:** A user-friendly web interface built with Gradio to interact with the recommender system.

## Getting Started

### Prerequisites

- Python 3.x
- An OpenAI API key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Deepika-20022004/Project-Book-Recommender.git
    cd "Project-Book-Recommender"
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

## Usage

To start the Gradio dashboard, run the following command:

```bash
python 5_gradio-dashboard.py
```

This will launch a local web server, and you can access the dashboard in your browser at the provided URL (usually `http://127.0.0.1:7860`).

## Project Structure

-   `1_data-exploration.ipynb`: Jupyter notebook for initial data exploration and cleaning.
-   `2_vector-search.ipynb`: Jupyter notebook for creating vector embeddings and setting up the semantic search.
-   `3_text-classification.ipynb`: Jupyter notebook for classifying books into categories.
-   `4_sentiment-analysis.ipynb`: Jupyter notebook for analyzing the emotional tone of book descriptions.
-   `5_gradio-dashboard.py`: Python script to launch the Gradio web interface.
-   `books_cleaned.csv`: The cleaned dataset of books.
-   `books_with_categories.csv`: The dataset with added book categories.
-   `books_with_emotions.csv`: The final dataset with added emotional analysis.
-   `tagged_description.txt`: A text file containing the tagged descriptions used for vector search.

