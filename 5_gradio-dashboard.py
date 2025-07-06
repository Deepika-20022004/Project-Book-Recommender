import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
# gradio- open source py package to build dashboards, specifically to showcase ML models

load_dotenv() # load OPENAI_API_KEY

# Code to load parts of the dashboard

# Book thumbnail
# For books that already have thumbnail in the dataset, use it, else use cover-not-found.jpg
books=pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"]=books["thumbnail"]+"&fife=w888"
books["large_thumbnail"]=np.where(
    books["large_thumbnail"].isna(),"cover-not-found.jpg",
    books["large_thumbnail"])

# Vector database- semantic search
# Load data
loader=TextLoader("tagged_description.txt", encoding="utf-8")
raw_documents=loader.load()
# Chunk
text_splitter=CharacterTextSplitter(separator="\n",chunk_size=0,chunk_overlap=0)
documents=text_splitter.split_documents(raw_documents)
# Embed and store
db_books=Chroma.from_documents(documents,OpenAIEmbeddings())

# Now let's create a function to:
#   a. retrieve the semantic recommendation from the books dataset
#   b. apply filtering based on category
#   c. sort based on emotional tone
def retrieve_semantic_recommendations (
    query: str,
    category: str=None,
    tone: str=None,
    initial_top_k: int=50,
    final_top_k: int = 16,
)-> pd.DataFrame:
    # a.
    recs=db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec, score in recs]
    book_recs=books[books["isbn13"].isin(books_list)].head(final_top_k)

    # b.
    if category!="ALL":
        book_recs = book_recs[book_recs["simple_categories"]==category].head(final_top_k)
    else:
        book_recs=book_recs.head(final_top_k)

    # c.
    if tone=="Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone=="Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone=="Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone=="Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone=="Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

# Now let's create a function to specify what to display on the gradio dashboard
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone) # we have a set of recommendations
    results = [] # Final results list to return

    # Iterate over recommendations
    for _, row in recommendations.iterrows():
        # Work on description
        # Store each description
        description=row["description"]
        # Split the description into a list of words
        truncated_desc_split=description.split()
        # Join upto 30 words back into description, followed by '...' as we don't want the entire description to be displayed and keep it uniform
        truncated_description="".join(truncated_desc_split[:30])+'...'

        # Work on author
        # Each row may have 1+ authors separated by ';', so split them
        authors_split=row["authors"].split(';')
        if len(authors_split)==2: # Join the names by '<author1> and <author2>'
            authors_str=f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split)>2: # Join the names by comma (with 'and' before the last author)
            authors_str=f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else: # We have just 1 author so leave it as it is
            authors_str=row["authors"]


        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories=["All"]+ sorted(books["simple_categories"].unique())
tones=["All"]+["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard: # Can choose any theme: https://www.gradio.app/guides/theming-guide
    gr.Markdown("# Semantic book recommender")
    with gr.Row():
        user_query=gr.Textbox(label="Please enter a description of a book:",
                              placeholder = "e.g., A story about love")
        category_dropdown=gr.Dropdown(choices=categories,
                                      label="Select a category:",
                                      value="All")
        tone_dropdown=gr.Dropdown(choices=tones,
                                  label="Select an emotional tone:",
                                  value = "All")
    submit_button=gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output=gr.Gallery(label="Recommended books", columns=8, rows=2)
    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__=="__main__":
    dashboard.launch()
