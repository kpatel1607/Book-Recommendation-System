#Collaborative Filtering User-Based
import streamlit as st
import pickle


book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))
model = pickle.load(open('recommendation_model.pkl', 'rb'))
book_sparse = pickle.load(open('book_sparse.pkl', 'rb'))

query = book_pivot.iloc[393].values.reshape(1, -1)
distances, suggestions = model.kneighbors(query, n_neighbors=6)


def recommend_book(book_name, book_pivot, book_sparse, model):
    if book_name not in book_pivot.index:
        return ["❌ Book not found."]

    # Get index of the book
    book_index = book_pivot.index.get_loc(book_name)

    # Query on sparse matrix
    query = book_sparse[book_index]

    # Get nearest neighbors
    distances, indices = model.kneighbors(query, n_neighbors=6)

    # Get recommendations (excluding the book itself at index 0)
    recommendations = []
    for i in range(1, len(indices[0])):
        title = book_pivot.index[indices[0][i]]
        sim_score = 1 - distances[0][i]
        recommendations.append(f"➡️ {title}")

    return recommendations



st.title("📚 Book Recommendation System")

book_list = list(book_pivot.index)
selected_book = st.selectbox("🔍 Select a book to get recommendations:", book_list)

if st.button("Recommend"):
    with st.spinner("Finding similar books..."):
        recommendations = recommend_book(selected_book, book_pivot, book_sparse, model)
    st.subheader("📖 Top Recommendations:")
    for rec in recommendations:
        st.write(rec)



# Content-Based Filtering
# books = pickle.load(open("books_cbf.pkl", "rb"))
# cosine_sim = pickle.load(open("cosine_sim_cbf.pkl", "rb"))
# title_to_index = pickle.load(open("title_to_index_cbf.pkl", "rb"))
#
#
# def recommend_cbf(book_name, top_n=5):
#     book_name = book_name.lower()
#     if book_name not in title_to_index:
#         return ["❌ Book not found in CBF dataset."]
#     idx = title_to_index[book_name]
#     similarity_scores = list(enumerate(cosine_sim[idx]))
#     similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
#     similarity_scores = similarity_scores[1:top_n + 1]  # skip itself
#     recs = []
#     for i, score in similarity_scores:
#         rec_title = books.iloc[i]["title"]
#         recs.append(f"➡️ {rec_title} (Score: {score:.2f})")
#     return recs
#
# st.title("🔍 Content-Based Book Recommender")
#
# book_list = list(books["title"])
# selected_book = st.selectbox("Select a book:", book_list)
#
# if st.button("Recommend"):
#     with st.spinner("Finding recommendations..."):
#         recs = recommend_cbf(selected_book)
#     st.subheader("Recommendations:")
#     for r in recs:
#         st.write(r)
