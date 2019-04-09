# Setting up the paths and loading data into DataFrames
books_path = './dataset/books.csv'
booktags_path = './dataset/book_tags.csv'
ratings_path = './dataset/ratings.csv'
tags_path = './dataset/tags.csv'
toread_path = './dataset/to_read.csv'

books = pd.read_csv(books_path)
booktags = pd.read_csv(booktags_path)
ratings = pd.read_csv(ratings_path)
tags = pd.read_csv(tags_path)
toread = pd.read_csv(toread_path)

# Exploring the books data
print(books.shape)
print(books.info())
print(books.columns)
books.head()

# Exploring the tags data
print(tags.shape)
print(tags.info())
print(tags.columns)
tags.head()

tags['tag_name'].sample(20)

# Cleaning up the tags to match genres in Good Reads website
# Excluding Chick Lit, Fiction, NonFiction from the genres
goodread_genres = ["Art", "Biography", "Business", "Children", "Christian", "Classic",
                   "Comic", "Contemporary", "Cookbook", "Crime", "Ebook", "Fantasy",
                   "Gay", "Lesbian", "Graphic Novel", "Historical Fiction", "History",
                   "Horror", "Humor", "Comedy", "Manga", "Memoir", "Music", "Mystery",
                   "Paranormal", "Philosophy", "Poetry", "Psychology", "Religion",
                   "Romance", "Science", "Science Fiction", "Self Help", "Suspense",
                   "Spirituality", "Sport", "Thriller", "Travel", "Young Adult"]
goodread_tags = [x.lower() for x in goodread_genres]

# Iterating through the user inputted tags and creating new tags from the Good Reads genres
match_tag = []
for genre in tags['tag_name']:
    check = set([tag if ' ' + tag in ' ' + genre.replace('-', ' ').lower() else '0' for tag in goodread_tags])
    if len(check) > 1:
        match_tag.append(check - {'0'})
    else:
        match_tag.append('0')

tags['matched_tag'] = match_tag
tags['matched_tag'] = tags['matched_tag'].apply(lambda x: ', '.join(x))

# Dropping tags which do not coincide with any Good Reads genre
newtags = tags[tags['matched_tag'] != '0'].drop(columns='tag_name')

# Exploring the booktags data
print(booktags.shape)
print(booktags.info())
print(booktags.columns)
print("Checking if which book id is used:")
print(booktags['goodreads_book_id'].unique())
booktags.head()

# Appending the Good Reads genres on the right of the booktags
booktags = pd.merge(booktags, newtags, how='right', on='tag_id')

# Aggregating the genres for each book, creating metadata for book content
bookcontent = pd.merge(books, booktags, how='inner', left_on='book_id', right_on='goodreads_book_id')
bookcontent = bookcontent.groupby('book_id')['matched_tag'].apply(' '.join).reset_index()

# Adding in Author names to complete the book contents metadata
bookcontent = pd.merge(bookcontent, books[['book_id', 'authors', 'id']], on='book_id', how='inner')
bookcontent['content'] = (pd.Series(bookcontent[['authors', 'matched_tag']].fillna('').values.tolist()).str.join(' '))

# Ensuring that the bookcontent DataFrame is ordered according to 'id'
bookcontent = bookcontent.sort_values('id')

print(bookcontent.shape)
bookcontent.head()

# Exploring the ratings data
print(ratings.shape)
print(ratings.info())
print(ratings.columns)
print("Checking if which book id is used:")
print(ratings['book_id'].unique())
ratings.head()

# Subsetting the ratings data to allow ease of computation
# Subset books & users that have at least 100 ratings
book_count = ratings['book_id'].value_counts()
user_count = ratings['user_id'].value_counts()
df_rating = ratings[ratings['user_id'].isin(user_count[user_count >= 100].index)]
df_rating = df_rating[df_rating['book_id'].isin(book_count[book_count == 100].index)]
print(df_rating.shape)
df_rating.head()

# Creating user/item utility matrix
rating_matrix = pd.pivot_table(df_rating, index='user_id', columns='book_id', aggfunc='mean')
rating_matrix = rating_matrix.apply(lambda x: x - np.mean(x), axis=1).fillna(0)
print(rating_matrix.shape)
rating_matrix.head()

# Looking at the top rated books
books.sort_values(by="average_rating", ascending=False).head(20)['title']

# Function to create user-user cosine similarity matrix


def user_similarity(utility_matrix):
    '''
    Function to create a user-user cosine similarity matrix using pairwise_distances
    Returns a DataFrame of the user-user cosine similarity matrix
    '''
    temp_df = 1 - pairwise_distances(utility_matrix, utility_matrix, metric='cosine')

    return pd.DataFrame(temp_df, index=utility_matrix.index, columns=utility_matrix.index)


# Functions to score the recommendations


def dcg_at_k(r, k):
    '''
    Function to compute discounted cumulative gain (dcg)
    Returns dcg
    r: Relevance scores (list or numpy) in rank order (first element is the first item)
    k: Number of results to consider
    '''
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    '''
    Function to compute normalised discounted cumulative gain (ndcg)
    Returns ndcg
    r: Relevance scores (list or numpy) in rank order (first element is the first item)
    k: Number of results to consider
    '''
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

# Functions to recommend based on user-user collaborative filtering or content-based filtering


def user_recommendation(ratings, k=2):
    '''
    Function to recommend books based on user's book ratings and nearest user cosine similarities
    Returns list of recommended books in rank order
    ratings: Dictionary of id in books and rating key:value pairs
    k: Number of nearest users to consider (default=2)
    '''

    # Creating a MultiIndex DataFrame from the user ratings
    new_user = pd.DataFrame.from_dict({'rating': ratings}).unstack().rename('111111')

    # Appending the user to the rating matrix before calculating cosine similarity
    temp_matrix = rating_matrix.append(new_user)
    temp_matrix = temp_matrix.apply(lambda x: x - np.mean(x), axis=1).fillna(0)
    mc_df = user_similarity(temp_matrix)

    # Identifying the k nearest users
    temp_df = rating_matrix[rating_matrix.index.isin(list(mc_df.loc[:, '111111'].sort_values(ascending=False)[1:k + 1].index.values))]

    # Creating recommendations based on user-user cosine similarity excluding user read books
    rec_book = temp_df.loc[:, (temp_df != 0).any(axis=0)].mean().sort_values(ascending=False).index.get_level_values('book_id')
    user_read = ratings.keys()
    rec_bookid = [i for i in rec_book if i not in user_read]

    return books.iloc[[books[books['id'] == recid].index.values[0] for recid in rec_bookid]]['title']


def content_recommendation(ratings, k=2):
    '''
    Function to recommend books based on user's book ratings and their contents
    Ratings are centred to the mean rating before books which are positively rated are used
    to get book recommendations based on the cosine similarity of the book content
    Returns list of recommended books in rank order
    ratings: Dictionary of book_id and rating key:value pairs
    k: Number of books' content to consider based on rank order
      (first element is based on highest ratings) (default=1)
    '''

    # Creating a MultiIndex DataFrame from the user ratings
    new_user = pd.DataFrame.from_dict({'rating': ratings}).unstack().rename('111111')

    if len(ratings) == 1 or k == 1:
        # Creating content utility matrix via TF-IDF vectorizer
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        content_matrix = tfidf.fit_transform(bookcontent['content'])

        temp_df = linear_kernel(content_matrix, content_matrix)
        k, v = next(iter(ratings.items()))
        idx = pd.Series(books.index, index=books['id'])[k]
        book_index = sorted(list(enumerate(temp_df[idx])), key=lambda x: x[1], reverse=True)[1:21]
        return books.iloc[[i[0] for i in book_index]]['title']

    else:
        # Appending the user to the rating matrix
        temp_matrix = rating_matrix.append(new_user).fillna(0)

        book_list = []
        for i, n in enumerate(temp_matrix.loc['111111'].sort_values(ascending=False)):
            if n > 0:
                book_list.append(temp_matrix.loc['111111'].sort_values(ascending=False).index.get_level_values('book_id')[i])

        # Creating the book content based on the books selected for consideration
        new_content = ', '.join(bookcontent[bookcontent['id'].isin(book_list[:k])]['content'].values)

        # Appending content to the bookcontent DataFrame
        temp_content = bookcontent.append({
            'book_id': 111111,
            'matched_tag': 'User selected',
            'authors': 'User selected',
            'id': 111111,
            'content': new_content
        }, ignore_index=True)

        # Creating content utility matrix via TF-IDF vectorizer
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        content_matrix = tfidf.fit_transform(temp_content['content'])

        temp_df = linear_kernel(content_matrix, content_matrix)

        book_index = sorted(list(enumerate(temp_df[10000])), key=lambda x: x[1], reverse=True)[1:21]
        rec_book = books['id'].iloc[[i[0] for i in book_index]]
        user_read = ratings.keys()
        rec_bookid = [i for i in rec_book if i not in user_read]

        return books.iloc[[books[books['id'] == recid].index.values[0] for recid in rec_bookid]]['title']


def id_recommendation(user_id, k=2):
    '''
    Function to recommend books based on user id and nearest user cosine similarities
    Returns list of recommended books in rank order
    user_id: the user ID
    k: Number of nearest users to consider (default=2)
    '''

    # Calculating cosine similarity
    mc_df = user_similarity(rating_matrix)

    # Identifying the k nearest users
    temp_df = rating_matrix[rating_matrix.index.isin(list(mc_df.loc[:, user_id].sort_values(ascending=False)[1:k + 1].index.values))]

    # Creating recommendations based on user-user cosine similarity excluding user read books
    rec_book = temp_df.loc[:, (temp_df != 0).any(axis=0)].mean().sort_values(ascending=False).index.get_level_values('book_id')
    user_read = ratings.keys()
    rec_bookid = [i for i in rec_book if i not in user_read]

    return books.iloc[[books[books['id'] == recid].index.values[0] for recid in rec_bookid]]['title']
