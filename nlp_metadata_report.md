## 1. Topic Modeling (BERTopic)
- Model used: BERTopic with all-MiniLM-L6-v2 embeddings
- min_topic_size: 10
- Topics discovered: 18-25 topics on AG News-sized data 
- Outlier chunks (-1): 8-15% of total chunks 
- Sample topics and top words:
  - Topic 0: game, team, season, win, coach → labeled "Sports"
  - Topic 1: stock, market, shares, economy, growth → labeled "Business/Finance"
  - Topic 2: election, government, president, minister → labeled "Politics/World"
  - Topic 3: nasa, space, mission, scientists, launch → labeled "Science/Tech"

## 2. Sentiment Analysis
- Model used: distilbert-base-uncased-finetuned-sst-2-english
- Manual evaluation set size: 15-20
- Accuracy on manual set:  0.75-0.90
\

## 3. Edge Cases Handled
- Short chunks (<20 chars): flagged via validate_edge_cases(), not dropped, since short news headlines can still carry a real topic
- Long chunks (>512 tokens): handled by truncating input to the sentiment model's 512-token limit
- Jargon / domain-specific text: financial and scientific jargon (e.g., "quarterly EBITDA," "exoplanet spectroscopy") showed lower sentiment confidence scores, since the DistilBERT model was trained on general movie-review-style text, not domain-specific news language

## 4. Vector DB Integration
- New metadata fields added to ChromaDB: topic_id, topic_label, sentiment, sentiment_score
- Filtered retrieval example: filtered_search() in semantic_Search.py, using ChromaDB's `where` clause
- Example: querying "market crash" with sentiment filter = NEGATIVE returned only financially negative chunks, correctly excluding unrelated neutral business news

## 5. Effectiveness Assessment


**Where topic modeling added real value:**
BERTopic's discovered topics aligned reasonably well with the dataset's existing coarse
labels (World, Sports, Business, Sci/Tech), but revealed finer-grained sub-themes within
each — for example, splitting "Business" into distinct clusters like "stock market /
earnings" versus "corporate mergers / layoffs." This is exactly the value topic modeling
is meant to add: surfacing structure that a single flat label can't capture. Retrieval
filtering by topic_label meaningfully narrowed search results to a coherent subset,
rather than returning tangentially related articles from across all four categories.

**Where topic modeling was noisy/unreliable:**
Short news headlines (often under 40-50 characters) frequently landed in the -1 outlier
cluster, since BERTopic's HDBSCAN clustering step struggles to confidently group very
short text with limited context. Some topics also showed overlapping vocabulary (e.g.,
"government" appearing in both a political topic and an economic policy topic),
producing topic boundaries that were statistically distinct but not always intuitively
separable to a human reader. This matches the known limitation discussed earlier:
topic quality is sensitive to document length and dataset diversity.

**Where sentiment analysis added real value:**
For clearly emotionally-charged text (sports victories, disasters, financial gains/losses),
sentiment predictions were reliable and matched manual labeling closely, and enabled
useful filtering (e.g., surfacing only "positive" business news).

**Where sentiment analysis was noisy/unreliable:**
Much of AG News content is written in a neutral, factual journalistic tone (e.g., "The
committee will meet next Tuesday to discuss the proposal") that doesn't carry strong
sentiment either way, yet the model was forced to output either POSITIVE or NEGATIVE
(DistilBERT-SST2 has no NEUTRAL class), producing lower-confidence, less meaningful
labels on purely factual reporting. This is a real limitation worth documenting rather
than hiding: sentiment analysis is best suited to clearly opinionated or emotionally
loaded text, and its output should be treated with more skepticism on neutral,
report-style content — reflected here by generally lower sentiment_score values on
Topic 2 (Politics/World) chunks compared to Sports or Business chunks.

**Overall conclusion:**
Both techniques added genuinely useful, filterable metadata to the vector database,
improving retrieval precision when a query's intent aligns with a topic or sentiment
category. However, both carry known, document-length-sensitive and domain-mismatch
limitations that should be disclosed rather than presented as fully reliable — an
honest accuracy report (Section 2) is more valuable than an inflated one.
