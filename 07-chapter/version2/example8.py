import matplotlib.pyplot as plt
import networkx as nx
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

# 初始化向量存储,用于存储记忆
recall_vector_store = InMemoryVectorStore(DashScopeEmbeddings())

records = recall_vector_store.similarity_search(
    "莫尔索", k=2, filter=lambda doc: doc.metadata["user_id"] == "1"
)


# Plot graph
plt.figure(figsize=(6, 4), dpi=80)
G = nx.DiGraph()

for record in records:
    G.add_edge(
        record.metadata["subject"],
        record.metadata["object_"],
        label=record.metadata["predicate"],
    )

pos = nx.spring_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    font_weight="bold",
    arrows=True,
)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
plt.show()