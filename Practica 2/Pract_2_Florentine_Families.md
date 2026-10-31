# Practice 2 — Florentine Families: Multiplex Networks, Brokerage and Political Power

### Complex Network Features: Marriage and Business Ties, Multiplex Networks, Degree Centrality, Betweenness Centrality, Brokerage, NetworkX, Visualization

**Subject:** Social Networks (SNET)  
**Language:** English  
**Main libraries:** pandas, NetworkX, Matplotlib

## Objectives of the practice

1. Work with a classic historical social-network database: **Padgett's Florentine Families**.
2. Download and read a **two-layer multiplex network** with `pandas`.
3. Distinguish between **marriage alliances** and **business relationships**.
4. Construct separate and combined graphs with **NetworkX**.
5. Compare the topology of different relationship layers.
6. Calculate and interpret node degree, degree centrality, betweenness centrality, adjacency and clustering coefficient.
7. Analyze the **structural position of the Medici family**.
8. Introduce **brokerage**, **multiplexity**, and **political power**.
9. Produce an executive summary and an infographic.

Central question:

> **How can a family become politically powerful not only by having many connections, but by occupying a strategic position between otherwise weakly connected groups?**

##  About the use of AI

The Economist  (18 August 2026),  **Does AI stop children from learning?**

<img src="does-al-stop-children-from-learning-v0-m7gwmp4jkekh1.webp" alt="DoubleGraph" width="600"/>


##  One solution
**Learning How to Learn: How to Succeed in School Without Spending All Your Time Studying; A Guide for Kids and Teens** by Barbara Oakley, PhD, Terrence Sejnowski, PhD, and Alistair McConville (published 2018).

# 1. Description of the problem

Renaissance Florence was organized through overlapping systems of family alliances, economic relationships, and political competition. The same families could be connected through **marriage alliances**, **business relationships**, or both.

This is naturally represented as a **multiplex network**: the same node set participates in several relational layers.

The practical problem is to determine whether the structural position of a family can help explain its capacity to coordinate, broker, or mediate between different parts of the Florentine elite.

## <font color="blue"><b>1.1. Historical background</b></font>

The dataset contains a subset of prominent Florentine families involved in political competition around the early fifteenth century. The classic study is:

**Padgett, J. F. & Ansell, C. K. (1993). _Robust Action and the Rise of the Medici, 1400–1434_. American Journal of Sociology.**

The two relations are:

1. **Marriage alliances**
2. **Business relationships**, including recorded financial ties such as loans, credits and joint partnerships.

The data are conventionally represented as **undirected and unweighted** relations.

A major historical question is why the **Medici** became politically dominant despite the presence of other wealthy and influential families such as the **Strozzi**. Network analysis reformulates the problem structurally: did the Medici connect otherwise separated parts of the elite network?

## <font color="blue"><b>1.1.1 Additional Background background</b></font>

**Ferguson, N. (2017).** *_The Square and the Tower: Networks, Hierarchies and the Struggle for Global Power_*. **Allen Lane (Penguin Books)**.
- Explanation of the difference between vertical hierarchies (i.e. Tower), and flat/network hierarchies (i.e. the city square)

**Easley, D. & Kleinberg, J. (2010).** *_Networks, Crowds, and Markets: Reasoning about a Highly Connected World_*. **Cambridge University Press**.
- Good explanation of the concepts that appear in this practice.

## <font color="blue"><b>1.2. The database</b></font>

We use the openly available **Padgett–Florentine Families multiplex dataset** maintained by CoMuNeLab. The repository documents **16 family nodes**, **2 network layers**, and **35 multiplex connections**. The network is undirected and unweighted.


## <font color="blue"><b>1.3. Complex Network features used in this practice</b></font>

| Feature | Short explanation |
|---|---|
| **Node** | One Florentine family. |
| **Edge** | A recorded relationship between two families. |
| **Layer** | One type of relationship, such as marriage or business. |
| **Multiplex network** | A network in which the same nodes may be connected by several different kinds of edges. |
| **Adjacency** | Two families are adjacent when they share a direct tie in a given layer. |
| **Degree** | Number of direct ties of a node. |
| **Degree centrality** | Degree normalized by the maximum possible number of neighbors. |
| <font color="blue"><b>Betweenness centrality</b></font> | Measures how often a node lies on shortest paths between other nodes; often related to brokerage. |
| <font color="blue"><b>Clustering coefficient</b></font>  | Measures how interconnected a node's neighbors are. |
| **Brokerage** | A structural position connecting actors or groups that otherwise have few direct links. |
| **Visualization** | Graphical representation of families and their relationships. |

####  Comparison <font color="blue"><b>Betweenness centrality</b></font> vs  <font color="blue"><b>Clustering coefficient</b></font>.
<img src="ComparaFamilyBroker.png" alt="DoubleGraph" width="600"/>

## <font color="red"><b>Exercise 1</b></font>

1. What are the **nodes** in this database?
2. What are the two types of **edges**?
3. For clarifying concepts, in two sentences, explain why is this network better represented as a **multiplex network** (mariage and business relationships) than as a single ordinary graph?
4.  For clarifying concepts, , in two sentences, explain from the figure above the pros and cons of the concepts of <font color="blue"><b>Betweenness centrality</b></font> and  <font color="blue"><b>Clustering coefficient</b></font>.

## <font color="green"><b>Exercise 1</b></font>

1. The nodes are the Florentine families included in the dataset.
2. Marriage alliances or buisness relationships.
3. A multiplex network allows us to distinguish between marriage alliances and business relationships while studying the same families. A single ordinary graph may lose this information by combining both types of relationships into one layer.
4. A high clustering coefficient indicates strong local cohesion, which can increase trust and cooperation but may limit connections to other groups (image 1 - high clustering coefficient for all nodes). High betweenness centrality allows a family such as the Medici to act as a broker between groups, gaining strategic importance through its position in the network (image 2).

<img src="DoubleGraph.png" alt="DoubleGraph" width="600"/>

## <font color="red"><b>Exercise 2</b></font>

Suppose two families are connected by both a marriage alliance and a business relationship.

1. Sketch a simple graph of this case.
2. How many edges appear between them in a simple combined graph?
3. What information is lost when layers are collapsed?

## <font color="green"><b>Exercise 2</b></font>

1.

![image.png](Pract_2_Florentine_Families_files/image.png)

2. There are 8 edges between families, the sames as in a multiplex network. I don't understand the Exercise 2, I think the example is redundant, the importance of multiplex networks is shown better with other exercises.
3. When they are colapsed you don't know if the relations between the two families are of marriage alliances or business relationships, you only know that the families are related

# 3. Practical part

We will download the data, inspect them with `pandas`, map IDs to family names, separate the two layers, and construct NetworkX graphs.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
```

## 3.1. Read the database with pandas

We read from a pickle file the pandas dataframes.

- Note: If you do not know what is a "pickle", it is important that you check what it is and how it is used, which advantatges it has.


```python
with open("florentine_families.pkl", "rb") as f:
    loaded = pickle.load(f)

edges_raw = loaded["edges_raw"]
nodes     = loaded["nodes"]
layers    = loaded["layers"]

print("Loaded successfully from pickle")
print("edges_raw")
print(edges_raw.head(10))
print("Edges table shape:", edges_raw.shape)
print("nodes")
print(nodes.head(10))
print("Nodes table shape:", nodes.shape)
print("layers")
print(layers)
print("Layers table shape:", layers.shape)
```

    Loaded successfully from pickle
    edges_raw
       layer_id  source_id  target_id  weight
    0         1          1          9       1
    1         1          2          6       1
    2         1          2          7       1
    3         1          2          9       1
    4         1          3          5       1
    5         1          3          9       1
    6         1          4          7       1
    7         1          4         11       1
    8         1          4         15       1
    9         1          5         11       1
    Edges table shape: (35, 4)
    nodes
       nodeID  nodeLabel
    0       1  ACCIAIUOL
    1       2    ALBIZZI
    2       3  BARBADORI
    3       4   BISCHERI
    4       5  CASTELLAN
    5       6     GINORI
    6       7   GUADAGNI
    7       8  LAMBERTES
    8       9     MEDICI
    9      10      PAZZI
    Nodes table shape: (16, 2)
    layers
       layerID          layerLabel
    0        1  marriage_alliances
    1        2            business
    Layers table shape: (2, 2)
    

####  Safety check. 

- Always check for mistakes


```python
print("Duplicate edge rows:", edges_raw.duplicated().sum())
self_loops = edges_raw[edges_raw["source_id"] == edges_raw["target_id"]]
print("Self-loops:", len(self_loops))
all_ids = pd.unique(pd.concat([edges_raw["source_id"], edges_raw["target_id"]]))
print("Unique node IDs appearing in edges:", len(all_ids))
```

    Duplicate edge rows: 0
    Self-loops: 0
    Unique node IDs appearing in edges: 15
    

## <font color="red"><b>Exercise 3</b></font>

1. How many edge records are present?
2. How many belong to the marriage layer?
3. How many belong to the business layer?
4. Are there self-loops? Why a self-loop should be considered a mistake?
5. Why should these properties be checked before graph construction?

## <font color="green"><b>Exercise 3</b></font>

1. 35 (code below)
2. 20 (code below)
3. 15 (code below)
4. There are not self-loops, is checked in the code before. It can be considered a mistake beacuse is does not show meaningful relationship, so can be a data-quality problem
5. Because they may indicate errors in the original relationship data, and we want to ensure the network accurately represents real relationships.


```python
print("Number of edges: ", len(edges_raw))
marriage_edges = edges_raw[edges_raw["layer_id"] == 1]
print("Number of marriage edges: ", len(marriage_edges))
business_edges = edges_raw[edges_raw["layer_id"] == 2]
print("Number of business edges: ", len(business_edges))
```

    Number of edges:  35
    Number of marriage edges:  20
    Number of business edges:  15
    

## 3.3. Replace numeric IDs by family names


```python
node_map = dict(zip(nodes["nodeID"], nodes["nodeLabel"]))
layer_map = dict(zip(layers["layerID"], layers["layerLabel"]))

edges = edges_raw.copy()
edges["source"] = edges["source_id"].map(node_map)
edges["target"] = edges["target_id"].map(node_map)
edges["relation"] = edges["layer_id"].map(layer_map)

edges.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>layer_id</th>
      <th>source_id</th>
      <th>target_id</th>
      <th>weight</th>
      <th>source</th>
      <th>target</th>
      <th>relation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>ACCIAIUOL</td>
      <td>MEDICI</td>
      <td>marriage_alliances</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>ALBIZZI</td>
      <td>GINORI</td>
      <td>marriage_alliances</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>ALBIZZI</td>
      <td>GUADAGNI</td>
      <td>marriage_alliances</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>ALBIZZI</td>
      <td>MEDICI</td>
      <td>marriage_alliances</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>BARBADORI</td>
      <td>CASTELLAN</td>
      <td>marriage_alliances</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>BARBADORI</td>
      <td>MEDICI</td>
      <td>marriage_alliances</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>BISCHERI</td>
      <td>GUADAGNI</td>
      <td>marriage_alliances</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>4</td>
      <td>11</td>
      <td>1</td>
      <td>BISCHERI</td>
      <td>PERUZZI</td>
      <td>marriage_alliances</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>4</td>
      <td>15</td>
      <td>1</td>
      <td>BISCHERI</td>
      <td>STROZZI</td>
      <td>marriage_alliances</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>5</td>
      <td>11</td>
      <td>1</td>
      <td>CASTELLAN</td>
      <td>PERUZZI</td>
      <td>marriage_alliances</td>
    </tr>
  </tbody>
</table>
</div>




```python
families = nodes.sort_values("nodeID").copy()
families
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nodeID</th>
      <th>nodeLabel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>ACCIAIUOL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>ALBIZZI</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>BARBADORI</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>BISCHERI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>CASTELLAN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>GINORI</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>GUADAGNI</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>LAMBERTES</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>MEDICI</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>PAZZI</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>PERUZZI</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>PUCCI</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>RIDOLFI</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>SALVIATI</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>STROZZI</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>TORNABUON</td>
    </tr>
  </tbody>
</table>
</div>



## <font color="red"><b>Exercise 4</b></font>

Note: make short answers,  it is just to be aware.

1. Find the node IDs corresponding to **MEDICI** and **STROZZI**.  
- Note: These are the families that were historically opposed. 
2. Why is a mapping table useful even if NetworkX can work with numeric IDs?
3. Which representation is preferable for interpretation: IDs or names?

<font color="red"><b>Python tips</b></font> 

 Understand an practice the following python commands (Matlab lines to Python lines -> 30 to 1)
- dict(zip(nodes["nodeID"], nodes["nodeLabel"]))
- edges["source_id"].map(node_map)
- Why edges = edges_raw.copy()?  Check the meaning of  assignment by reference vs value

## <font color="green"><b>Exercise 4</b></font>

1. MEDICI = 9, STROZZI = 15
2. Becasue it's easier to work with for humans and analyze the results.
3. Names is easier to interpret

# 4. Constructing the three NetworkX graphs

We construct `G_marriage`, `G_business`, and `G_combined`. All 16 families are explicitly added to every graph so isolates, if any, are preserved.


```python
marriage_edges = edges[edges["relation"] == "marriage_alliances"].copy()
business_edges = edges[edges["relation"] == "business"].copy()

all_families = nodes["nodeLabel"].tolist()
G_marriage = nx.Graph(); G_marriage.add_nodes_from(all_families)
G_business = nx.Graph(); G_business.add_nodes_from(all_families)

G_marriage.add_edges_from(marriage_edges[["source", "target"]].itertuples(index=False, name=None))
G_business.add_edges_from(business_edges[["source", "target"]].itertuples(index=False, name=None))
G_combined = nx.compose(G_marriage, G_business)

print("Marriage graph:", G_marriage.number_of_nodes(), "nodes,", G_marriage.number_of_edges(), "edges")
print("Business graph:", G_business.number_of_nodes(), "nodes,", G_business.number_of_edges(), "edges")
print("Combined graph:", G_combined.number_of_nodes(), "nodes,", G_combined.number_of_edges(), "unique edges")
```

    Marriage graph: 16 nodes, 20 edges
    Business graph: 16 nodes, 15 edges
    Combined graph: 16 nodes, 27 unique edges
    

## 4.1. Basic network summaries


```python
def graph_summary(G, name):
    N, M = G.number_of_nodes(), G.number_of_edges()
    return {
        "network": name,
        "nodes": N,
        "edges": M,
        "average_degree": 2*M/N if N else np.nan,
        "density": nx.density(G),
        "components": nx.number_connected_components(G),
        "average_clustering": nx.average_clustering(G)
    }

summary = pd.DataFrame([
    graph_summary(G_marriage, "Marriage"),
    graph_summary(G_business, "Business"),
    graph_summary(G_combined, "Combined")
])
summary.round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>network</th>
      <th>nodes</th>
      <th>edges</th>
      <th>average_degree</th>
      <th>density</th>
      <th>components</th>
      <th>average_clustering</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Marriage</td>
      <td>16</td>
      <td>20</td>
      <td>2.500</td>
      <td>0.167</td>
      <td>2</td>
      <td>0.150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Business</td>
      <td>16</td>
      <td>15</td>
      <td>1.875</td>
      <td>0.125</td>
      <td>6</td>
      <td>0.298</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Combined</td>
      <td>16</td>
      <td>27</td>
      <td>3.375</td>
      <td>0.225</td>
      <td>2</td>
      <td>0.415</td>
    </tr>
  </tbody>
</table>
</div>



## <font color="red"><b>Exercise 5</b></font>

1. Which layer contains more edges?
2. Which layer is denser?
3. Which layer has the larger average degree?
4. Are all families connected in each layer?
5. Why does collapsing the layers generally increase density?


```python
pos = nx.spring_layout(G_marriage, seed=42)

degrees = dict(G_marriage.degree())
node_sizes = [150 + 45 * degrees[node] for node in G_marriage.nodes()]
node_colors = [degrees[node] for node in G_marriage.nodes()]

plt.figure(figsize=(11, 8))

nx.draw_networkx_nodes(
    G_marriage,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.YlOrRd,
    alpha=0.9
)
nx.draw_networkx_edges(G_marriage, pos, alpha=0.4)
nx.draw_networkx_labels(G_marriage, pos, font_size=9)

plt.title("Marriage layer — Node Size and Color Proportional to Degree")
plt.axis("off")
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_33_0.png)
    



```python
pos = nx.spring_layout(G_business, seed=42)

degrees = dict(G_business.degree())
node_sizes = [150 + 45 * degrees[node] for node in G_business.nodes()]
node_colors = [degrees[node] for node in G_business.nodes()]

plt.figure(figsize=(11, 8))

nx.draw_networkx_nodes(
    G_business,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.YlOrRd,
    alpha=0.9
)
nx.draw_networkx_edges(G_business, pos, alpha=0.4)
nx.draw_networkx_labels(G_business, pos, font_size=9)

plt.title("Business layer — Node Size and Color Proportional to Degree")
plt.axis("off")
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_34_0.png)
    


## <font color="green"><b>Exercise 5</b></font>

1. The Marriage layer contains more edges.
2. The Marriage layer is more dense, with a 0.167 density.
3. The Marriage layer has larger averagde degree (2.5)
4. Not all families are connected in both layers. In the Marriage layer Pucci is not connected to anyone. In the Business layer, Pucc, Acciaiuol, Ridolfi, Strozzi and Albizzi are not connected to anyone.
5. Because when collapsing layers you add edges in one layer that may not exist in the other layer, so is really probable that you are adding edges that did not exist in the other layer, and this in both ways (in this case in both layers).

# 5. Adjacency and multiplexity

Each layer has its own adjacency matrix. Two families have a **multiplex tie** when they are adjacent in both layers.


```python
A_marriage = nx.to_pandas_adjacency(G_marriage, nodelist=all_families, dtype=int)
A_business = nx.to_pandas_adjacency(G_business, nodelist=all_families, dtype=int)

print("Marriage adjacency matrix:")
display(A_marriage)
print("Business adjacency matrix:")
display(A_business)
```

    Marriage adjacency matrix:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ACCIAIUOL</th>
      <th>ALBIZZI</th>
      <th>BARBADORI</th>
      <th>BISCHERI</th>
      <th>CASTELLAN</th>
      <th>GINORI</th>
      <th>GUADAGNI</th>
      <th>LAMBERTES</th>
      <th>MEDICI</th>
      <th>PAZZI</th>
      <th>PERUZZI</th>
      <th>PUCCI</th>
      <th>RIDOLFI</th>
      <th>SALVIATI</th>
      <th>STROZZI</th>
      <th>TORNABUON</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ACCIAIUOL</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ALBIZZI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BARBADORI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BISCHERI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>CASTELLAN</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GINORI</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GUADAGNI</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LAMBERTES</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MEDICI</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PAZZI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PERUZZI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PUCCI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>RIDOLFI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SALVIATI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>STROZZI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TORNABUON</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    Business adjacency matrix:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ACCIAIUOL</th>
      <th>ALBIZZI</th>
      <th>BARBADORI</th>
      <th>BISCHERI</th>
      <th>CASTELLAN</th>
      <th>GINORI</th>
      <th>GUADAGNI</th>
      <th>LAMBERTES</th>
      <th>MEDICI</th>
      <th>PAZZI</th>
      <th>PERUZZI</th>
      <th>PUCCI</th>
      <th>RIDOLFI</th>
      <th>SALVIATI</th>
      <th>STROZZI</th>
      <th>TORNABUON</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ACCIAIUOL</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ALBIZZI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BARBADORI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BISCHERI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>CASTELLAN</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GINORI</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GUADAGNI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LAMBERTES</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MEDICI</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PAZZI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PERUZZI</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PUCCI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>RIDOLFI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SALVIATI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>STROZZI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TORNABUON</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# Load data (or use the already-loaded variables)
# -------------------------------------------------
BASE_URL = "https://raw.githubusercontent.com/CoMuNeLab/Padgett-Florentine-Families/main/Dataset/"
edges_raw = pd.read_csv(BASE_URL + "Padgett-Florentine-Families_multiplex.edges",
                        sep=r"\s+", header=None,
                        names=["layer_id", "source_id", "target_id", "weight"])
nodes  = pd.read_csv(BASE_URL + "Padgett-Florentine-Families_nodes.txt", sep=r"\s+")
layers = pd.read_csv(BASE_URL + "Padgett-Florentine-Families_layers.txt", sep=r"\s+")

# Map nodeID → family name
id2name = dict(zip(nodes["nodeID"], nodes["nodeLabel"]))
names   = [id2name[i] for i in sorted(id2name)]

# -------------------------------------------------
# Build the two graphs
# -------------------------------------------------
# Marriage (layer 1)
marriage_edges = edges_raw[edges_raw["layer_id"] == 1][["source_id", "target_id"]]
G_mar = nx.from_pandas_edgelist(marriage_edges, "source_id", "target_id")
G_mar = nx.relabel_nodes(G_mar, id2name)

# Business (layer 2)
business_edges = edges_raw[edges_raw["layer_id"] == 2][["source_id", "target_id"]]
G_bus = nx.from_pandas_edgelist(business_edges, "source_id", "target_id")
G_bus = nx.relabel_nodes(G_bus, id2name)

# Ensure all 16 families appear (even isolates)
for n in names:
    G_mar.add_node(n)
    G_bus.add_node(n)

# -------------------------------------------------
# Adjacency matrices (ordered by family name)
# -------------------------------------------------
A_mar = nx.to_pandas_adjacency(G_mar, nodelist=names, dtype=int)
A_bus = nx.to_pandas_adjacency(G_bus, nodelist=names, dtype=int)

# -------------------------------------------------
# Side-by-side heatmaps
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Marriage
im1 = axes[0].imshow(A_mar, cmap="Blues", vmin=0, vmax=1)
axes[0].set_title("Marriage alliances", fontsize=14)
axes[0].set_xticks(range(len(names)))
axes[0].set_yticks(range(len(names)))
axes[0].set_xticklabels(names, rotation=90, fontsize=8)
axes[0].set_yticklabels(names, fontsize=8)

# Business
im2 = axes[1].imshow(A_bus, cmap="Reds", vmin=0, vmax=1)
axes[1].set_title("Business relationships", fontsize=14)
axes[1].set_xticks(range(len(names)))
axes[1].set_yticks(range(len(names)))
axes[1].set_xticklabels(names, rotation=90, fontsize=8)
axes[1].set_yticklabels(names, fontsize=8)

fig.suptitle("Florentine Families – Adjacency Matrices", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_38_0.png)
    


## 5.1. Number of paths from one node to another: $A^N$
### Why does a power of the adjacency matrix count walks?
The **adjacency matrix** is that its powers provide information about the number of ways of moving between nodes in a network.

Consider an **unweighted graph** with adjacency matrix $A$, where

$A_{ij} = 1$  if nodes $i$ and $j$ are connected, and

$A_{ij} = 0$  otherwise.

The entry $A_{ij}$ tells us whether it is possible to move directly from node $i$ to node $j$ in **one step**.



### From $A$ to $A^2$

The square of the adjacency matrix:

$A^2 = A \cdot A$, then a given element {i,j} has the value  $(A^2)_{ij} = \sum_k A_{ik}A_{kj}$

Each product $A_{ik}A_{kj}$  is equal to 1 only when **both edges exist**:


Then, **$(A^2)_{ij}$ is the number of walks of length 2 from $i$ to $j$.**



### From $A^2$ to $A^3$

For three steps,

$A^3=A\cdot A\cdot A$,  has as  its entries $(A^3)_{ij} = \sum_{k,l} A_{ik}A_{kl}A_{lj}$

A product contributes 1 whenever all three required edges exist:

$i\rightarrow k\rightarrow l\rightarrow j$

Therefore,

**$(A^3)_{ij}$ is the number of walks of length 3 from $i$ to $j$.**



### Generalization to $A^N$

The same argument can be generalized to any positive integer $N$:

**$(A^N)_{ij}$ is the number of walks of length $N$ from node $i$ to node $j$.**

Thus:

| Matrix | Interpretation of entry $(i,j)$ |
|---|---|
| $A$ | Number of walks of length 1 from $i$ to $j$ |
| $A^2$ | Number of walks of length 2 |
| $A^3$ | Number of walks of length 3 |
| $\vdots$ | $\vdots$ |
| $A^N$ | Number of walks of length $N$ |




---

### Important: walks are not necessarily simple paths

Strictly speaking, powers of the adjacency matrix count **walks**, not necessarily **simple paths**.

A **walk** is allowed to visit the same node or edge more than once.

For example,

$A \rightarrow B \rightarrow C \rightarrow B \rightarrow D$

is a valid walk of length 4, even though node $B$ appears twice.

A **simple path**, in contrast, does not repeat nodes.

Therefore, the mathematically precise statement is:

**$(A^N)_{ij}$ gives the number of walks of length exactly $N$ from node $i$ to node $j$.**

It does not, in general, give the number of simple paths.

---

### Application to the Florentine Families network

For the Florentine Families network, calculating

$A^5$

allows us to investigate how many different **five-step walks** connect every pair of families.

A high value of

$(A^5)_{ij}$

means that there are many different sequences of five relationships through which family $i$ can reach family $j$.

This provides information about **indirect connectivity and network structure** that cannot be observed by examining only the immediate relationships represented by $A$.

### 5.1.1 Identify relationships present in both layers


```python
marriage_pairs = {frozenset(e) for e in G_marriage.edges()}
business_pairs = {frozenset(e) for e in G_business.edges()}
both_pairs = marriage_pairs & business_pairs
marriage_only = marriage_pairs - business_pairs
business_only = business_pairs - marriage_pairs

multiplex_summary = pd.DataFrame({
    "tie_type": ["Marriage only", "Business only", "Both layers"],
    "number_of_pairs": [len(marriage_only), len(business_only), len(both_pairs)]
})
multiplex_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tie_type</th>
      <th>number_of_pairs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Marriage only</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Business only</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Both layers</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
both_edges_df = pd.DataFrame(
    [sorted(tuple(pair)) for pair in both_pairs],
    columns=["family_1", "family_2"]
).sort_values(["family_1", "family_2"]).reset_index(drop=True)
both_edges_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family_1</th>
      <th>family_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BARBADORI</td>
      <td>CASTELLAN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BARBADORI</td>
      <td>MEDICI</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BISCHERI</td>
      <td>GUADAGNI</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BISCHERI</td>
      <td>PERUZZI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CASTELLAN</td>
      <td>PERUZZI</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GUADAGNI</td>
      <td>LAMBERTES</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MEDICI</td>
      <td>SALVIATI</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MEDICI</td>
      <td>TORNABUON</td>
    </tr>
  </tbody>
</table>
</div>



### 5.1.2 Paths between nodes.
- Now connected are two families by:
    - Mariage
    - Business

**Meaning:** How many times one gets to a node if one randomly moves on the graph, i.e. how connected they are


```python
import numpy as np
import matplotlib.pyplot as plt

# A_mar and A_bus are the pandas adjacency matrices from the previous cell
# Convert to numpy for matrix power
A_mar_np = A_mar.values.astype(float)
A_bus_np = A_bus.values.astype(float)

# Compute A^N for each layer
N = 15
A_mar_15 = np.linalg.matrix_power(A_mar_np, N)
A_bus_15 = np.linalg.matrix_power(A_bus_np, N)

# -------------------------------------------------
# Side-by-side plots of A^15
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Marriage A^15
im1 = axes[0].imshow(A_mar_15, cmap="Blues")
axes[0].set_title(r"Marriage  $A^{15}$", fontsize=14)
axes[0].set_xticks(range(len(names)))
axes[0].set_yticks(range(len(names)))
axes[0].set_xticklabels(names, rotation=90, fontsize=8)
axes[0].set_yticklabels(names, fontsize=8)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Business A^15
im2 = axes[1].imshow(A_bus_15, cmap="Reds")
axes[1].set_title(r"Business  $A^{15}$", fontsize=14)
axes[1].set_xticks(range(len(names)))
axes[1].set_yticks(range(len(names)))
axes[1].set_xticklabels(names, rotation=90, fontsize=8)
axes[1].set_yticklabels(names, fontsize=8)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

fig.suptitle(r"Florentine Families – Number of walks of length 15 ($A^{15}$)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_44_0.png)
    


### Order the families according to the total number of walks of length \(N\) in which they appear as the starting node. 
- For each family \(i\), this is the row sum of \(A^N\): $W_i^{(N)} = \sum_j (A^N)_{ij}$


```python
import numpy as np
import matplotlib.pyplot as plt

# Convert adjacency matrices to NumPy
A_mar_np = A_mar.values.astype(float)
A_bus_np = A_bus.values.astype(float)

# Compute A^N
N = 15
A_mar_N = np.linalg.matrix_power(A_mar_np, N)
A_bus_N = np.linalg.matrix_power(A_bus_np, N)

# Total number of walks starting from each family
walks_mar = A_mar_N.sum(axis=1)
walks_bus = A_bus_N.sum(axis=1)

# Order families from highest to lowest number of walks
order_mar = np.argsort(walks_mar)[::-1]
order_bus = np.argsort(walks_bus)[::-1]

# Reorder rows and columns
A_mar_ordered = A_mar_N[np.ix_(order_mar, order_mar)]
A_bus_ordered = A_bus_N[np.ix_(order_bus, order_bus)]

names_mar = np.array(names)[order_mar]
names_bus = np.array(names)[order_bus]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].imshow(A_mar_ordered, cmap="Blues")
axes[0].set_title(rf"Marriage  $A^{{{N}}}$")
axes[0].set_xticks(range(len(names_mar)))
axes[0].set_yticks(range(len(names_mar)))
axes[0].set_xticklabels(names_mar, rotation=90, fontsize=8)
axes[0].set_yticklabels(names_mar, fontsize=8)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(A_bus_ordered, cmap="Reds")
axes[1].set_title(rf"Business  $A^{{{N}}}$")
axes[1].set_xticks(range(len(names_bus)))
axes[1].set_yticks(range(len(names_bus)))
axes[1].set_xticklabels(names_bus, rotation=90, fontsize=8)
axes[1].set_yticklabels(names_bus, fontsize=8)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

fig.suptitle(
    rf"Florentine Families – Walks of length {N} ($A^{{{N}}}$), ordered by total walks"
    "\n"
    rf"$W_i^{{({N})}} = \sum_j (A^{{{N}}})_{{ij}}$",
    fontsize=16, y=1.05
)

plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_46_0.png)
    


## <font color="red"><b>Exercise 6</b></font>

1. How many pairs are connected only by marriage in the Adjencency Matrix?. Which are the most connected by number of paths?
2. How many only by business?. Which are the most connected by number of paths?
3. How many are connected in both layers in the adjacency matrices? Is it inherited in the number of paths matrix?



```python
print("Marriage: ", len(marriage_only))
print("Business: ", len(business_only))
```

    Marriage:  12
    Business:  7
    


```python
import numpy as np
import matplotlib.pyplot as plt

# Convert adjacency matrices to NumPy
# A_mar = A_mar.values.astype(float)
# A_bus = A_bus.values.astype(float)

# Total number of walks starting from each family
walks_mar = A_mar.sum(axis=1)
walks_bus = A_bus.sum(axis=1)

# Order families from highest to lowest number of walks
order_mar = np.argsort(walks_mar)[::-1]
order_bus = np.argsort(walks_bus)[::-1]

# Reorder rows and columns
A_mar_ordered = A_mar[np.ix_(order_mar, order_mar)]
A_bus_ordered = A_bus[np.ix_(order_bus, order_bus)]

names_mar = np.array(names)[order_mar]
names_bus = np.array(names)[order_bus]

# Plot
N = 1
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].imshow(A_mar_ordered, cmap="Blues")
axes[0].set_title(rf"Marriage  $A^{{{N}}}$")
axes[0].set_xticks(range(len(names_mar)))
axes[0].set_yticks(range(len(names_mar)))
axes[0].set_xticklabels(names_mar, rotation=90, fontsize=8)
axes[0].set_yticklabels(names_mar, fontsize=8)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(A_bus_ordered, cmap="Reds")
axes[1].set_title(rf"Business  $A^{{{N}}}$")
axes[1].set_xticks(range(len(names_bus)))
axes[1].set_yticks(range(len(names_bus)))
axes[1].set_xticklabels(names_bus, rotation=90, fontsize=8)
axes[1].set_yticklabels(names_bus, fontsize=8)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

fig.suptitle(
"Florentine Families – Walks of length 1 (A), ordered by total walks"
    ,fontsize=16, y=1.05
)

plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_49_0.png)
    


## <font color="green"><b>Exercise 6</b></font>

1. There are 12 pairs only connected in the Marriage layer. The once with most paths are the Medici with 6, the Strozzi and the Guadagni and with 4 and then the Ridolfi, the Peruzzi, the Albizzi, the Castellan, the Tornaboun and the Bizceri with 3. This taking the Adjacency Matrix of length 1.
2. There are 7 pairs only connected in the Business layer. The once with most paths are the Medici with 6, the Peruzzi, Barbadori and Lamberts with 4, followed by the Castellan ans Bischeri with 3. This taking the Adjacency Matrix of length 1.
3. There are 8 pairs connected in both layers. Yes it is inherited, because the families that are connected in both layers are all in between the most connected nodes when working with the adjacency matrix of length 15, in exeption of the Strozzi, that is not in both layers but is in the second place of the marriadeg adgacency matrix of length 15. 


# 6. Network visualization

Use the same node positions for every layer so that differences in the figures reflect network structure rather than a different layout.


```python
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Spring layout
pos_spring = nx.spring_layout(G_combined, seed=42)
nx.draw_networkx(G_marriage, pos=pos_spring, ax=axes[0], with_labels=True, node_size=900, font_size=9, width=1.5, node_color="skyblue", alpha=0.6, edgecolors="black")
axes[0].set_title("Marriage Alliances — Spring Layout")
axes[0].axis("off")

# Right: Circular layout
pos_circle = nx.circular_layout(G_combined)
nx.draw_networkx(G_marriage, pos=pos_circle, ax=axes[1], with_labels=True, node_size=900, font_size=9, width=1.5, node_color="skyblue", alpha=0.6, edgecolors="black")
axes[1].set_title("Marriage Alliances — Circular Layout")
axes[1].axis("off")

plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_52_0.png)
    



```python
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Spring layout
pos_spring = nx.spring_layout(G_combined, seed=42)
nx.draw_networkx(G_business, pos=pos_spring, ax=axes[0], with_labels=True, node_size=900, font_size=9, width=1.5, node_color="lightcoral", alpha=0.6, edgecolors="black")
axes[0].set_title("Business Relationships — Spring Layout")
axes[0].axis("off")

# Right: Circular layout
pos_circle = nx.circular_layout(G_combined)
nx.draw_networkx(G_business, pos=pos_circle, ax=axes[1], with_labels=True, node_size=900, font_size=9, width=1.5, node_color="lightcoral", alpha=0.6, edgecolors="black")
axes[1].set_title("Business Relationships — Circular Layout")
axes[1].axis("off")

plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_53_0.png)
    


## 6.1. Multiplex visualization

Solid edges represent marriage; dashed edges represent business.


```python
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# -------------------------------------------------
# Left: Spring layout
# -------------------------------------------------
pos_spring = nx.spring_layout(G_combined, seed=42)

nx.draw_networkx_nodes(G_combined, pos_spring, ax=axes[0], node_size=950, node_color="lightgray", alpha=0.7, edgecolors="black")
nx.draw_networkx_labels(G_combined, pos_spring, ax=axes[0], font_size=9)
nx.draw_networkx_edges(G_marriage, pos_spring, ax=axes[0], width=2.0, alpha=0.7, edge_color="blue", label="Marriage")
nx.draw_networkx_edges(G_business, pos_spring, ax=axes[0], width=1.5, alpha=0.7, style="dashed", edge_color="red", label="Business")
axes[0].set_title("Multiplex View — Spring Layout")
axes[0].axis("off")
axes[0].legend()

# -------------------------------------------------
# Right: Circular layout
# -------------------------------------------------
pos_circle = nx.circular_layout(G_combined)

nx.draw_networkx_nodes(G_combined, pos_circle, ax=axes[1], node_size=950, node_color="lightgray", alpha=0.7, edgecolors="black")
nx.draw_networkx_labels(G_combined, pos_circle, ax=axes[1], font_size=9)
nx.draw_networkx_edges(G_marriage, pos_circle, ax=axes[1], width=2.0, alpha=0.7, edge_color="blue", label="Marriage")
nx.draw_networkx_edges(G_business, pos_circle, ax=axes[1], width=1.5, alpha=0.7, style="dashed", edge_color="red", label="Business")
axes[1].set_title("Multiplex View — Circular Layout")
axes[1].axis("off")
axes[1].legend()

plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_55_0.png)
    


## <font color="red"><b>Exercise 7</b></font>

1. Which families appear visually central in the marriage network?
2. Which appear central in the business network?
3. Does MEDICI occupy the same structural position in both layers?
4. Identify a family that is more prominent in one layer than the other.
5. Why is a common layout useful for comparison? Are there shorter paths when both networks are joined together?

## <font color="green"><b>Exercise 7</b></font>

1. The families: Medici, Albizzi, Ridolfi and Guadagni.
2. The families: Medici, Barbadori and Ginori.
3. Yes, the Medici family ocuccups a similar central position in both layers.
4. The Ridolfi family is central and important node of connections in the marriage layer but has no edges in the business layer.
5. Common layout is useful for comparison beacuse it keeps the position of the families the same for both networks, and this makes easier to identify similarities and differences between marriage and business. Yes both networks are joined there are shorter paths, beacuse appears edges that colud not appear before, a connection that was 3 steps in marriage layer, with the addition of the business layer maight turn to 1 step.

# 7. Degree and degree centrality

For an undirected graph, degree centrality is

$
C_D(i)=\frac{k_i}{N-1}.
$

Degree is a **local** measure because it considers direct connections only.


```python
dcm = nx.degree_centrality(G_marriage)
dcb = nx.degree_centrality(G_business)
dcc = nx.degree_centrality(G_combined)

degree_table = pd.DataFrame({
    "family": all_families,
    "degree_marriage": [G_marriage.degree(f) for f in all_families],
    "degree_business": [G_business.degree(f) for f in all_families],
    "degree_combined": [G_combined.degree(f) for f in all_families],
    "degree_centrality_marriage": [dcm[f] for f in all_families],
    "degree_centrality_business": [dcb[f] for f in all_families],
    "degree_centrality_combined": [dcc[f] for f in all_families]
})
degree_table.sort_values("degree_combined", ascending=False).reset_index(drop=True).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family</th>
      <th>degree_marriage</th>
      <th>degree_business</th>
      <th>degree_combined</th>
      <th>degree_centrality_marriage</th>
      <th>degree_centrality_business</th>
      <th>degree_centrality_combined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MEDICI</td>
      <td>6</td>
      <td>5</td>
      <td>8</td>
      <td>0.400</td>
      <td>0.333</td>
      <td>0.533</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PERUZZI</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0.200</td>
      <td>0.267</td>
      <td>0.333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BISCHERI</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>0.200</td>
      <td>0.200</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BARBADORI</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>0.133</td>
      <td>0.267</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LAMBERTES</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0.067</td>
      <td>0.267</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>5</th>
      <td>STROZZI</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>0.267</td>
      <td>0.000</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GUADAGNI</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>0.267</td>
      <td>0.133</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CASTELLAN</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>0.200</td>
      <td>0.200</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TORNABUON</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0.200</td>
      <td>0.067</td>
      <td>0.200</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GINORI</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0.067</td>
      <td>0.133</td>
      <td>0.200</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ALBIZZI</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0.200</td>
      <td>0.000</td>
      <td>0.200</td>
    </tr>
    <tr>
      <th>11</th>
      <td>RIDOLFI</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0.200</td>
      <td>0.000</td>
      <td>0.200</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SALVIATI</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.133</td>
      <td>0.067</td>
      <td>0.133</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PAZZI</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.067</td>
      <td>0.067</td>
      <td>0.133</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ACCIAIUOL</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.067</td>
      <td>0.000</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PUCCI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>



<font color="red"><b>Python tips</b></font> 
Understand the 'sort_values' command, i.e. degree_table.sort_values("degree_combined", ascending=False)

## 7.1. Marriage degree versus business degree


```python
plt.figure(figsize=(10,7))
plt.scatter(degree_table["degree_marriage"], degree_table["degree_business"], s=90, alpha=0.75)
for _, row in degree_table.iterrows():
    plt.annotate(row["family"], (row["degree_marriage"], row["degree_business"]),
                 xytext=(5,5), textcoords="offset points", fontsize=8)
plt.xlabel("Marriage Degree"); plt.ylabel("Business Degree")
plt.title("Marriage Degree vs Business Degree")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_62_0.png)
    


## <font color="red"><b>Exercise 8</b></font>

1. Which family has the highest marriage degree?
2. Which has the highest business degree?
3. Which families are highly connected in both layers?
4. Does the family with the highest total degree necessarily have the greatest brokerage potential?
5. Why may degree alone be insufficient for analyzing political power?
6. Comment briefly on the scatter plot Marriage Degree vs Business Degree. Note that powerfull families like Strozzi and Lambertes have power due to different origins.

## <font color="green"><b>Exercise 8</b></font>

1. The Medici family has the highest marriage degree, with 6.
2. The Medici family has the highest business degree, with 5.
3. Medici is the family with most connections it both layers, with 8 connections, and the second one is Peruzzi with 5 combined connections.
4. Not necessary, but in this case is true, the Medici family is the most connected and also is the one with highest brokerage. But can be a case with 9 nodes, 4 conneted to each other and 4 conneted to each other and then 1 only node connecting bot groups, in this case this node is the less dense but has the hoghest brokerage.
5. Becasue in polithics is importante to have contacas, but it is also really imporant to be a brokerage node, to have links between different groups.
6. In the scatter plot we can se why Medici family was the most influenctial, and also we can note why Strozzi that has a lof of marriages and Lambertes that hat a high busines power were not the most influential families, and this is beacuse they had a grate power (connections) in one layer but insufficient in the other layer.

# 8. Betweenness centrality and brokerage

**Betweenness centrality** measures how often a node lies on shortest paths between other nodes:

$
C_B(v)=\sum_{s
eq v
eq t}\frac{\sigma_{st}(v)}{\sigma_{st}}.
$

A node with high betweenness can occupy a **brokerage position** between groups. A broker may connect otherwise separated actors, mediate access, coordinate information, bridge factions, or gain strategic autonomy by avoiding complete dependence on one dense cluster.


```python
bet_marriage = nx.betweenness_centrality(G_marriage, normalized=True)
bet_business = nx.betweenness_centrality(G_business, normalized=True)
bet_combined = nx.betweenness_centrality(G_combined, normalized=True)

centrality_table = degree_table.copy()
centrality_table["betweenness_marriage"] = centrality_table["family"].map(bet_marriage)
centrality_table["betweenness_business"] = centrality_table["family"].map(bet_business)
centrality_table["betweenness_combined"] = centrality_table["family"].map(bet_combined)
centrality_table.sort_values("betweenness_combined", ascending=False).reset_index(drop=True).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family</th>
      <th>degree_marriage</th>
      <th>degree_business</th>
      <th>degree_combined</th>
      <th>degree_centrality_marriage</th>
      <th>degree_centrality_business</th>
      <th>degree_centrality_combined</th>
      <th>betweenness_marriage</th>
      <th>betweenness_business</th>
      <th>betweenness_combined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MEDICI</td>
      <td>6</td>
      <td>5</td>
      <td>8</td>
      <td>0.400</td>
      <td>0.333</td>
      <td>0.533</td>
      <td>0.452</td>
      <td>0.229</td>
      <td>0.413</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BARBADORI</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>0.133</td>
      <td>0.267</td>
      <td>0.267</td>
      <td>0.081</td>
      <td>0.238</td>
      <td>0.158</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GUADAGNI</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>0.267</td>
      <td>0.133</td>
      <td>0.267</td>
      <td>0.221</td>
      <td>0.000</td>
      <td>0.112</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIDOLFI</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0.200</td>
      <td>0.000</td>
      <td>0.200</td>
      <td>0.098</td>
      <td>0.000</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PERUZZI</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0.200</td>
      <td>0.267</td>
      <td>0.333</td>
      <td>0.019</td>
      <td>0.129</td>
      <td>0.061</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ALBIZZI</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0.200</td>
      <td>0.000</td>
      <td>0.200</td>
      <td>0.184</td>
      <td>0.000</td>
      <td>0.059</td>
    </tr>
    <tr>
      <th>6</th>
      <td>STROZZI</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>0.267</td>
      <td>0.000</td>
      <td>0.267</td>
      <td>0.089</td>
      <td>0.000</td>
      <td>0.054</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TORNABUON</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0.200</td>
      <td>0.067</td>
      <td>0.200</td>
      <td>0.079</td>
      <td>0.000</td>
      <td>0.052</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CASTELLAN</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>0.200</td>
      <td>0.200</td>
      <td>0.267</td>
      <td>0.048</td>
      <td>0.048</td>
      <td>0.033</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LAMBERTES</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0.067</td>
      <td>0.267</td>
      <td>0.267</td>
      <td>0.000</td>
      <td>0.057</td>
      <td>0.032</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BISCHERI</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>0.200</td>
      <td>0.200</td>
      <td>0.267</td>
      <td>0.090</td>
      <td>0.024</td>
      <td>0.031</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GINORI</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0.067</td>
      <td>0.133</td>
      <td>0.200</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.012</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ACCIAIUOL</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.067</td>
      <td>0.000</td>
      <td>0.067</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PAZZI</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.067</td>
      <td>0.067</td>
      <td>0.133</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PUCCI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SALVIATI</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.133</td>
      <td>0.067</td>
      <td>0.133</td>
      <td>0.124</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Sort by combined betweenness
df_plot = centrality_table.sort_values("betweenness_combined", ascending=False).reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.bar(df_plot["family"], df_plot["betweenness_combined"], color="seagreen", edgecolor="black", alpha=0.8)

plt.ylabel("Normalized Betweenness Centrality")
plt.title("Betweenness Centrality — Combined Network")
plt.xticks(rotation=90, fontsize=9)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_67_0.png)
    


## 8.1. Degree versus betweenness

This scatter plot separates **many direct ties** from **strategic position between others**.


```python
plt.figure(figsize=(8,8))
plt.scatter(centrality_table["degree_combined"], centrality_table["betweenness_combined"], s=100, alpha=0.75)
for _, row in centrality_table.iterrows():
    plt.annotate(row["family"], (row["degree_combined"], row["betweenness_combined"]),
                 xytext=(5,5), textcoords="offset points", fontsize=9)
plt.xlabel("Degree in Combined Network"); plt.ylabel("Betweenness Centrality")
plt.title("Degree vs Betweenness — Combined Florentine Network")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_69_0.png)
    


## <font color="red"><b>Exercise 9</b></font>

1. Which family has the highest betweenness in the combined network? Make a short reflection based on the paths observed in $A^{5}$.
2. Compare MEDICI and STROZZI.
3. Does the family with highest betweenness also have the highest degree?
4. Explain the difference between **connectivity** and **brokerage**.
- Note: 
**Connectivity** refers to how many direct relationships an actor has in the network. In Network Science, this is closely related to **degree**. A highly connected family has many direct links to other families.

**Brokerage** refers to occupying a strategic position between actors or groups that are not strongly connected to each other. A broker may therefore have fewer total links than another actor, but its links connect otherwise separated parts of the network.


$$
\text{Connectivity} \approx \text{How many actors do I know?}
$$

$$
\text{Brokerage} \approx \text{Which otherwise separated actors or groups do I connect?}
$$


5. Why might brokerage provide a political advantage?


```python
import numpy as np
import matplotlib.pyplot as plt

# A_mar and A_bus are the pandas adjacency matrices from the previous cell
# Convert to numpy for matrix power
# A_mar_np = A_mar.values.astype(float)
# A_bus_np = A_bus.values.astype(float)

# Compute A^N for each layer
N = 5
A_mar_5 = np.linalg.matrix_power(A_mar_np, N)
A_bus_5 = np.linalg.matrix_power(A_bus_np, N)

# -------------------------------------------------
# Side-by-side plots of A^5
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Marriage A^5
im1 = axes[0].imshow(A_mar_5, cmap="Blues")
axes[0].set_title(r"Marriage  $A^{5}$", fontsize=14)
axes[0].set_xticks(range(len(names)))
axes[0].set_yticks(range(len(names)))
axes[0].set_xticklabels(names, rotation=90, fontsize=8)
axes[0].set_yticklabels(names, fontsize=8)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Business A^5
im2 = axes[1].imshow(A_bus_5, cmap="Reds")
axes[1].set_title(r"Business  $A^{5}$", fontsize=14)
axes[1].set_xticks(range(len(names)))
axes[1].set_yticks(range(len(names)))
axes[1].set_xticklabels(names, rotation=90, fontsize=8)
axes[1].set_yticklabels(names, fontsize=8)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

fig.suptitle(r"Florentine Families – Number of walks of length 5 ($A^{5}$)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_71_0.png)
    


## <font color="green"><b>Exercise 9</b></font>

1. The Medici family has the highest betweenness in the combined network. In the $A^{5}$ matrix we can see that Medici family has a high brokerage beacuse with a legth of 5 is the one with most paths, and in business layer is also important, in this case not the most important, but also important, this makes the Medici family the most influential family in Florece beacuse it connects the other important families int both important social groups (layers in this case).
2. Strozzi has a high degree in Marriage layer and also has a considerable value of betweenness, but for the Business layer it is horrible, this is the highest difference in comaprison to the Medici family.
3. In this case yes, but as said before is not necessary.
4. Connectivity is the number of direct connections a node has. Brokerage is the number of paths that go through your node in order to connect other nodes.
5. Becasue in polithics is really important to be a union point to differenet ideologies and groups, it has the power to speak with both and also to group them or to know both next steps.

# 9. Focus on the Medici family

We now compare **MEDICI** and **STROZZI**. Network measures describe structural opportunities and constraints; they do not by themselves prove historical causality.


```python
focus = centrality_table[centrality_table["family"].isin(["MEDICI", "STROZZI"])].copy()
focus[["family", "degree_marriage", "degree_business", "degree_combined",
       "betweenness_marriage", "betweenness_business", "betweenness_combined"]].round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family</th>
      <th>degree_marriage</th>
      <th>degree_business</th>
      <th>degree_combined</th>
      <th>betweenness_marriage</th>
      <th>betweenness_business</th>
      <th>betweenness_combined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>MEDICI</td>
      <td>6</td>
      <td>5</td>
      <td>8</td>
      <td>0.452</td>
      <td>0.229</td>
      <td>0.413</td>
    </tr>
    <tr>
      <th>14</th>
      <td>STROZZI</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>0.089</td>
      <td>0.000</td>
      <td>0.054</td>
    </tr>
  </tbody>
</table>
</div>



## 9.1. Ego networks

An **ego network** contains a focal node, its direct neighbors, and the relationships among those neighbors. It can reveal whether contacts form a closed circle or extend into structurally diverse regions.


```python
medici_ego = nx.ego_graph(G_combined, "MEDICI")
strozzi_ego = nx.ego_graph(G_combined, "STROZZI")

print("MEDICI ego network:", medici_ego.number_of_nodes(), "nodes,", medici_ego.number_of_edges(), "edges")
print("Neighbors:", sorted(G_combined.neighbors("MEDICI")))
print("\nSTROZZI ego network:", strozzi_ego.number_of_nodes(), "nodes,", strozzi_ego.number_of_edges(), "edges")
print("Neighbors:", sorted(G_combined.neighbors("STROZZI")))
```

    MEDICI ego network: 9 nodes, 12 edges
    Neighbors: ['ACCIAIUOL', 'ALBIZZI', 'BARBADORI', 'GINORI', 'PAZZI', 'RIDOLFI', 'SALVIATI', 'TORNABUON']
    
    STROZZI ego network: 5 nodes, 6 edges
    Neighbors: ['BISCHERI', 'CASTELLAN', 'PERUZZI', 'RIDOLFI']
    


```python
# MEDICI and STROZZI ego networks side by side

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# MEDICI ego network
pos_medici = nx.spring_layout(medici_ego, seed=42)

nx.draw_networkx(medici_ego, pos=pos_medici, ax=axes[0], with_labels=True, node_size=1100, font_size=9, width=1.5, node_color="skyblue", alpha=0.6)

axes[0].set_title("MEDICI Ego Network — Combined Relations")
axes[0].axis("off")


# STROZZI ego network
pos_strozzi = nx.spring_layout(strozzi_ego, seed=42)

nx.draw_networkx(strozzi_ego, pos=pos_strozzi, ax=axes[1], with_labels=True, node_size=1100, font_size=9, width=1.5, node_color="skyblue", alpha=0.6)

axes[1].set_title("STROZZI Ego Network — Combined Relations")
axes[1].axis("off")


plt.tight_layout()
plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_77_0.png)
    


## <font color="red"><b>Exercise 10</b></font>

1. How many direct neighbors does each family have?
2. Are their neighbors highly interconnected?
3. Which ego network appears more closed?
4. Which appears to connect more structurally diverse actors?
5. Relate your observations to betweenness centrality.

## <font color="green"><b>Exercise 10</b></font>

1. The Medici family has 8 direct neighbors, and the Strozzi family has 4.
2. The Medici neighbours are almost all connectes with one neighbour more with exception of one, in the Strozzi family there are 3 neighbour connected and 1 not connected to nor the others.
3. The Strozzi network appears more closed, less nodes also.
4. The Medici network appears more structurally diverse, having 8 neighbours.
5. As we saw in earlier questions, the more diverse netwokr has a hghest betweenness centrality and the one more closed has less.

# 10. Clustering coefficient and closed local circles

The clustering coefficient distinguishes a node whose neighbors all know one another from a node whose neighbors belong to different regions. A family with high degree but relatively low clustering may connect groups that are not strongly linked to one another — a pattern compatible with brokerage.


```python
clustering_combined = nx.clustering(G_combined)
brokerage_table = centrality_table.copy()
brokerage_table["clustering_combined"] = brokerage_table["family"].map(clustering_combined)

brokerage_table[["family", "degree_combined", "betweenness_combined", "clustering_combined"]]     .sort_values("betweenness_combined", ascending=False).reset_index(drop=True).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family</th>
      <th>degree_combined</th>
      <th>betweenness_combined</th>
      <th>clustering_combined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MEDICI</td>
      <td>8</td>
      <td>0.413</td>
      <td>0.143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BARBADORI</td>
      <td>4</td>
      <td>0.158</td>
      <td>0.333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GUADAGNI</td>
      <td>4</td>
      <td>0.112</td>
      <td>0.167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIDOLFI</td>
      <td>3</td>
      <td>0.071</td>
      <td>0.333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PERUZZI</td>
      <td>5</td>
      <td>0.061</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ALBIZZI</td>
      <td>3</td>
      <td>0.059</td>
      <td>0.333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>STROZZI</td>
      <td>4</td>
      <td>0.054</td>
      <td>0.333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TORNABUON</td>
      <td>3</td>
      <td>0.052</td>
      <td>0.333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CASTELLAN</td>
      <td>4</td>
      <td>0.033</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LAMBERTES</td>
      <td>4</td>
      <td>0.032</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BISCHERI</td>
      <td>4</td>
      <td>0.031</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GINORI</td>
      <td>3</td>
      <td>0.012</td>
      <td>0.667</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ACCIAIUOL</td>
      <td>1</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PAZZI</td>
      <td>2</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PUCCI</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SALVIATI</td>
      <td>2</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(11,8))
plt.scatter(brokerage_table["clustering_combined"], brokerage_table["betweenness_combined"], s=100, alpha=0.75)
for _, row in brokerage_table.iterrows():
    plt.annotate(row["family"], (row["clustering_combined"], row["betweenness_combined"]),
                 xytext=(5,5), textcoords="offset points", fontsize=9)
plt.xlabel("Clustering Coefficient"); plt.ylabel("Betweenness Centrality")
plt.title("Local Cohesion vs Brokerage")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
```


    
![png](Pract_2_Florentine_Families_files/Pract_2_Florentine_Families_82_0.png)
    


## <font color="red"><b>Exercise 11</b></font>

1. Locate MEDICI in the clustering-versus-betweenness plot.
2. Is its neighborhood highly closed or relatively open?
3. Why can low/moderate local clustering coexist with high structural importance?
4. Explain how this combination can be interpreted in terms of brokerage.

## <font color="green"><b>Exercise 11</b></font>

1. The Medici is with a low clustering coefficient and a high betweenness centrality.
2. His neighbours are relatively open.
3. Because the strictural importance depends more in the brokerage than in clustering, is better to connect big diverse groups than closed groups, with diverse groups you can achive more actors.
4. The once with most brokerage have less clustering coefficient, and the once with less brokerage and less connections has more clustering coefficient.

## <font color="red"><b>Exercise 12 — Executive Summary</b></font>

Write an **executive summary of approximately 200–250 words** for a reader who is highly busy, and wants an overview.

## <font color="red"><b>Exercise 13 — Infographic</b></font>

Create a **one-page infographic** entitled:

### *Florentine Families: Networks, Brokerage and the Rise of the Medici*

Should provide a clear view of your findings
