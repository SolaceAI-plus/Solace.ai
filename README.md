## Solace.ai Vector data model training based on NLP-BERT model

### 1. Product positioning

#### 1.1  AI Multi-modal large model aggregation

Solace brings together a variety of state-of-the-art ai services and models to provide an efficient, cost-optimized AI solution designed to revolutionize data processing and intelligent services.  

#### 1.2  Data assets and decentralized data trading market

Data capitalization is one of the core concepts of the Solace.ai platform. Users can freely trade on the trading market based on the data deposited by the multi-modal large model platform. All data transactions made on the Solace.ai platform will be recorded on the blockchain. This distributed ledger technology ensures the immutability and transparency of transaction records, providing a foundation of trust for all parties involved.。 

#### 1.3  Digital Avatar

Based on the data deposited by the multi-modal large model platform and the data obtained from trading market transactions, Solace.ai creates a digital entity in the world of Web3 through the analysis of learning and behavior patterns based on user data. This concept transcends the limitations of real life, allowing users to experience the possibilities of self-extension and self-expression in the digital realm.

### 2. Model implementation

#### 2.1 Large model development

Solace's core strength lies in the ability to integrate different AI models to provide more accurate and faster data processing services. This strategy ensures that users get the best analysis results and decision support. Through intelligent algorithms, platforms are able to select the most cost-effective AI services to minimize costs. This enables users to enjoy high-quality AI processing power at a lower price, making AI technology more widespread and accessible.  

##### 2.1.1 Collecting Data

Big data model based on open source data model library huggingface pulls data multi-modal big data model, the main models are as follows:

Autoregressive: GPT2 Trasnformer-XL XLNet

Self-encoding: BERT ALBERT RoBERTa ELECTRA

SeqtoSeq: BART Pegasus T5

Retrieves the Dataset from the huggingface Hub

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

# train data
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print(raw_train_dataset.features)
```

##### 2.1.2 Data preprocessing

In order to solve real world problems with deep learning, we often start by preprocessing raw data, rather than those prepared in tensor format. Among the data analysis tools commonly used in Python, we usually use the 'pandas' package. Like many other extension packages in the vast Python ecosystem, pandas is compatible with tensors.
We first create a manual data set and store it in a CSV (comma-separated values) file... /data/house_tiny.csv '. Data stored in other formats can also be processed in a similar way. Next we write the data set in rows to a CSV file.

```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

To load the raw data set from the created CSV file, we import the pandas package and call the read_csv function. The dataset has four rows and three columns. Each line describes the number of rooms (" NumRooms "), the type of Alley (" alley "), and the Price of the house (" price ").

```python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

To deal with missing data, typical methods include interpolation, where the interpolation replaces the missing value with a substitute value, and deletion, where the deletion rule simply ignores the missing value. Here, we will consider interpolation.

Using the location index iloc, we divide the data into inputs and outputs, where the former is the first two columns of the data and the latter is the last column of the data. For values missing from inputs, we replace the "NaN" entry with the mean of the same column.

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

For category values or discrete values in inputs, we treat "NaN" as a category. Since the "Alley" column accepts only two types of category values "Pave" and "NaN", pandas can automatically convert this column to two columns "Alley_Pave" and "Alley_nan". If the alley type is Pave, Alley_Pave is set to 1, and Alley_nan is set to 0. Guilds that lack alley types set "Alley_Pave" and "Alley_nan" to 0 and 1, respectively.

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

Now that all entries in inputs and outputs are numeric types, they can be converted to tensor format.

```python
import warnings

warnings.filterwarnings(action='ignore')
import paddle

X, y = paddle.to_tensor(inputs.values), paddle.to_tensor(outputs.values)
X, y
```

##### 2.1.3 Model training

ZoKrates is the zkSNARKs toolbox on Ethereum. It helps you use verifiable calculations in Dapps, from specifying programs in high-level languages to generating proofs of calculations to validating those proofs using Solidity.
One particular family of ZKP is described as zero-knowledge concise non-interactive AR knowledge proofs, aka zkSNARK. zkSNARK is the most widely used zero-knowledge protocol, with anonymous cryptocurrency Zcash and smart contract platform Ethereum being notable early adopters.
After passing the contract to the validator, the proof can be checked. For example, with 'web3', the call would look like this:

```js
const accounts = await web3.eth.getAccounts();
const address = '0x456...'; // verifier contract address

let verifier = new web3.eth.Contract(abi, address, {
    from: accounts[0], // default from address
    gasPrice: '20000000000000'; // default gas price in wei
});

let result = await verifier.methods
    .verifyTx(proof.proof, proof.inputs)
    .call({ from: accounts[0] });

```

When generating R1CS constraints, very large numbers are often used, which makes it difficult for humans to read ZIR. To alleviate this situation, ZIR applies isomorphism when displaying field elements: they appear as members of the interval '[- (p-1)/2, (p-1)/2]'. In other words, use the following mapping: 

- Element '[0, (p-1)/2]' maps to itself
- Element '[(p + 1)/2, p-1]' maps to itself minus' p '

Therefore, instead of writing 'p-1' :

```js
21888242871839275222246405745257275088548364400416034343698204186575808495616
```

 In ZIR, we simply write:

```js
-1
```

To programmatically interact with compiled ZoKrates programs, ZoKrates supports the use of ABI to pass parameters.

To illustrate this, we'll use the following sample program:

```js
struct Bar {
    field a;
}

struct Foo {
    u8 a;
    Bar b;
}

def main(Foo foo, bool[2] bar, field num) -> field {
    return 42;
}
```

When a program is compiled, an ABI specification is generated that describes the program's interface.

In this example, the ABI specification is:

```json
{
   "inputs":[
      {
         "name":"foo",
         "public":true,
         "type":"struct",
         "components":{
            "name":"Foo",
            "members":[
               {
                  "name":"a",
                  "type":"u8"
               },
               {
                  "name":"b",
                  "type":"struct",
                  "components":{
                     "name":"Bar",
                     "members":[
                        {
                           "name":"a",
                           "type":"field"
                        }
                     ]
                  }
               }
            ]
         }
      },
      {
         "name":"bar",
         "public":true,
         "type":"array",
         "components":{
            "size":2,
            "type":"bool"
         }
      },
      {
         "name":"num",
         "public":true,
         "type":"field"
      }
   ],
   "output": {
     "type":"field"
   }
}
```

When executing the program, parameters can be passed as JSON objects of the following form:

```json
[
   {
      "a":"0x2a",
      "b":{
         "a":"42"
      }
   },
   [
      true,
      false
   ],
   "42"
]
```

Please note the following:

- Field elements are passed as JSON strings to support arbitrarily large numbers
- Unsigned integers are passed as JSON strings containing their hexadecimal representation
- The struct is passed as a JSON object, ignoring the struct name

##### 2.1.4 Model fine-tuning

Adapter-tuning: Adapter is a fairly simple and effective lightweight tuning method in the early days. Smaller neural network layers or modules are inserted into each layer of the pre-trained model. These newly inserted neural modules become adapters, and only these Adapter parameters are updated when the downstream task is fine-tuning. The number of parameters for Adapter tuning is approximately 3.6% of the LM parameters.

Prefix/P-v1/P-v2 Tuning: Add K additional trainable prefix Tokens in the input or hidden layer of the model, and then update only those prefix parameters. 

-  prefix-tuning :

  Construct a set of task-related virtual tokens as a Prefix before entering the token; Then, during training, only the Prefix part of the parameter is updated, while the rest of the PLM parameters are fixed. In order to prevent the unstable training and performance degradation caused by the direct update of the parameters of Prefix, the MLP structure is added in front of the Prefix layer. After the training is completed, only the parameters of Prefix are retained. The size of Prefix Tuning parameters is about 0.1% of the overall size of the LM model.

-  p-tuning v2 :

  This is very similar to prefix-tuning. The changes are as follows:

  - Remove heavy parameterized encoders. Previous methods used reparameterization to improve training speed and robustness (e.g. MLP in Prefix Tuning, LSTM in P-Tuning). In P-tuning v2, the authors found little improvement in reparameterization, especially for smaller models, while also affecting model performance.
  - Use different prompt lengths for different tasks. Prompt length plays a key role in hyperparameter search of prompt optimization methods. In the experiment, it is found that different comprehension tasks usually use different prompt lengths to achieve their best performance, which is consistent with the finding in Prefix-Tuning. Different text generation tasks may have different optimal prompt lengths.
  - Introduce multitasking learning. Pre-train the Prompt on multiple tasks before adapting it to downstream tasks. Multitasking learning is optional to our approach, but can be quite helpful. On the one hand, the randomness of the continuous prompts creates difficulties for optimization, which can be mitigated by more training data or unsupervised pre-training related to the task; Continuous prompts, on the other hand, are perfect carriers of task-specific knowledge across tasks and datasets. Our experiments show that multi-task learning can be a useful supplement to P-tuning v2 in some difficult sequential tasks.

lora (Low Rank Adaptive) is one of the broadest and most effective techniques available for efficiently training large models of custom languages.

The basic principle is to freeze the weight parameters of the pre-trained model, and in the case of freezing the parameters of the original model, by adding additional network layers to the model, and only train these network layer parameters.

Based on transformer structure, lora generally only fine-adjusts the self-attention parts of each layer, namely Wq, Wk, and Wv

lora fine-tuning some parameter Settings:

- Hyperparameter r：It determines the rank and dimension of the lora matrix and directly affects the complexity and capacity of the model. A higher r means greater expressiveness, but may lead to overfitting; Lower r can reduce overfitting, but at the cost of reduced expressiveness. As a general rule of thumb, the more diverse the tasks in the data set, the larger the r will be set.
- Hyperparameter alpha：Scaling scale, the larger the value of alpha, the greater the weight impact of lora, and lower alpha will reduce its impact, making the model more dependent on the original parameters. Adjusting alpha helps to find a balance between fitting the data and preventing overfitting by regularizing the model. As a rule of thumb, when fine-tuning LLM, the choice of alpha is usually twice that of rank.
- Hyperparameter alpha-dropout：dropout coefficient in lora fine tuning

qlora builds on lora, introducing 4-bit quantization, 4-bit NormalFloat data types, double-quantization, and a paging optimizer to further reduce memory usage.

dora is an improvement or extension of lora. dora is a decomposition of a pre-trained weight matrix into an amplitude vector m and a direction matrix v.

The motivation for developing this approach is based on analyzing and comparing the differences between lora and the fully fine-tuned model. The paper found that lora can increase or decrease amplitude and direction updates proportionally, but appears to lack the ability to make subtle directional changes like full fine-tuning. Therefore, the decoupling of amplitude and direction components is proposed. lora is applied to the directional component (while allowing the amplitude component to be trained separately).

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F


# This layer is dropped into your pre-trained PyTorch model where nn.Linear is used
class DoRALayer(nn.Module):
    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
        super().__init__()

        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.Tensor(d_out), requires_grad=False)

        # m = Magnitude column-wise across output dimension
        self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))
        
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = nn.Parameter(torch.randn(d_out, rank)*std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_in))

    def forward(self, x):
        lora = torch.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return F.linear(x, calc_weights, self.bias)
```

##### 2.1.5 Deploy Query inference

In fact, using trained models to make predictions about new data is a more technical term in machine learning engineering called "Inference." The following diagram details the training and reasoning process of a neural network model. The prediction of new input data by a neural network built through a training set is inference.

![](.\images\document-uid214893labid7506timestamp1553237955368.png)

In general, reasoning can be divided into static reasoning and dynamic reasoning.

Static reasoning is well understood by centrally inferring batch data and storing the results in a data table or database. When there is a need, then directly through the query to obtain inference results.

Dynamic inference generally means that we deploy the model to the server. When needed, the prediction returned by the model is obtained by sending a request to the server. Unlike static reasoning, the process of dynamic reasoning is calculated in real time, while static reasoning is handled in batches in advance.

Of course, both static and dynamic reasoning have advantages and disadvantages. Static inference is suitable for processing large amounts of data, because dynamic inference is very time consuming for large amounts of data. However, static inference cannot be updated in real time, while the result of dynamic inference is an immediate calculation result.

Making predictions about new data is actually similar to the process of static reasoning. All you need to do is use the 'predict' operation provided by scikit-learn. Dynamic inference requires deploying the scikit-learn model using a RESTful API and completing it. To deploy the scikit-learn model, of course, you need to complete the model training first.

```shell
wget -nc "https://labfile.oss.aliyuncs.com/courses/2616/seaborn-data.zip"
!unzip seaborn-data.zip -d ~/
```

```python
from seaborn import load_dataset
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

df = load_dataset("titanic")
df
```

As you can see, the dataset contains a total of 15 columns with a total of 891 samples. We selected three characteristics, including passenger location pclass, sex, and embarkation port embarked, and used alive or not as the target value.

```python
X = df[["pclass", "sex", "embarked"]]
y = df["alive"]
```

 Before training, we first perform unique thermal coding on the feature data

```python
X = pd.get_dummies(X)
X.head()
```

Next, it's time to start training. Here the random forest approach is used to model and cross-validation is used to see how the model performs. 

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
np.mean(cross_val_score(model, X, y, cv=5))
```

Cross-validation shows that the classification accuracy of the model is about 81%.

To facilitate the deployment of the model, we need to store the trained model. Here, you can use the 'sklearn.externals.joblib' provided by scikit-learn to save the model as a '.pkl 'binary.

```python
from sklearn.externals import joblib

model.fit(X, y)
joblib.dump(model, "titanic.pkl")
```

Now that you have the model file, you can deploy the model. This is implemented with Flask. Flask is Python's famous Web application framework that can be used to build a RESTful API.

```python
%%writefile predict.py
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    columns_onehot = ["pclass", "sex_female", "sex_male",
                      "embarked_C", "embarked_Q", "embarked_S"]
    query = pd.get_dummies(query_df).reindex(
        columns=columns_onehot, fill_value=0)
    clf = joblib.load("titanic.pkl")
    prediction = clf.predict(query)
    return jsonify({"prediction": list(prediction)})
```

First, an app created by Flask here can send new data to the server via POST in an HTTP method and return the inference result. We define new data to be sent in JSON format and can send multiple pieces at the same time.

```python
import json

sample = [{"pclass": 1, "sex": "male", "embarked": "C"},
          {"pclass": 2, "sex": "female", "embarked": "S"}]
sample_df = pd.read_json(json.dumps(sample))
sample_df

pd.get_dummies(sample_df).reindex(columns=X.columns, fill_value=0)
```

Due to the limitations of the Jupyter Notebook environment, you can only start Flask as a child process here, otherwise you cannot request the API behind the Jupyter Notebook.

```python
import subprocess as sp

server = sp.Popen("FLASK_APP=predict.py flask run", shell=True)
server
```

By now, our Flask should be running, and you can't see the standard output because you're using the child process. In fact, the default local link and port on which the application runs is' http://127.0.0.1:5000 '.

Next, we can use HTTP POST to dynamically reason about the new data.

```python
import requests

requests.post(url="http://127.0.0.1:5000",
              json=sample).content
```

Get test results: 

```python
sample = [{"pclass": 1, "sex": "male", "embarked": "C"},
          {"pclass": 2, "sex": "female", "embarked": "S"},
          {"pclass": 3, "sex": "male", "embarked": "Q"},
          {"pclass": 3, "sex": "female", "embarked": "S"}]

requests.post(url="https://ai.solana.plus", json=sample).content
```

#### 2.2 Decentralized market

Data capitalization is one of the core concepts of the Solace.ai platform. We not only allow users to store their data, but also provide mechanisms for users to set prices and trade their data. 

Solace creates an open marketplace where users can freely price and trade data based on market supply and demand. This market mechanism ensures the fairness and transparency of data prices.

All data transactions made on the Solace.ai platform will be recorded on the blockchain. This distributed ledger technology ensures the immutability and transparency of transaction records, providing a foundation of trust for all parties involved.

##### 2.2.1 IPFS

Interstellar file system (IPFS) is a set of composable point-to-point protocol, used in file system in decentralized addressing, routing, and addressing [content] (https://docs.ipfs.tech/concepts/glossary/#content-addressing) data. Many popular Web3 projects are built on IPFS, we use IPFS as the underlying data storage, relying on IPFS protocol self-built nodes and public chain nodes open, to achieve the distribution of user data stored on different nodes on the blockchain.

Bitswap is a core module of IPFS that is used to exchange data blocks. It directs block requests and sending between other peers in the network. Bitswap is a message-based protocol. Bitswap has a JavaScript implementation, we use ipfs-bitswap.js to implement this core module.

```js
<script src="https://unpkg.com/ipfs-bitswap/dist/index.min.js"></script>
```

Bitswap has two main tasks:

- Gets the block requested by the client from the network.
- sends the blocks it owns to other peers that need them.

When nodes running Bitswap want to get files, they send 'want-lists' to their peers. A 'want-list 'is the CID list of the blocks that the peer wants to receive. Each node remembers which blocks its peers want. Each time a node receives a block, it checks to see if any of its peers want the block, and if so, it sends it to its peers.

Here is a simplified version of 'want-list' ：

```js
Want-list {
  QmZtmD2qt6fJot32nabSP3CUjicnypEBz7bHVDhPQt9aAy, WANT,
  QmTudJSaoKxtbEnTddJ9vh8hbN84ZLVvD5pNpUaSbxwGoa, WANT,
  ...
}
```

To find a peer that owns a file, a node running the Bitswap protocol first sends a request called "Want to own" to all of its connected peers. This "want to have" request contains the CID of the root block of the file (the root block is at the top of the DAG of the blocks that make up the file). The peer that owns the root block sends a "own" response and is added to the session. The peer that does not have the block sends a "do not own" response. Bitswap builds a map showing which nodes own each block and which nodes don't.

![](.\images\diagram-of-the-want-have-want-block-process.6ef862a2.png)

Bitswap sends the desired block to the peers who own the block, and those peers respond with the block itself. If no peer has a root block, Bitswap queries the Distributed hash table (DHT) and asks who can provide the root block.

A distributed hash table (DHT) is a distributed system that maps keys to values. In IPFS, DHT is used as a basic component of the content routing system, acting as an intersection between the directory and navigation systems. It maps what the user is looking for to a peer that stores the matching content. Think of it as a giant table for storing * who * owns * what * data. There are three types of key-value pairs that use DHT mapping:

| Type       | Aim                                                         | Users                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Provider record | Map data identifiers (that is, multiple hashes) to peers that claim to own the content and are willing to provide it to you. | - Find content by IPFS - Find other members of a PubSub topic by pubsub's IPNS. |
| IPNS record  | Map IPNS keys (that is, the hash of the public key) to IPNS records (that is, signed and versioned Pointers to similar paths /ipfs/bafyxyz...) | - IPNS                                                       |
| Peer record   | Map Peerids to a set of multiple addresses that can be contacted by peers                 | - IPFS is when we know that there is a peer with content, but do not know its address. - Manual connection (e.g. ipfs swarm connect /p2p/Qmxyz...) |

The Kademlia algorithm, which has been around for a while, aims to build DHTS on top of three system parameters:

1. Address * space * is the way in which all network peers can be uniquely identified. In IPFS, this is all the numbers from '0' to '2^256-1'.
2. A metric used to sort peers in the address space, so that all peers can be visualized as a line in order from smallest to largest. IPFS treats * it * as an integer between SHA256(PeerID).   0   2 ^ 256-1
3. Projection will take and calculate the location of the peer in the address space that is best suited to store the record. IPFS* Uses *. record key``SHA256(Record Key)

With this address space and peer sort metric, we can search the network like a sort list. Specifically, we can turn the system into something like a jump list, where one peer knows the other peers that are approximately away from it. This will enable us to find time '1,2,4,8... ') Pairs of number relationships within the time search list. O(log(N))

Unlike jump lists, Kademlia is a bit unstable because peers can join, leave, and rejoin the network at any time. In order to cope with the unstable nature of the system, the Kademlia peer not only retains a link to the peer that is further away from it. ` 1,2,4,8... Instead, it will keep up to 20 K links for distances that are multiples of 2. In IPFS 'K = 20'. For example, a peer does not keep a single link with a distance of 128, but 20 links with a distance of 65 to 128.

The choice of the network range parameter 'K' is not arbitrary. It depends on the average * churn * 'K' observed in the network and how often the network republishes the information. System parameters, for example, are calculated to maximize the probability that the network will remain connected and not lose any data, while maintaining the latency required for queries and assuming that the average churn rate observations remain constant. These system and network parameters drive decisions in Kademlia's two main components: the routing table, which keeps track of all the links in the network, and the lookup algorithm, which determines how to traverse those links to store and retrieve data.

The IPFS Gateway is a web-based service that takes content from the IPFS network (private or public cluster supported by Amino DHT) and serves the content over HTTP, allowing browsers, tools, and software that are not compatible with IPFS to benefit from content addressing.

We use IPFS Kubo to build the IPFS gateway.

```yaml
version: '3.8'
services:
  kubo:
    image: ipfs/kubo:latest
    container_name: kubo
    hostname: kubo
    ports:
      - 4001:4001
      - 5001:5001
      - 6060:8080
    restart: always
```

When a client request for a CID reaches the IPFS gateway, the gateway first checks whether the CID is cached locally. At this point, one of the following things happens:

- If the CID is cached locally, the gateway will respond with the content referenced by the CID, and the lifecycle is complete.
- If the CID is not in the local cache, the gateway will try to retrieve it from the network.

 The CID search process consists of two parts: content discovery/routing and content retrieval: 

1. In the content discovery/routing step, the gateway determines the provider location; To find the * location of the data specified by CID: *
   - Asks whether the peer directly connected to it has the data specified by the CID.
   - Queries the DHT to obtain the ID and network address of the peer with the CID specified data.
2. Next, the gateway performs content retrieval, which can be divided into the following steps:
   1. The gateway connects to the provider.
   2. The gateway obtains the CID content.
   3. The gateway streams the content to the client.

In order to ensure the security of data, we restrict the authentication of the gateway's request. The interface provided by the gateway is authenticated through our sso-auth(Authentication Service) reverse proxy to the gateway request. The reverse proxy can also preserve the original IPFS API calls, enabling the gateway to accommodate all IPFS SDKS and toolkits. 

![](.\images\public-authed-gateway.59d1f96a.png)

Interplanetary Network Indexers (IPNI) (opens in new window), also known as Network indexers, indexers, and IPNI, are capable of quickly and efficiently searching content-addressable data available on Interplanetary file systems (IPFS) and Filecoin networks.

To support the retrieval of unsealed Filecoin and IPFS fixed data at a speed comparable to that of a CDN, a reliable distributed index of all data and its associated peers must be maintained near the lookup point. This is necessary to complete lookups that cannot be done quickly using DHT.

With this in mind, the network indexer was created as an alternative content routing system to Kadmelia DHT used by IPFS. While DHT is a key component of the IPFS ecosystem, IPNI can use Lotus and Boost to support content routing at a much larger scale and faster pace.

 Indexers provide several benefits to IPFS, including:

- ** Faster data retrieval ** : By maintaining additional layers of information on top of DHT, indexers can help speed up data location and retrieval in IPFS.
- **Reduced resource consumption**：Indexers can help reduce the bandwidth and processing power required to locate and retrieve data, thereby improving the performance of individual nodes and the entire network.
- **Increased Scalability **：With indexers, IPFS can better handle user base and data volume growth, allowing it to scale more efficiently and support larger networks.

Indexers work in conjunction with existing DHTS to improve data location and retrieval in IPFS. It maintains an up-to-date index of web content that has been published to it and provides additional layers of information that can be used to quickly locate and retrieve data.

When a user searches for data using CID or Doha, the indexer is first queried. If data is found in the index, the user will connect directly to the node hosting the data, which speeds up retrieval. If the data is not found in the index, the user falls back to a traditional DHT-based search to ensure that the data can still be found even if it is not in the indexer.

By providing this additional layer of information, indexers help speed up data location and retrieval, reduce resource consumption, and improve the overall scalability of IPFS.

Content addressing in IPFS is inherently immutable: when we add a file to IPFS, it creates a hash from the data and builds a CID out of it. Changing a file changes its hash, which changes the CID used as an address.

However, in many cases, content-addressing data needs to be updated regularly, for example, data that users have trained through big data models will become more accurate over time. Sharing a new CID is impractical. With variable Pointers, you can share the address of a pointer once and update the pointer to a new CID each time you publish a change.

The Interstellar Name System (IPNS) is a system for creating a variable pointer to a CID (called a name or IPNS name). IPNS names can be thought of as links that can be updated over time, while retaining the verifiability of content addressing.

The name in IPNS is the hash of the public key. It is associated with the IPNS record (opens in new window) and contains the content path to which it is linked (/ipfs/CID) along with other information such as the expiration date, version number, and cryptographic signature signed by the corresponding private key. Private key holders can sign and publish new records at any time.

For example, here is the IPNS name represented by CIDv1 of the public key:

```js
k51qzi5uqu5dlvj2baxnqndepeb86cbk3ng7n3i46uzyxzyqj2xjonzllnv0v8
```

IPNS records can point to immutable or mutable paths. The meaning behind CID used in paths depends on the namespace used:

- /ipfs/<cid>–  Immutable content on IPFS (because CID contains multiple hashes)
- /ipns/<cid-of-libp2p-key>– Variable encrypted IPNS name which corresponds to the libp2p public key.

```js
IPFS = immutable *Pointer => content
IPNS = **Pointer => content
```

IPNS names are essentially Pointers (IPNS names) to Pointers (IPFS Cids), and IPFS Cids are immutable (because they are derived from content) to Pointers to content.

##### 2.2.2 libp2p

libp2p (short for "library peer-to-peer") is a peer-to-peer (P2P) networking framework that can be used to develop P2P applications. It consists of a set of protocols, specifications, and libraries that facilitate P2P communication between network participants (i.e., peers).

We use the libp2p framework to communicate between IPFS self-built nodes, ensuring that data synchronization/data redundancy is not affected when the network environment is poor between each independent node and node.

 We choose libp2p mainly because it has the following advantages:

- Flexible addressing
- Traffic agnosticism
- Customizable security
- Peer status.
- Peer routing
- NAT convenience

In addition, libp2p supports the Pub/Sub model. Publish/subscribe (PubSub) is a messaging model in which peers gather around topics of interest and exchange messages accordingly. In IPFS, libp2p's PubSub system allows peers to easily join and communicate on topics in real time, providing a scalable and fault-tolerant solution for P2P communication.

A key challenge for P2P-based PubSub systems is to deliver messages quickly and efficiently to all subscribers, especially in large dynamic networks. To overcome these challenges, IPFS has adopted libp2p's "GossipSub" protocol, which operates by "gossiping" with peers about the messages they receive, enabling an efficient messaging network.

##### 2.2.3 IPFS-Encrypt

IPFS does not support data encryption by default. In order to solve the data security problem, we used IPFS-Encrypt to encrypt/decrypt the data.

Ipfs-encrypt is a Node.js module for uploading and downloading encrypted folders from IPFS using AES-256-CBC encryption.

Encrypt data and upload it to IPFS 

```js
import { uploadEncryptionIpfs } from "ipfs-encrypted";
const token = "my_web3_storage_token";
const folderPath = "/path/to/folder";
const password = "my_password";

uploadEncryptionIpfs(token, folderPath, password)
  .then((cid) => console.log(`Folder uploaded and encrypted with CID ${cid}`))
  .catch((error) => console.error(`Error: ${error.message}`));
```

Download data from IPFS and decrypt it

```js
import { decryptFolderIpfs } from "ipfs-encrypted";
const token = "my_web3_storage_token";
const cid = "Qm1234abcd";
const password = "my_password";
const downloadLocation = "/path/to/folder";

decryptFolderIpfs(token, cid, password, downloadLocation)
  .then((folderPath) =>
    console.log(`Folder decrypted and saved to ${folderPath}`)
  )
  .catch((error) => console.error(`Error: ${error.message}`));
```

#### 2.3 Digital life

Through the analysis of learning and behavior patterns based on user data, Solace.ai creates digital entities in a Web3 world. This concept transcends the limitations of real life, allowing users to experience the possibilities of self-extension and self-expression in the digital realm. Solace.ai through the deep expansion of digital life, with cutting-edge AI technology and the optimization of a large number of user data, to help users truly become themselves in the digital world, show themselves. We are committed to creating a unique digital representation of each user. This digital representation is not only a simple data model, but also a symbol of your authentic expression and unique experience in the digital realm.

##### 2.3.1 Perceptrons and Artificial neural networks

Artificial neural network (Ann) is a kind of machine learning algorithm developed earlier and very commonly. Because of its characteristics of imitating the work of human neurons, artificial neural networks are given high expectations in both supervised and unsupervised learning fields. At present, convolutional neural networks and recurrent neural networks developed from traditional artificial neural networks have become the cornerstones of deep learning.

Because the data sets facing deep learning are very large, the traditional gradient descent is inefficient on the full data set. But SGD only computes one sample at a time, so it is very fast and suitable for updating models online.

```python
from sklearn.utils import shuffle


def perceptron_sgd(X, Y, alpha, epochs):
    w = np.zeros(len(X[0])) 
    b = np.zeros(1)

    for t in range(epochs):
        for i, x in enumerate(X):
            if ((np.dot(X[i], w) + b) * Y[i]) <= 0:
                w = w + alpha * X[i] * Y[i]
                b = b + alpha * Y[i]

    return w, b
```

We realized 1 forward → reverse transfer of a single sample in the neural network, and used gradient descent to complete 1 weight update.

```python
class NeuralNetwork:
    def __init__(self, X, y, lr):
        self.input_layer = X
        self.W1 = np.random.rand(self.input_layer.shape[1], 3)
        self.W2 = np.random.rand(3, 1)
        self.y = y
        self.lr = lr
        self.output_layer = np.zeros(self.y.shape)

    def forward(self):
        self.hidden_layer = sigmoid(np.dot(self.input_layer, self.W1))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.W2))

    def backward(self):
        d_W2 = np.dot(
            self.hidden_layer.T,
            (
                2
                * (self.output_layer - self.y)
                * sigmoid_derivative(np.dot(self.hidden_layer, self.W2))
            ),
        )

        d_W1 = np.dot(
            self.input_layer.T,
            (
                np.dot(
                    2
                    * (self.output_layer - self.y)
                    * sigmoid_derivative(np.dot(self.hidden_layer, self.W2)),
                    self.W2.T,
                )
                * sigmoid_derivative(np.dot(self.input_layer, self.W1))
            ),
        )

        self.W1 -= self.lr * d_W1
        self.W2 -= self.lr * d_W2
```

Next, we input it into the network and iterate 100 times:

```python
nn = NeuralNetwork(X, y, lr=0.001) 
loss_list = [] 

for i in range(100):
    nn.forward() 
    nn.backward() 
    loss = np.sum((y - nn.output_layer) ** 2) 
    loss_list.append(loss)

print("final loss:", loss)
plt.plot(loss_list) 
```

```python
final loss: 133.26148346146633
```

```python
[<matplotlib.lines.Line2D at 0x156b4e470>]
```

![](.\images\e654ab8321a17ad2b7332d795099f16ec1dca6f5177f68c2c8dc688366207125.png)

As can be seen, the loss function gradually decreases and approaches convergence, and the change curve is much smoother than the perceptron calculation. However, because we have removed the intercept term and the network structure is too simple, the convergence is not ideal. In addition, it is important to note that because the weights are randomly initialized, the results of multiple runs will be different.