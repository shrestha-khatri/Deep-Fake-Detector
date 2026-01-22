Deepfake Video Detection using Qdrant Vector Memory

Convolve 4.0 – Pan-IIT AI/ML Hackathon Submission

Theme

**Misinformation & Digital Trust**

## Problem Statement

The rapid advancement of generative AI has enabled the creation of highly realistic **deepfake videos**, posing severe risks to **digital trust**, **public safety**, and **information integrity**. Deepfakes are increasingly used to spread misinformation, impersonate individuals, and manipulate public opinion.

Most existing deepfake detection systems rely on opaque, black-box classifiers that provide little insight into why a video is flagged as fake. This lack of transparency limits real-world adoption, auditability, and trust.

---

##  Our Solution

<img width="1600" height="740" alt="image" src="https://github.com/user-attachments/assets/8b74b66f-8901-4d56-ba41-4b78dcca5029" />



This project presents an **explainable deepfake video detection system** that combines computer vision with **vector similarity search**, using **Qdrant** as a **long-term semantic memory**.

Instead of directly classifying videos, the system:

* Extracts facial embeddings from videos
* Stores embeddings from known real and fake videos in Qdrant
* Retrieves similar historical examples during inference
* Makes an **evidence-based decision** grounded in retrieved data

This design explicitly fulfills the **Search + Memory + Reasoning** requirements of Convolve 4.0.

---

## Why Qdrant Is the Core of This System

Qdrant is not used as simple storage.

It acts as:

* **Long-term memory** for known facial patterns
* **Semantic search engine** for high-dimensional embeddings
* **Reasoning backbone** for explainable predictions

Without Qdrant, this system would be a black-box classifier.
With Qdrant, it becomes a **transparent, auditable, decision-support system**.

---

## System Architecture (End-to-End)

**Overall Flow**

```
Video Upload
   ↓
Frame Extraction (OpenCV)
   ↓
Face Detection (MTCNN)
   ↓
Face Embedding Generation (ResNet18)
   ↓
Qdrant Vector Database (Memory)
   ↓
Top-K Similarity Search
   ↓
Evidence-Based Prediction (Real / Deepfake + Confidence)
```

---

## Training vs Inference

### Training Phase (Memory Ingestion)

Training in this system does **not** mean model fitting.
It means **building long-term memory**.

```
Real & Fake Videos
→ Frame Extraction
→ Face Detection
→ Embedding Generation
→ Store embeddings in Qdrant with labels
```

* Each video produces multiple facial embeddings
* Embeddings are stored with metadata (`real` / `fake`)
* Qdrant becomes the knowledge base of known patterns

---

### Inference Phase (Prediction)

```
New Video
→ Extract facial embeddings
→ Query Qdrant for similar embeddings
→ Analyze neighbor labels
→ Final prediction + confidence
```

The system reasons using **past knowledge**, not just raw input.

---

##  Multimodal Strategy

The project is inherently multimodal:

| Stage          | Modality                 |
| -------------- | ------------------------ |
| Input          | Video                    |
| Processing     | Image (frames & faces)   |
| Representation | Vector embeddings        |
| Memory         | Vector database (Qdrant) |

This allows robust similarity search and semantic reasoning.

---

##  Persistent Memory Across Sessions

* Facial embeddings persist across application restarts
* New videos can be added without retraining
* Knowledge grows continuously
* Enables long-term reasoning beyond a single prompt

---

##  Dataset

* **Total videos:** 106

  * 53 Real
  * 53 Deepfake
* **Source:** Open-source, publicly available videos
* **Processing:** Frame-level face extraction

Each video generates multiple embeddings, resulting in hundreds of vectors stored in Qdrant.

---

## Evidence-Based Output

Predictions are not hallucinated.

Each prediction is based on:

* Top-K nearest embeddings retrieved from Qdrant
* Known labels of retrieved embeddings
* Confidence computed from neighbor label distribution

This makes the system **transparent and explainable**.

---

## Limitations & Ethics

### Limitations

* Requires visible faces
* Performance drops for very low-resolution videos
* Not intended as forensic or legal proof

### Ethical Considerations

* No permanent storage of user-uploaded videos
* Decision-support system, not a final authority
* Dataset bias is acknowledged and documented

---

## Technology Stack

* Python
* OpenCV (video processing)
* PyTorch & TorchVision
* FaceNet-PyTorch (MTCNN)
* Qdrant Vector Database
* Streamlit (interactive UI)

---

##  How to Run the Project

### 1️ Clone the Repository

```bash
git clone https://github.com/shrestha-khatri/Deep-Fake-Detector.git
cd Deep-Fake-Detector
```

---

### 2️ Create Conda Environment

```bash
conda create -n deepfake python=3.9
conda activate deepfake
```

---

### 3️ Install Dependencies

```bash
pip install opencv-python torch torchvision facenet-pytorch qdrant-client streamlit pillow
```

---

### 4️ Start Qdrant (Docker Required)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Verify:

```
http://localhost:6333/dashboard
```

---

### 5️ Prepare Dataset

```
data/
└── train/
    ├── real/
    │   ├── real01.mp4
    │   └── ...
    └── fake/
        ├── fake01.mp4
        └── ...
```

---

### 6️ Initialize Qdrant Collection

```bash
python backend/init_qdrant.py
```

---

### 7️ Train (Ingest Data into Memory)

```bash
python backend/ingest_data.py
```

This step extracts faces, generates embeddings, and stores them in Qdrant.

---

### 8️ Run the Application

```bash
python -m streamlit run app.py
```

Open:

```
http://localhost:8501
```

Upload a video to get predictions.

---

<img width="1600" height="792" alt="image" src="https://github.com/user-attachments/assets/e472fac9-3fee-452f-9d45-df46ff787003" />


##  Societal Impact

This system contributes to:

* Combating misinformation
* Restoring trust in digital media
* Promoting explainable AI
* Supporting responsible AI deployment

---

##  Why This Project Stands Out

✔ Qdrant used as **core long-term memory**
✔ Explicit **search + memory + reasoning**
✔ Fully working end-to-end system
✔ Explainable predictions
✔ Real societal relevance

---

##  One-Line Summary

> An explainable deepfake detection system that leverages facial embeddings and Qdrant’s vector memory to deliver trustworthy, evidence-based predictions.

---


