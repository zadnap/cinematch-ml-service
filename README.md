---
title: Cinematch ML Service
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

## CineMatch - Movie Recommendation System

### Overview

- A movie recommendation web application (used to be) built as part of [The Odin Project](https://www.theodinproject.com/lessons/node-path-react-new-shopping-cart)
  curriculum. Now it serves as recommendation core for the application. See [CineMatch Client](https://github.com/zadnap/cinematch-client) and [CineMatch API](https://github.com/zadnap/cinematch-api) for more information.
- This project focuses on building a movie recommendation API by integrating MovieLens and TMDB data, implementing content-based and hybrid recommendation techniques, and designing a scalable backend service using Flask to deliver personalized movie suggestions.
- See the project in action: [CineMatch](https://cinematch-client.vercel.app).

### Installation & Usage

1. Clone repository

   ```bash
   git clone https://github.com/zadnap/cinematch-ml-service.git
   ```

2. Create virtual environment

   ```bash
   python3.11 -m venv venv
   ```

3. Activate virtual environment

- macOS / Linux:

  ```bash
  source venv/bin/activate
  ```

- Windows:

  ```bash
  venv\Scripts\activate
  ```

4. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

5. Create an .env file with content

   ```bash
   HF_TOKEN=<your_hf_token>
   REPO_ID=<your_hf_repo_id>
   CORS_ORIGINS=<web_service_url>
   WEB_API_URL=<web_service_url>
   TF_USE_LEGACY_KERAS=1
   ```

6. Run the server

   ```bash
   flask run
   ```

### Training & Uploading Artifacts

1. Prepare data

   ```bash
   python -m scripts/download_dataset.py
   ```

2. Train the model

   ```bash
   python -m ml.training.train
   ```

3. Upload artifacts to Hugging Face Hub

   ```bash
   python scripts/upload_artifacts.py
   ```
